"""Isolated subprocess client for the pinned standalone vLLM worker."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from heterospawn.backends.vllm_rollout.models import (
    VllmSamplingConfig,
    VllmWorker,
    VllmWorkerResult,
    VllmWorkerRuntime,
    VllmWorkerSpec,
)
from heterospawn.errors import ConfigurationError, RolloutServiceError

_PASSTHROUGH_ENVIRONMENT = (
    "CUDA_HOME",
    "LANG",
    "LD_LIBRARY_PATH",
    "PATH",
    "TMPDIR",
)
_OPEN_UNIX_CONNECTION_ATTRIBUTE = "open_unix_connection"


@dataclass
class _PendingGeneration:
    prompt_ids: tuple[int, ...]
    sampling: VllmSamplingConfig
    future: asyncio.Future[VllmWorkerResult]


class SubprocessVllmWorkerFactory:
    """Launch workers without importing vLLM into the training process."""

    async def create(self, spec: VllmWorkerSpec) -> VllmWorker:
        if sys.platform == "win32":
            raise ConfigurationError("standalone vLLM workers require a Linux host")

        launch_id = uuid.uuid4().hex
        launch_dir = spec.config.runtime_dir / _safe_name(spec.deployment.deployment_id)
        launch_dir = launch_dir / launch_id
        socket_path = Path("/tmp") / f"heterospawn-vllm-{launch_id}.sock"
        spec_path = launch_dir / "worker-spec.json"
        log_path = launch_dir / "worker.log"
        await asyncio.to_thread(_prepare_launch_files, launch_dir, spec_path, spec)
        log_handle = await asyncio.to_thread(log_path.open, "ab", buffering=0)
        environment = _worker_environment(
            os.environ,
            spec.deployment.cuda_device,
            home=launch_dir,
        )
        try:
            process = await asyncio.create_subprocess_exec(
                str(spec.config.python_executable),
                "-m",
                "heterospawn.backends.vllm_rollout.worker",
                "--spec",
                str(spec_path),
                "--socket",
                str(socket_path),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=log_handle,
                stderr=log_handle,
                env=environment,
                start_new_session=True,
            )
        except Exception:
            log_handle.close()
            raise

        worker = SubprocessVllmWorker(
            spec=spec,
            process=process,
            socket_path=socket_path,
            log_path=log_path,
            log_handle=log_handle,
        )
        try:
            await worker.wait_until_ready()
        except Exception:
            await worker.close()
            raise
        return worker


class SubprocessVllmWorker:
    """Batches concurrent requests and exchanges exact-token JSON over a Unix socket."""

    def __init__(
        self,
        *,
        spec: VllmWorkerSpec,
        process: asyncio.subprocess.Process,
        socket_path: Path,
        log_path: Path,
        log_handle: Any,
    ) -> None:
        self._spec = spec
        self._process = process
        self._socket_path = socket_path
        self._log_path = log_path
        self._log_handle = log_handle
        self._queue: asyncio.Queue[_PendingGeneration] = asyncio.Queue()
        self._closed = False
        self._close_lock = asyncio.Lock()
        self._batch_task: asyncio.Task[None] | None = None

    @property
    def adapter_digest(self) -> str:
        return self._spec.artifact.artifact_digest

    async def wait_until_ready(self) -> None:
        deadline = asyncio.get_running_loop().time() + self._spec.config.startup_timeout_seconds
        last_error: Exception | None = None
        while asyncio.get_running_loop().time() < deadline:
            if self._process.returncode is not None:
                raise RolloutServiceError(
                    f"vLLM worker exited during startup; inspect {self._log_path}"
                )
            try:
                response = await self._rpc({"operation": "health"}, timeout_seconds=1.0)
                if response.get("adapter_digest") != self.adapter_digest:
                    raise RolloutServiceError("vLLM health check reported another adapter digest")
                self._batch_task = asyncio.create_task(
                    self._batch_loop(),
                    name=f"vllm-batch-{self._spec.deployment.deployment_id}",
                )
                return
            except (OSError, TimeoutError, RolloutServiceError) as error:
                last_error = error
                await asyncio.sleep(0.1)
        raise RolloutServiceError(
            f"vLLM worker startup timed out; inspect {self._log_path}"
        ) from last_error

    async def generate(
        self,
        prompt_ids: tuple[int, ...],
        sampling: VllmSamplingConfig,
    ) -> VllmWorkerResult:
        if self._closed or self._batch_task is None or self._batch_task.done():
            raise RolloutServiceError("vLLM worker is unavailable")
        future: asyncio.Future[VllmWorkerResult] = asyncio.get_running_loop().create_future()
        await self._queue.put(
            _PendingGeneration(
                prompt_ids=prompt_ids,
                sampling=sampling,
                future=future,
            )
        )
        return await future

    async def runtime_metrics(self) -> VllmWorkerRuntime:
        if self._closed:
            raise RolloutServiceError("vLLM worker is unavailable")
        response = await self._rpc({"operation": "health"}, timeout_seconds=5.0)
        return VllmWorkerRuntime.model_validate_json(
            json.dumps(
                response.get("runtime"),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            if self._batch_task is not None:
                self._batch_task.cancel()
                await asyncio.gather(self._batch_task, return_exceptions=True)
            self._reject_queued(RolloutServiceError("vLLM worker was closed"))
            if self._process.returncode is None:
                try:
                    await self._rpc(
                        {"operation": "shutdown"},
                        timeout_seconds=self._spec.config.shutdown_timeout_seconds,
                    )
                except Exception:
                    self._process.terminate()
                try:
                    await asyncio.wait_for(
                        self._process.wait(),
                        timeout=self._spec.config.shutdown_timeout_seconds,
                    )
                except TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            self._log_handle.close()
            await asyncio.to_thread(self._socket_path.unlink, missing_ok=True)

    async def _batch_loop(self) -> None:
        try:
            while True:
                first = await self._queue.get()
                batch = [first]
                if self._spec.config.batch_wait_ms:
                    await asyncio.sleep(self._spec.config.batch_wait_ms / 1000)
                while len(batch) < self._spec.config.max_num_seqs:
                    try:
                        batch.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                active = [item for item in batch if not item.future.cancelled()]
                if not active:
                    continue
                try:
                    payload = {
                        "operation": "generate_batch",
                        "requests": [
                            {
                                "prompt_ids": item.prompt_ids,
                                "sampling": item.sampling.model_dump(mode="json"),
                            }
                            for item in active
                        ],
                    }
                    response = await self._rpc(payload, timeout_seconds=None)
                    raw_results = response.get("results")
                    if not isinstance(raw_results, list) or len(raw_results) != len(active):
                        raise RolloutServiceError("vLLM worker returned an invalid batch")
                    for item, raw_result in zip(active, raw_results, strict=True):
                        if not item.future.done():
                            item.future.set_result(
                                VllmWorkerResult.model_validate_json(
                                    json.dumps(
                                        raw_result,
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                    )
                                )
                            )
                except Exception as error:
                    for item in active:
                        if not item.future.done():
                            item.future.set_exception(error)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._reject_queued(error)
            raise

    async def _rpc(
        self,
        payload: Mapping[str, object],
        *,
        timeout_seconds: float | None,
    ) -> dict[str, Any]:
        async def exchange() -> dict[str, Any]:
            open_unix_connection = cast(
                Any,
                getattr(asyncio, _OPEN_UNIX_CONNECTION_ATTRIBUTE),
            )
            reader, writer = await open_unix_connection(path=str(self._socket_path))
            try:
                writer.write(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
                )
                await writer.drain()
                line = await reader.readline()
                if not line:
                    raise RolloutServiceError("vLLM worker closed the RPC connection")
                response: object = json.loads(line)
                if not isinstance(response, dict):
                    raise RolloutServiceError("vLLM worker returned a non-object response")
                if response.get("ok") is not True:
                    error_code = response.get("error_code", "worker_error")
                    raise RolloutServiceError(f"vLLM worker RPC failed: {error_code}")
                return response
            except json.JSONDecodeError as error:
                raise RolloutServiceError("vLLM worker returned invalid JSON") from error
            finally:
                writer.close()
                await writer.wait_closed()

        if timeout_seconds is None:
            return await exchange()
        return await asyncio.wait_for(exchange(), timeout=timeout_seconds)

    def _reject_queued(self, error: BaseException) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if not item.future.done():
                item.future.set_exception(error)


def _prepare_launch_files(
    launch_dir: Path,
    spec_path: Path,
    spec: VllmWorkerSpec,
) -> None:
    launch_dir.mkdir(parents=True, exist_ok=False)
    spec_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")


def _worker_environment(
    source: Mapping[str, str],
    cuda_device: str,
    *,
    home: Path,
) -> dict[str, str]:
    environment = {name: source[name] for name in _PASSTHROUGH_ENVIRONMENT if name in source}
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": cuda_device,
            "HF_HUB_OFFLINE": "1",
            "HOME": str(home),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_ATTENTION_BACKEND": "XFORMERS",
            "VLLM_USE_V1": "0",
        }
    )
    return environment


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)
