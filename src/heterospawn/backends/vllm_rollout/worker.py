"""Entrypoint for an isolated vLLM V0 + XFormers rollout process."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.metadata
import json
import traceback
from pathlib import Path
from typing import Any, Literal, cast

from heterospawn.backends.vllm_rollout.models import (
    VllmSamplingConfig,
    VllmWorkerResult,
    VllmWorkerRuntime,
    VllmWorkerSpec,
    rollout_artifact_path,
)

_START_UNIX_SERVER_ATTRIBUTE = "start_unix_server"


class _Runtime:
    def __init__(self, spec: VllmWorkerSpec) -> None:
        vllm = importlib.import_module("vllm")
        lora_request_module = importlib.import_module("vllm.lora.request")
        llm_type = cast(Any, vllm.LLM)
        lora_request_type = cast(Any, lora_request_module.LoRARequest)

        self._spec = spec
        self._artifact_path = rollout_artifact_path(spec.artifact)
        self._llm = llm_type(
            model=str(spec.config.model_path),
            tokenizer=str(spec.config.model_path),
            dtype=spec.config.dtype,
            enforce_eager=spec.config.enforce_eager,
            gpu_memory_utilization=spec.config.gpu_memory_utilization,
            max_model_len=spec.config.max_model_len,
            max_num_seqs=spec.config.max_num_seqs,
            enable_lora=True,
            max_lora_rank=spec.config.max_lora_rank,
            trust_remote_code=False,
            disable_log_stats=True,
        )
        self._lora_request = lora_request_type(
            spec.deployment.deployment_id,
            1,
            str(self._artifact_path),
        )
        tokenizer = self._llm.get_tokenizer()
        self._eos_token_id = tokenizer.eos_token_id

    @property
    def adapter_digest(self) -> str:
        return self._spec.artifact.artifact_digest

    def runtime_metrics(self) -> VllmWorkerRuntime:
        torch = importlib.import_module("torch")

        free_memory, total_memory = torch.cuda.mem_get_info()
        return VllmWorkerRuntime(
            gpu_name=torch.cuda.get_device_name(),
            total_memory_bytes=int(total_memory),
            device_used_bytes_at_query=int(total_memory - free_memory),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
            versions=(
                ("numpy", importlib.metadata.version("numpy")),
                ("torch", importlib.metadata.version("torch")),
                ("transformers", importlib.metadata.version("transformers")),
                ("vllm", importlib.metadata.version("vllm")),
                ("xformers", importlib.metadata.version("xformers")),
            ),
        )

    def generate_batch(self, requests: list[dict[str, Any]]) -> list[VllmWorkerResult]:
        vllm = importlib.import_module("vllm")
        sampling_params_type = cast(Any, vllm.SamplingParams)
        guided_decoding_type = cast(
            Any,
            importlib.import_module("vllm.sampling_params").GuidedDecodingParams,
        )

        prompts: list[dict[str, list[int]]] = []
        sampling_params = []
        for request in requests:
            prompt_ids = tuple(request["prompt_ids"])
            sampling = VllmSamplingConfig.model_validate(request["sampling"])
            prompts.append({"prompt_token_ids": list(prompt_ids)})
            sampling_params.append(
                sampling_params_type(
                    max_tokens=sampling.max_new_tokens,
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    top_k=sampling.top_k,
                    seed=sampling.seed,
                    logprobs=1,
                    guided_decoding=(
                        guided_decoding_type(regex=sampling.guided_regex)
                        if sampling.guided_regex is not None
                        else None
                    ),
                )
            )
        generate = cast(Any, self._llm.generate)
        outputs = generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=self._lora_request,
            use_tqdm=False,
        )
        if len(outputs) != len(requests):
            raise RuntimeError("vLLM returned another number of outputs")
        results = []
        for request, output in zip(requests, outputs, strict=True):
            requested_prompt = tuple(request["prompt_ids"])
            if output.prompt_token_ids is None:
                raise RuntimeError("vLLM omitted prompt token IDs")
            actual_prompt = tuple(output.prompt_token_ids)
            if actual_prompt != requested_prompt:
                raise RuntimeError("vLLM did not preserve input prompt token IDs")
            if len(output.outputs) != 1:
                raise RuntimeError("vLLM returned multiple completions for one request")
            completion = output.outputs[0]
            response_ids = tuple(completion.token_ids)
            if not response_ids:
                raise RuntimeError("vLLM returned an empty response")
            log_probs = _selected_log_probs(response_ids, completion.logprobs)
            results.append(
                VllmWorkerResult(
                    prompt_ids=actual_prompt,
                    response_ids=response_ids,
                    response_log_probs=log_probs,
                    stop_reason=_stop_reason(
                        completion.finish_reason,
                        response_ids,
                        self._eos_token_id,
                    ),
                    adapter_digest=self.adapter_digest,
                )
            )
        return results


class _Server:
    def __init__(self, runtime: _Runtime, socket_path: Path) -> None:
        self._runtime = runtime
        self._socket_path = socket_path
        self._shutdown = asyncio.Event()

    async def run(self) -> None:
        self._socket_path.unlink(missing_ok=True)
        start_unix_server = cast(Any, getattr(asyncio, _START_UNIX_SERVER_ATTRIBUTE))
        server = await start_unix_server(self._handle, path=str(self._socket_path))
        async with server:
            await self._shutdown.wait()
            server.close()
            await server.wait_closed()
        self._socket_path.unlink(missing_ok=True)

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            raw_request = await reader.readline()
            request = json.loads(raw_request)
            operation = request.get("operation")
            if operation == "health":
                response = {
                    "ok": True,
                    "adapter_digest": self._runtime.adapter_digest,
                    "runtime": self._runtime.runtime_metrics().model_dump(mode="json"),
                }
            elif operation == "generate_batch":
                raw_items = request.get("requests")
                if not isinstance(raw_items, list) or not raw_items:
                    raise ValueError("generate_batch requires a non-empty request list")
                results = self._runtime.generate_batch(raw_items)
                response = {
                    "ok": True,
                    "results": [result.model_dump(mode="json") for result in results],
                }
            elif operation == "shutdown":
                response = {"ok": True}
                self._shutdown.set()
            else:
                response = {"ok": False, "error_code": "unknown_operation"}
        except Exception:
            traceback.print_exc()
            response = {"ok": False, "error_code": "worker_failure"}
        writer.write(
            json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()


def _selected_log_probs(
    token_ids: tuple[int, ...],
    token_log_probs: list[dict[int, Any]] | None,
) -> tuple[float, ...]:
    if token_log_probs is None or len(token_log_probs) != len(token_ids):
        raise RuntimeError("vLLM did not return aligned per-token log-probabilities")
    selected: list[float] = []
    for token_id, candidates in zip(token_ids, token_log_probs, strict=True):
        candidate = candidates.get(token_id)
        if candidate is None:
            raise RuntimeError("selected token is absent from vLLM log-probabilities")
        value = candidate.logprob if hasattr(candidate, "logprob") else candidate
        selected.append(float(value))
    return tuple(selected)


def _stop_reason(
    finish_reason: str | None,
    response_ids: tuple[int, ...],
    eos_token_id: int | None,
) -> Literal["eos", "length", "stop", "cancelled"]:
    if finish_reason == "length":
        return "length"
    if finish_reason == "abort":
        return "cancelled"
    if eos_token_id is not None and response_ids[-1] == eos_token_id:
        return "eos"
    return "stop"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    spec = VllmWorkerSpec.model_validate_json(args.spec.read_text(encoding="utf-8"))
    if spec.config.engine != "v0" or spec.config.attention_backend != "xformers":
        raise RuntimeError("worker requires the pinned vLLM V0 + XFormers stack")
    asyncio.run(_Server(_Runtime(spec), args.socket).run())


if __name__ == "__main__":
    main()
