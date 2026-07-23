"""Revision-safe rollout service with restart-based LoRA synchronization."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from heterospawn.backends.vllm_rollout.models import (
    VllmPolicyDeployment,
    VllmRolloutConfig,
    VllmSamplingConfig,
    VllmWorker,
    VllmWorkerFactory,
    VllmWorkerResult,
    VllmWorkerRuntime,
    VllmWorkerSpec,
    rollout_artifact_path,
    validate_artifact_version,
)
from heterospawn.domain.ids import PolicyId
from heterospawn.domain.training import GenerationRequest, GenerationResult, RolloutArtifact
from heterospawn.domain.versions import RolloutRevision, WeightVersion
from heterospawn.errors import (
    CheckpointIntegrityError,
    RolloutRevisionMismatch,
    RolloutServiceError,
    TrainingBatchError,
    WeightVersionMismatch,
)


@dataclass
class _PolicyState:
    deployment: VllmPolicyDeployment
    artifact: RolloutArtifact
    revision: RolloutRevision
    worker: VllmWorker | None
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    sync_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_generations: int = 0
    transitioning: bool = False


class VllmPolicyEndpoint:
    def __init__(self, service: VllmRolloutService, policy_id: PolicyId) -> None:
        self._service = service
        self._policy_id = policy_id

    @property
    def policy_id(self) -> PolicyId:
        return self._policy_id

    async def current_rollout_revision(self) -> RolloutRevision:
        return self._service.rollout_revision(self._policy_id)

    async def generate(
        self,
        request: GenerationRequest,
        expected_revision: RolloutRevision,
    ) -> GenerationResult:
        return await self._service.generate(self._policy_id, request, expected_revision)


class VllmRolloutService:
    """Owns one independently restartable rollout worker per physical policy."""

    def __init__(
        self,
        *,
        config: VllmRolloutConfig,
        worker_factory: VllmWorkerFactory,
        states: dict[PolicyId, _PolicyState],
    ) -> None:
        self.config = config
        self._worker_factory = worker_factory
        self._states = states
        self._closed = False
        self._close_lock = asyncio.Lock()

    @classmethod
    async def start(
        cls,
        *,
        config: VllmRolloutConfig,
        deployments: tuple[tuple[VllmPolicyDeployment, RolloutArtifact], ...],
        worker_factory: VllmWorkerFactory,
    ) -> VllmRolloutService:
        if not deployments:
            raise TrainingBatchError("vLLM rollout requires at least one policy deployment")
        policy_ids = tuple(deployment.policy_id for deployment, _ in deployments)
        if len(set(policy_ids)) != len(policy_ids):
            raise TrainingBatchError("vLLM policy deployments must be unique")
        _validate_runtime_config(config)

        states: dict[PolicyId, _PolicyState] = {}
        created: list[VllmWorker] = []
        try:
            for deployment, artifact in deployments:
                validate_artifact_version(
                    artifact,
                    deployment.policy_id,
                    artifact.weight_version,
                )
                rollout_artifact_path(artifact)
                spec = VllmWorkerSpec(
                    config=config,
                    deployment=deployment,
                    artifact=artifact,
                )
                worker = await worker_factory.create(spec)
                if worker.adapter_digest != artifact.artifact_digest:
                    await worker.close()
                    raise CheckpointIntegrityError(
                        "vLLM worker loaded an unexpected adapter digest"
                    )
                created.append(worker)
                revision = RolloutRevision(
                    policy_id=deployment.policy_id,
                    weight_version=artifact.weight_version,
                    deployment_id=deployment.deployment_id,
                    replica_set_revision=0,
                )
                states[deployment.policy_id] = _PolicyState(
                    deployment=deployment,
                    artifact=artifact,
                    revision=revision,
                    worker=worker,
                )
        except Exception:
            await asyncio.gather(*(worker.close() for worker in created), return_exceptions=True)
            raise
        return cls(config=config, worker_factory=worker_factory, states=states)

    def endpoint(self, policy_id: PolicyId) -> VllmPolicyEndpoint:
        self._state(policy_id)
        return VllmPolicyEndpoint(self, policy_id)

    def rollout_revision(self, policy_id: PolicyId) -> RolloutRevision:
        return self._state(policy_id).revision

    async def runtime_metrics(self, policy_id: PolicyId) -> VllmWorkerRuntime:
        self._ensure_open()
        state = self._state(policy_id)
        async with state.condition:
            if state.transitioning or state.worker is None:
                raise RolloutServiceError("vLLM rollout worker is transitioning or unavailable")
            worker = state.worker
            state.active_generations += 1
        try:
            return await worker.runtime_metrics()
        finally:
            await self._release_generation(state)

    async def generate(
        self,
        policy_id: PolicyId,
        request: GenerationRequest,
        expected_revision: RolloutRevision,
    ) -> GenerationResult:
        self._ensure_open()
        state = self._state(policy_id)
        worker = await self._acquire_generation(state, request, expected_revision)
        try:
            sampling = _sampling_config(request, self.config)
            worker_result = await worker.generate(request.prompt_ids, sampling)
            _validate_worker_result(
                worker_result,
                request=request,
                artifact=state.artifact,
            )
            if state.revision != expected_revision:
                raise RolloutRevisionMismatch("vLLM rollout revision changed during generation")
            return GenerationResult(
                request_id=request.request_id,
                policy_id=policy_id,
                rollout_revision=expected_revision,
                response_ids=worker_result.response_ids,
                response_log_probs=worker_result.response_log_probs,
                stop_reason=worker_result.stop_reason,
                usage=(
                    ("prompt_tokens", len(request.prompt_ids)),
                    ("completion_tokens", len(worker_result.response_ids)),
                ),
            )
        finally:
            await self._release_generation(state)

    async def sync_rollout_weights(
        self,
        policy_id: PolicyId,
        trained_version: WeightVersion,
        artifact: RolloutArtifact,
    ) -> RolloutRevision:
        self._ensure_open()
        validate_artifact_version(artifact, policy_id, trained_version)
        rollout_artifact_path(artifact)
        state = self._state(policy_id)
        async with state.sync_lock:
            if state.revision.weight_version == trained_version:
                if state.artifact.artifact_digest != artifact.artifact_digest:
                    raise CheckpointIntegrityError(
                        "same weight version was supplied with another artifact digest"
                    )
                return state.revision
            if trained_version.optimizer_step <= state.revision.weight_version.optimizer_step:
                raise WeightVersionMismatch("cannot deploy stale vLLM rollout weights")
            old_worker, old_artifact = await self._begin_transition(state)
            try:
                await old_worker.close()
                candidate = await self._worker_factory.create(
                    VllmWorkerSpec(
                        config=self.config,
                        deployment=state.deployment,
                        artifact=artifact,
                    )
                )
                if candidate.adapter_digest != artifact.artifact_digest:
                    await candidate.close()
                    raise CheckpointIntegrityError(
                        "replacement vLLM worker loaded an unexpected adapter digest"
                    )
            except Exception as replacement_error:
                await self._rollback_transition(
                    state,
                    old_artifact,
                    replacement_error,
                )
                raise RolloutServiceError(
                    "vLLM replacement failed; prior rollout revision was restored"
                ) from replacement_error

            revision = RolloutRevision(
                policy_id=policy_id,
                weight_version=trained_version,
                deployment_id=state.revision.deployment_id,
                replica_set_revision=state.revision.replica_set_revision + 1,
            )
            async with state.condition:
                state.worker = candidate
                state.artifact = artifact
                state.revision = revision
                state.transitioning = False
                state.condition.notify_all()
            return revision

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            errors: list[Exception] = []
            for state in self._states.values():
                try:
                    async with state.sync_lock:
                        worker, _ = await self._begin_transition(state)
                        try:
                            await worker.close()
                        finally:
                            async with state.condition:
                                state.worker = None
                                state.transitioning = False
                                state.condition.notify_all()
                except Exception as error:
                    errors.append(error)
            if errors:
                raise RolloutServiceError(
                    "one or more vLLM workers failed to close"
                ) from ExceptionGroup(
                    "vLLM worker close failures",
                    errors,
                )

    async def _acquire_generation(
        self,
        state: _PolicyState,
        request: GenerationRequest,
        expected_revision: RolloutRevision,
    ) -> VllmWorker:
        async with state.condition:
            if state.revision != expected_revision:
                raise RolloutRevisionMismatch("vLLM rollout revision mismatch")
            if request.tokenizer_revision != self.config.tokenizer_revision:
                raise TrainingBatchError("vLLM tokenizer revision mismatch")
            if request.prompt_template_revision != self.config.prompt_template_revision:
                raise TrainingBatchError("vLLM prompt-template revision mismatch")
            if len(request.prompt_ids) >= self.config.max_model_len:
                raise TrainingBatchError("prompt exceeds vLLM max_model_len")
            if state.transitioning or state.worker is None:
                raise RolloutServiceError("vLLM rollout worker is transitioning or unavailable")
            state.active_generations += 1
            return state.worker

    @staticmethod
    async def _release_generation(state: _PolicyState) -> None:
        async with state.condition:
            state.active_generations -= 1
            if state.active_generations < 0:
                raise RuntimeError("vLLM active generation count became negative")
            state.condition.notify_all()

    @staticmethod
    async def _begin_transition(
        state: _PolicyState,
    ) -> tuple[VllmWorker, RolloutArtifact]:
        async with state.condition:
            state.transitioning = True
            while state.active_generations:
                await state.condition.wait()
            if state.worker is None:
                state.transitioning = False
                state.condition.notify_all()
                raise RolloutServiceError("vLLM rollout worker is unavailable")
            worker = state.worker
            state.worker = None
            return worker, state.artifact

    async def _rollback_transition(
        self,
        state: _PolicyState,
        old_artifact: RolloutArtifact,
        replacement_error: Exception,
    ) -> None:
        try:
            restored = await self._worker_factory.create(
                VllmWorkerSpec(
                    config=self.config,
                    deployment=state.deployment,
                    artifact=old_artifact,
                )
            )
            if restored.adapter_digest != old_artifact.artifact_digest:
                await restored.close()
                raise CheckpointIntegrityError(
                    "rollback vLLM worker loaded an unexpected adapter digest"
                )
        except Exception as rollback_error:
            async with state.condition:
                state.worker = None
                state.transitioning = False
                state.condition.notify_all()
            raise RolloutServiceError(
                "vLLM replacement and rollback both failed"
            ) from ExceptionGroup(
                "vLLM rollout transition failures",
                [replacement_error, rollback_error],
            )
        async with state.condition:
            state.worker = restored
            state.artifact = old_artifact
            state.transitioning = False
            state.condition.notify_all()

    def _state(self, policy_id: PolicyId) -> _PolicyState:
        try:
            return self._states[policy_id]
        except KeyError:
            raise TrainingBatchError(f"unknown vLLM policy: {policy_id}") from None

    def _ensure_open(self) -> None:
        if self._closed:
            raise RolloutServiceError("vLLM rollout service is closed")


def _sampling_config(
    request: GenerationRequest,
    config: VllmRolloutConfig,
) -> VllmSamplingConfig:
    params = dict(request.sampling_params)
    allowed = {
        "max_new_tokens",
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "guided_regex",
    }
    unknown = set(params) - allowed
    if unknown:
        raise TrainingBatchError(f"unsupported vLLM sampling parameters: {sorted(unknown)}")
    do_sample = _as_bool(params.get("do_sample", False), name="do_sample")
    max_new_tokens = _as_int(
        params.get("max_new_tokens", config.max_new_tokens),
        name="max_new_tokens",
    )
    max_new_tokens = min(max_new_tokens, config.max_model_len - len(request.prompt_ids))
    if max_new_tokens < 1:
        raise TrainingBatchError("max_new_tokens must be positive")
    temperature = (
        _as_float(params.get("temperature", 1.0), name="temperature") if do_sample else 0.0
    )
    top_p = _as_float(params.get("top_p", 1.0), name="top_p")
    top_k = _as_int(params.get("top_k", -1), name="top_k")
    seed_value = params.get("seed")
    seed = None if seed_value is None else _as_int(seed_value, name="seed")
    guided_regex_value = params.get("guided_regex")
    guided_regex = (
        None if guided_regex_value is None else _as_str(guided_regex_value, name="guided_regex")
    )
    try:
        return VllmSamplingConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            guided_regex=guided_regex,
        )
    except ValueError as error:
        raise TrainingBatchError("invalid vLLM sampling parameters") from error


def _validate_worker_result(
    result: VllmWorkerResult,
    *,
    request: GenerationRequest,
    artifact: RolloutArtifact,
) -> None:
    if result.prompt_ids != request.prompt_ids:
        raise RolloutServiceError("vLLM worker did not round-trip prompt token IDs")
    if result.adapter_digest != artifact.artifact_digest:
        raise RolloutServiceError("vLLM worker response used another adapter digest")


def _validate_runtime_config(config: VllmRolloutConfig) -> None:
    if not config.python_executable.is_file():
        raise TrainingBatchError("vLLM python_executable is missing")
    if not config.model_path.is_dir():
        raise TrainingBatchError("vLLM model_path is missing")
    model_weight = config.model_path / "model.safetensors"
    if not model_weight.is_file():
        raise TrainingBatchError("vLLM model_path must contain model.safetensors")
    if _file_sha256(model_weight) != config.expected_model_weight_sha256:
        raise CheckpointIntegrityError("vLLM base-model weight digest mismatch")
    config.runtime_dir.mkdir(parents=True, exist_ok=True)


def _file_sha256(path: Any) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingBatchError(f"{name} must be an integer")
    return value


def _as_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingBatchError(f"{name} must be numeric")
    return float(value)


def _as_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TrainingBatchError(f"{name} must be a boolean")
    return value


def _as_str(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrainingBatchError(f"{name} must be a non-empty string")
    return value
