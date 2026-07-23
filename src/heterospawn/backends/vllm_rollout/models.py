"""Backend-specific immutable contracts for standalone vLLM rollout workers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from pydantic import BaseModel, ConfigDict, Field, model_validator

from heterospawn.domain.ids import PolicyId
from heterospawn.domain.training import RolloutArtifact, canonical_digest
from heterospawn.domain.versions import WeightVersion
from heterospawn.errors import CheckpointIntegrityError


class VllmRolloutConfig(BaseModel):
    """Pinned process and model configuration for the Turing compatibility stack."""

    model_config = ConfigDict(frozen=True, strict=True, allow_inf_nan=False)

    python_executable: Path
    model_path: Path
    expected_model_weight_sha256: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    prompt_template_revision: str = Field(min_length=1)
    runtime_dir: Path
    dtype: Literal["float16"] = "float16"
    engine: Literal["v0"] = "v0"
    attention_backend: Literal["xformers"] = "xformers"
    enforce_eager: Literal[True] = True
    gpu_memory_utilization: float = Field(default=0.5, gt=0.0, lt=1.0)
    max_model_len: int = Field(default=512, ge=16)
    max_new_tokens: int = Field(default=32, ge=1)
    max_num_seqs: int = Field(default=4, ge=1)
    max_lora_rank: int = Field(default=8, ge=1)
    batch_wait_ms: int = Field(default=5, ge=0, le=100)
    startup_timeout_seconds: float = Field(default=120.0, gt=0)
    shutdown_timeout_seconds: float = Field(default=15.0, gt=0)


class VllmPolicyDeployment(BaseModel):
    """One physical rollout worker assignment."""

    model_config = ConfigDict(frozen=True, strict=True)

    policy_id: PolicyId
    cuda_device: str = Field(min_length=1)
    deployment_id: str = Field(min_length=1)


class VllmWorkerSpec(BaseModel):
    """Complete immutable identity used to launch one worker process."""

    model_config = ConfigDict(frozen=True, strict=True)

    config: VllmRolloutConfig
    deployment: VllmPolicyDeployment
    artifact: RolloutArtifact

    @model_validator(mode="after")
    def policies_must_match(self) -> VllmWorkerSpec:
        if self.deployment.policy_id != self.artifact.policy_id:
            raise ValueError("deployment and rollout artifact policies must match")
        return self


class VllmSamplingConfig(BaseModel):
    """Normalized subset of sampling controls accepted by the rollout contract."""

    model_config = ConfigDict(frozen=True, strict=True, allow_inf_nan=False)

    max_new_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    top_k: int = Field(ge=-1)
    seed: int | None = None
    guided_regex: str | None = Field(default=None, min_length=1)


class VllmWorkerResult(BaseModel):
    """Exact values returned by the process that sampled the response."""

    model_config = ConfigDict(frozen=True, strict=True, allow_inf_nan=False)

    prompt_ids: tuple[int, ...] = Field(min_length=1)
    response_ids: tuple[int, ...] = Field(min_length=1)
    response_log_probs: tuple[float, ...] = Field(min_length=1)
    stop_reason: Literal["eos", "length", "stop", "cancelled"]
    adapter_digest: str = Field(min_length=1)

    @model_validator(mode="after")
    def tokens_must_align(self) -> VllmWorkerResult:
        if len(self.response_ids) != len(self.response_log_probs):
            raise ValueError("worker response IDs and log-probabilities must align")
        return self


class VllmWorkerRuntime(BaseModel):
    """Credential-safe runtime and GPU measurements reported by one worker."""

    model_config = ConfigDict(frozen=True, strict=True)

    gpu_name: str = Field(min_length=1)
    total_memory_bytes: int = Field(gt=0)
    device_used_bytes_at_query: int = Field(ge=0)
    peak_allocated_bytes: int = Field(ge=0)
    peak_reserved_bytes: int = Field(ge=0)
    versions: tuple[tuple[str, str], ...] = ()


class VllmWorker(Protocol):
    @property
    def adapter_digest(self) -> str: ...

    async def generate(
        self,
        prompt_ids: tuple[int, ...],
        sampling: VllmSamplingConfig,
    ) -> VllmWorkerResult: ...

    async def runtime_metrics(self) -> VllmWorkerRuntime: ...

    async def close(self) -> None: ...


class VllmWorkerFactory(Protocol):
    async def create(self, spec: VllmWorkerSpec) -> VllmWorker: ...


def rollout_artifact_path(artifact: RolloutArtifact) -> Path:
    """Resolve and verify a local PEFT LoRA rollout artifact."""

    if artifact.format_revision != "peft-lora-v1":
        raise CheckpointIntegrityError("vLLM requires peft-lora-v1 rollout artifacts")
    parsed = urlparse(artifact.uri)
    if parsed.scheme != "file":
        raise CheckpointIntegrityError("vLLM rollout artifact URI must use file scheme")
    path = Path(url2pathname(unquote(parsed.path))).resolve()
    if not path.is_dir():
        raise CheckpointIntegrityError("vLLM rollout artifact directory is missing")
    digest = rollout_artifact_digest(path, artifact.format_revision)
    if digest != artifact.artifact_digest:
        raise CheckpointIntegrityError("vLLM rollout artifact digest mismatch")
    return path


def rollout_artifact_digest(path: Path, format_revision: str = "peft-lora-v1") -> str:
    required = ("adapter_config.json", "adapter_model.safetensors")
    file_digests: dict[str, str] = {}
    for name in required:
        target = path / name
        if not target.is_file():
            raise CheckpointIntegrityError(f"vLLM rollout artifact is missing {name}")
        file_digests[name] = _file_sha256(target)
    return canonical_digest(
        {
            "format_revision": format_revision,
            "file_digests": file_digests,
        }
    )


def validate_artifact_version(
    artifact: RolloutArtifact,
    policy_id: PolicyId,
    version: WeightVersion,
) -> None:
    if artifact.policy_id != policy_id or artifact.weight_version != version:
        raise CheckpointIntegrityError("rollout artifact does not match requested policy version")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
