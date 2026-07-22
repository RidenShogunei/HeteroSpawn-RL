"""Immutable exact-token rollout and policy-training contracts."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from heterospawn.domain.ids import (
    AgentInstanceId,
    CheckpointId,
    EpisodeId,
    PolicyId,
    RolloutId,
    StepId,
    TaskId,
)
from heterospawn.domain.versions import AgentRole, RolloutRevision, WeightVersion

JsonScalar: TypeAlias = None | bool | int | float | str
TrainingPhase: TypeAlias = Literal["main_update", "sub_update", "joint_update"]


def canonical_digest(value: object) -> str:
    """Return a stable SHA-256 digest for JSON-compatible domain data."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class GenerationRequest(BaseModel):
    """Token-level request for an auditable trainable rollout."""

    model_config = ConfigDict(frozen=True, strict=True)

    task_id: TaskId
    episode_id: EpisodeId
    rollout_id: RolloutId
    request_id: str = Field(min_length=1)
    agent_role: AgentRole
    agent_instance_id: AgentInstanceId
    prompt_ids: tuple[int, ...] = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    prompt_template_revision: str = Field(min_length=1)
    sampling_params: tuple[tuple[str, JsonScalar], ...] = ()


class GenerationResult(BaseModel):
    """Exact output returned by the policy that performed sampling."""

    model_config = ConfigDict(frozen=True, strict=True)

    request_id: str = Field(min_length=1)
    policy_id: PolicyId
    rollout_revision: RolloutRevision
    response_ids: tuple[int, ...] = Field(min_length=1)
    response_log_probs: tuple[float, ...] = Field(min_length=1)
    stop_reason: Literal["eos", "length", "stop", "cancelled"]
    usage: tuple[tuple[str, int], ...] = ()

    @model_validator(mode="after")
    def token_log_probs_must_align(self) -> GenerationResult:
        if len(self.response_ids) != len(self.response_log_probs):
            raise ValueError("response_ids and response_log_probs must align")
        if self.policy_id != self.rollout_revision.policy_id:
            raise ValueError("result policy_id must match rollout revision")
        return self


class TrajectoryStep(BaseModel):
    """Immutable MODEL event preserving values returned by the rollout backend."""

    model_config = ConfigDict(frozen=True, strict=True)

    task_id: TaskId
    episode_id: EpisodeId
    rollout_id: RolloutId
    step_id: StepId
    event_index: int = Field(ge=0)
    causal_step_ids: tuple[StepId, ...] = ()
    agent_role: AgentRole
    agent_instance_id: AgentInstanceId
    policy_id: PolicyId
    rollout_revision: RolloutRevision
    partner_rollout_revisions: tuple[RolloutRevision, ...] = ()
    prompt_ids: tuple[int, ...] = Field(min_length=1)
    response_ids: tuple[int, ...] = Field(min_length=1)
    response_log_probs: tuple[float, ...] = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    prompt_template_revision: str = Field(min_length=1)
    sampling_params: tuple[tuple[str, JsonScalar], ...] = ()
    stop_reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def provenance_must_be_consistent(self) -> TrajectoryStep:
        if len(self.response_ids) != len(self.response_log_probs):
            raise ValueError("response_ids and response_log_probs must align")
        if self.policy_id != self.rollout_revision.policy_id:
            raise ValueError("step policy_id must match rollout revision")
        return self


class PolicyTrainingSample(BaseModel):
    """Phase-derived training record referencing one immutable MODEL step."""

    model_config = ConfigDict(frozen=True, strict=True)

    task_id: TaskId
    episode_id: EpisodeId
    rollout_id: RolloutId
    source_step_id: StepId
    agent_role: AgentRole
    agent_instance_id: AgentInstanceId
    policy_id: PolicyId
    rollout_revision: RolloutRevision
    prompt_ids: tuple[int, ...] = Field(min_length=1)
    response_ids: tuple[int, ...] = Field(min_length=1)
    old_log_probs: tuple[float, ...] = Field(min_length=1)
    loss_mask: tuple[int, ...] = Field(min_length=1)
    advantage: float
    aggregation_weight: float = Field(gt=0)

    @model_validator(mode="after")
    def token_fields_must_align(self) -> PolicyTrainingSample:
        length = len(self.response_ids)
        if len(self.old_log_probs) != length or len(self.loss_mask) != length:
            raise ValueError("response, log-prob, and loss-mask lengths must align")
        if any(item not in (0, 1) for item in self.loss_mask):
            raise ValueError("loss_mask values must be zero or one")
        if not any(self.loss_mask):
            raise ValueError("training sample must contain at least one active token")
        return self


def training_batch_digest_payload(
    *,
    batch_id: str,
    phase: TrainingPhase,
    target_policy_id: PolicyId,
    expected_base_version: WeightVersion,
    samples: tuple[PolicyTrainingSample, ...],
) -> dict[str, object]:
    return {
        "batch_id": batch_id,
        "phase": phase,
        "target_policy_id": target_policy_id,
        "expected_base_version": expected_base_version.model_dump(mode="json"),
        "samples": [sample.model_dump(mode="json") for sample in samples],
        "loss_aggregation": "episode_balanced",
    }


class PolicyTrainingBatch(BaseModel):
    """A digest-protected policy update transaction input."""

    model_config = ConfigDict(frozen=True, strict=True)

    batch_id: str = Field(min_length=1)
    phase: TrainingPhase
    target_policy_id: PolicyId
    expected_base_version: WeightVersion
    samples: tuple[PolicyTrainingSample, ...] = ()
    loss_aggregation: Literal["episode_balanced"] = "episode_balanced"
    batch_digest: str = Field(min_length=1)

    @model_validator(mode="after")
    def batch_must_match_digest_and_target(self) -> PolicyTrainingBatch:
        for sample in self.samples:
            if sample.policy_id != self.target_policy_id:
                raise ValueError("all samples must target the physical policy")
            if sample.rollout_revision.weight_version != self.expected_base_version:
                raise ValueError("sample rollout weights must match expected base version")
        expected = canonical_digest(
            training_batch_digest_payload(
                batch_id=self.batch_id,
                phase=self.phase,
                target_policy_id=self.target_policy_id,
                expected_base_version=self.expected_base_version,
                samples=self.samples,
            )
        )
        if self.batch_digest != expected:
            raise ValueError("batch_digest does not match batch contents")
        return self


class CheckpointRef(BaseModel):
    """Immutable checkpoint identity, including optimizer state."""

    model_config = ConfigDict(frozen=True, strict=True)

    checkpoint_id: CheckpointId
    policy_id: PolicyId
    weight_version: WeightVersion
    uri: str = Field(min_length=1)
    optimizer_state_digest: str = Field(min_length=1)


class UpdateResult(BaseModel):
    """Result of one logical optimizer transaction."""

    model_config = ConfigDict(frozen=True, strict=True)

    policy_id: PolicyId
    base_version: WeightVersion
    trained_version: WeightVersion
    checkpoint: CheckpointRef
    metrics: tuple[tuple[str, float], ...] = ()

    @model_validator(mode="after")
    def versions_must_match_policy(self) -> UpdateResult:
        if any(
            policy_id != self.policy_id
            for policy_id in (
                self.base_version.policy_id,
                self.trained_version.policy_id,
                self.checkpoint.policy_id,
            )
        ):
            raise ValueError("update result policies must match")
        if self.checkpoint.weight_version != self.trained_version:
            raise ValueError("checkpoint must identify trained version")
        return self
