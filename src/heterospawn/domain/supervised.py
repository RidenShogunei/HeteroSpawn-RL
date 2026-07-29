"""Exact-token contracts for optional supervised policy warm starts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from heterospawn.domain.ids import PolicyId, TaskId
from heterospawn.domain.training import canonical_digest
from heterospawn.domain.versions import AgentRole, WeightVersion

SupervisedBehavior = Literal[
    "main_spawn",
    "main_final",
    "sub_search",
    "sub_access",
    "sub_summary",
]


class SupervisedPromptEncoding(BaseModel):
    """One prompt/target encoding produced directly by the pinned chat template."""

    model_config = ConfigDict(frozen=True, strict=True)

    prompt_ids: tuple[int, ...] = Field(min_length=1)
    target_ids: tuple[int, ...] = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    prompt_template_revision: str = Field(min_length=1)


def supervised_example_digest_payload(
    *,
    example_id: str,
    task_id: TaskId,
    agent_role: AgentRole,
    behavior: SupervisedBehavior,
    dataset_revision: str,
    source_digest: str,
    constructor_revision: str,
    encoding: SupervisedPromptEncoding,
    loss_mask: tuple[int, ...],
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "task_id": task_id,
        "agent_role": agent_role,
        "behavior": behavior,
        "dataset_revision": dataset_revision,
        "source_digest": source_digest,
        "constructor_revision": constructor_revision,
        "encoding": encoding.model_dump(mode="json"),
        "loss_mask": loss_mask,
    }


class SupervisedTrainingExample(BaseModel):
    """Plaintext-free token record; decoded targets never enter this contract."""

    model_config = ConfigDict(frozen=True, strict=True)

    example_id: str = Field(min_length=1)
    task_id: TaskId
    agent_role: AgentRole
    behavior: SupervisedBehavior
    dataset_revision: str = Field(min_length=1)
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    constructor_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoding: SupervisedPromptEncoding
    loss_mask: tuple[int, ...] = Field(min_length=2)
    example_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def role_mask_and_digest_must_match(self) -> SupervisedTrainingExample:
        expected_role: AgentRole = "main" if self.behavior.startswith("main_") else "sub"
        if self.agent_role != expected_role:
            raise ValueError("supervised behavior does not match agent role")
        prompt_length = len(self.encoding.prompt_ids)
        target_length = len(self.encoding.target_ids)
        expected_mask = (0,) * prompt_length + (1,) * target_length
        if self.loss_mask != expected_mask:
            raise ValueError("supervised loss_mask must select only target tokens")
        expected_digest = canonical_digest(
            supervised_example_digest_payload(
                example_id=self.example_id,
                task_id=self.task_id,
                agent_role=self.agent_role,
                behavior=self.behavior,
                dataset_revision=self.dataset_revision,
                source_digest=self.source_digest,
                constructor_revision=self.constructor_revision,
                encoding=self.encoding,
                loss_mask=self.loss_mask,
            )
        )
        if self.example_digest != expected_digest:
            raise ValueError("example_digest does not match supervised example")
        return self


def supervised_batch_digest_payload(
    *,
    batch_id: str,
    target_policy_id: PolicyId,
    expected_base_version: WeightVersion,
    examples: tuple[SupervisedTrainingExample, ...],
) -> dict[str, object]:
    return {
        "batch_id": batch_id,
        "target_policy_id": target_policy_id,
        "expected_base_version": expected_base_version.model_dump(mode="json"),
        "examples": [example.model_dump(mode="json") for example in examples],
        "objective": "causal_sft",
    }


class SupervisedTrainingBatch(BaseModel):
    """Digest-protected SFT input, deliberately separate from rollout/RL batches."""

    model_config = ConfigDict(frozen=True, strict=True)

    batch_id: str = Field(min_length=1)
    target_policy_id: PolicyId
    expected_base_version: WeightVersion
    examples: tuple[SupervisedTrainingExample, ...] = Field(min_length=1)
    objective: Literal["causal_sft"] = "causal_sft"
    batch_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def batch_identity_must_match(self) -> SupervisedTrainingBatch:
        if self.target_policy_id != self.expected_base_version.policy_id:
            raise ValueError("supervised target policy must match expected base version")
        if len({example.example_id for example in self.examples}) != len(self.examples):
            raise ValueError("supervised batch contains duplicate example IDs")
        if len({example.dataset_revision for example in self.examples}) != 1:
            raise ValueError("supervised batch cannot mix dataset revisions")
        if len({example.constructor_revision for example in self.examples}) != 1:
            raise ValueError("supervised batch cannot mix constructor revisions")
        if len({example.encoding.tokenizer_revision for example in self.examples}) != 1:
            raise ValueError("supervised batch cannot mix tokenizer revisions")
        expected_digest = canonical_digest(
            supervised_batch_digest_payload(
                batch_id=self.batch_id,
                target_policy_id=self.target_policy_id,
                expected_base_version=self.expected_base_version,
                examples=self.examples,
            )
        )
        if self.batch_digest != expected_digest:
            raise ValueError("batch_digest does not match supervised batch")
        return self
