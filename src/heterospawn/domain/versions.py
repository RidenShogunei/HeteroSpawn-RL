"""Policy weight, deployment revision, and role-binding contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from heterospawn.domain.ids import PolicyId

AgentRole = Literal["main", "sub"]


class WeightVersion(BaseModel):
    """Immutable identity of training weights and their checkpoint."""

    model_config = ConfigDict(frozen=True, strict=True)

    policy_id: PolicyId
    optimizer_step: int = Field(ge=0)
    checkpoint_digest: str = Field(min_length=1)


class RolloutRevision(BaseModel):
    """A fully synchronized rollout replica set serving one weight version."""

    model_config = ConfigDict(frozen=True, strict=True)

    policy_id: PolicyId
    weight_version: WeightVersion
    deployment_id: str = Field(min_length=1)
    replica_set_revision: int = Field(ge=0)

    @model_validator(mode="after")
    def policy_ids_must_match(self) -> RolloutRevision:
        if self.policy_id != self.weight_version.policy_id:
            raise ValueError("rollout and weight policy_id must match")
        return self


class RoleBinding(BaseModel):
    """Explicitly maps a logical role to a physical policy."""

    model_config = ConfigDict(frozen=True, strict=True)

    role: AgentRole
    policy_id: PolicyId
    trainable: bool
