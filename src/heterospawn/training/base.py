"""Backend-neutral protocols for exact-token generation and policy updates."""

from __future__ import annotations

from typing import Protocol

from heterospawn.domain.ids import PolicyId
from heterospawn.domain.training import (
    CheckpointRef,
    GenerationRequest,
    GenerationResult,
    PolicyTrainingBatch,
    RolloutArtifact,
    UpdateResult,
)
from heterospawn.domain.versions import RolloutRevision, WeightVersion


class PolicyService(Protocol):
    @property
    def policy_id(self) -> PolicyId: ...

    async def generate(
        self,
        request: GenerationRequest,
        expected_revision: RolloutRevision,
    ) -> GenerationResult: ...

    async def current_rollout_revision(self) -> RolloutRevision: ...


class TrainingBackend(Protocol):
    async def update_policy(
        self,
        policy_id: PolicyId,
        batch: PolicyTrainingBatch,
        expected_base_version: WeightVersion,
    ) -> UpdateResult: ...

    async def sync_rollout_weights(
        self,
        policy_id: PolicyId,
        trained_version: WeightVersion,
    ) -> RolloutRevision: ...

    async def save_checkpoint(self, policy_id: PolicyId) -> CheckpointRef: ...

    async def restore_checkpoint(self, checkpoint: CheckpointRef) -> WeightVersion: ...


class RolloutArtifactProvider(Protocol):
    async def export_rollout_artifact(
        self,
        policy_id: PolicyId,
        trained_version: WeightVersion,
    ) -> RolloutArtifact: ...
