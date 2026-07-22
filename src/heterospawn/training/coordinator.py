"""Fresh-rollout alternating coordinator over backend-neutral contracts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import NamedTuple

from heterospawn.domain.ids import PolicyId
from heterospawn.domain.training import PolicyTrainingBatch, TrainingPhase, UpdateResult
from heterospawn.domain.versions import RolloutRevision
from heterospawn.errors import TrainingBatchError
from heterospawn.training.base import TrainingBackend
from heterospawn.training.registry import PolicyRegistry

RolloutBatchFactory = Callable[
    [TrainingPhase, tuple[tuple[PolicyId, RolloutRevision], ...]],
    Awaitable[PolicyTrainingBatch],
]


class CycleResult(NamedTuple):
    main_update: UpdateResult | None
    sub_update: UpdateResult | None
    joint_update: UpdateResult | None
    empty_sub_batch: bool


class AlternatingCoordinator:
    def __init__(self, registry: PolicyRegistry, backend: TrainingBackend) -> None:
        self._registry = registry
        self._backend = backend

    async def run_cycle(self, rollout_batch: RolloutBatchFactory) -> CycleResult:
        if self._registry.is_shared_trainable:
            joint = await self._run_phase("joint_update", rollout_batch)
            return CycleResult(None, None, joint, False)

        main = await self._run_phase("main_update", rollout_batch)
        sub_target = self._registry.target_for_phase("sub_update")
        if sub_target is None:
            return CycleResult(main, None, None, False)
        sub_batch = await rollout_batch("sub_update", self._snapshot())
        if not sub_batch.samples:
            return CycleResult(main, None, None, True)
        sub = await self._apply_batch(sub_target, sub_batch)
        return CycleResult(main, sub, None, False)

    async def _run_phase(
        self,
        phase: TrainingPhase,
        rollout_batch: RolloutBatchFactory,
    ) -> UpdateResult | None:
        target = self._registry.target_for_phase(phase)
        if target is None:
            return None
        batch = await rollout_batch(phase, self._snapshot())
        if not batch.samples:
            raise TrainingBatchError(f"{phase} cannot be empty")
        return await self._apply_batch(target, batch)

    async def _apply_batch(
        self,
        target: PolicyId,
        batch: PolicyTrainingBatch,
    ) -> UpdateResult:
        policy_id = batch.target_policy_id
        if policy_id != target:
            raise TrainingBatchError("rollout factory returned a batch for another policy")
        update = await self._backend.update_policy(
            policy_id,
            batch,
            batch.expected_base_version,
        )
        revision = await self._backend.sync_rollout_weights(
            policy_id,
            update.trained_version,
        )
        self._registry.replace_revision(revision)
        return update

    def _snapshot(self) -> tuple[tuple[PolicyId, RolloutRevision], ...]:
        return self._registry.snapshot()
