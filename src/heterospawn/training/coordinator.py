"""Fresh-rollout alternating coordinator over backend-neutral contracts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import NamedTuple, Protocol

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


class PhaseTransactionHook(Protocol):
    async def prepare(
        self,
        phase: TrainingPhase,
        target: PolicyId,
        batch: PolicyTrainingBatch,
        snapshot: tuple[tuple[PolicyId, RolloutRevision], ...],
    ) -> None: ...

    async def record_update(
        self,
        phase: TrainingPhase,
        update: UpdateResult,
    ) -> None: ...

    async def commit(
        self,
        phase: TrainingPhase,
        update: UpdateResult | None,
        rollout_revision: RolloutRevision,
    ) -> None: ...


class AlternatingCoordinator:
    def __init__(
        self,
        registry: PolicyRegistry,
        backend: TrainingBackend,
        transaction_hook: PhaseTransactionHook | None = None,
    ) -> None:
        self._registry = registry
        self._backend = backend
        self._transaction_hook = transaction_hook

    async def run_cycle(self, rollout_batch: RolloutBatchFactory) -> CycleResult:
        if self._registry.is_shared_trainable:
            joint = await self._run_phase("joint_update", rollout_batch)
            return CycleResult(None, None, joint, False)

        main = await self._run_phase("main_update", rollout_batch)
        sub_target = self._registry.target_for_phase("sub_update")
        if sub_target is None:
            return CycleResult(main, None, None, False)
        snapshot = self._snapshot()
        sub_batch = await rollout_batch("sub_update", snapshot)
        if not sub_batch.samples:
            await self._prepare("sub_update", sub_target, sub_batch, snapshot)
            revision = dict(snapshot)[sub_target]
            await self._commit("sub_update", None, revision)
            return CycleResult(main, None, None, True)
        sub = await self._apply_batch("sub_update", sub_target, sub_batch, snapshot)
        return CycleResult(main, sub, None, False)

    async def _run_phase(
        self,
        phase: TrainingPhase,
        rollout_batch: RolloutBatchFactory,
    ) -> UpdateResult | None:
        target = self._registry.target_for_phase(phase)
        if target is None:
            return None
        snapshot = self._snapshot()
        batch = await rollout_batch(phase, snapshot)
        if not batch.samples:
            raise TrainingBatchError(f"{phase} cannot be empty")
        return await self._apply_batch(phase, target, batch, snapshot)

    async def _apply_batch(
        self,
        phase: TrainingPhase,
        target: PolicyId,
        batch: PolicyTrainingBatch,
        snapshot: tuple[tuple[PolicyId, RolloutRevision], ...],
    ) -> UpdateResult:
        policy_id = batch.target_policy_id
        if policy_id != target:
            raise TrainingBatchError("rollout factory returned a batch for another policy")
        await self._prepare(phase, target, batch, snapshot)
        update = await self._backend.update_policy(
            policy_id,
            batch,
            batch.expected_base_version,
        )
        if self._transaction_hook is not None:
            await self._transaction_hook.record_update(phase, update)
        revision = await self._backend.sync_rollout_weights(
            policy_id,
            update.trained_version,
        )
        await self._commit(phase, update, revision)
        self._registry.replace_revision(revision)
        return update

    async def _prepare(
        self,
        phase: TrainingPhase,
        target: PolicyId,
        batch: PolicyTrainingBatch,
        snapshot: tuple[tuple[PolicyId, RolloutRevision], ...],
    ) -> None:
        if self._transaction_hook is not None:
            await self._transaction_hook.prepare(phase, target, batch, snapshot)

    async def _commit(
        self,
        phase: TrainingPhase,
        update: UpdateResult | None,
        revision: RolloutRevision,
    ) -> None:
        if self._transaction_hook is not None:
            await self._transaction_hook.commit(phase, update, revision)

    def _snapshot(self) -> tuple[tuple[PolicyId, RolloutRevision], ...]:
        return self._registry.snapshot()
