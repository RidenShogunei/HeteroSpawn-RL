from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import pytest

from heterospawn.domain.ids import (
    AgentInstanceId,
    EpisodeId,
    PolicyId,
    RolloutId,
    StepId,
    TaskId,
)
from heterospawn.domain.training import (
    CheckpointRef,
    PolicyTrainingBatch,
    TrainingPhase,
    TrajectoryStep,
    UpdateResult,
)
from heterospawn.domain.versions import RoleBinding, RolloutRevision, WeightVersion
from heterospawn.errors import PhaseTransactionError
from heterospawn.training import (
    AlternatingCoordinator,
    MockTrainingBackend,
    PolicyRegistry,
    TrainingBatchBuilder,
)
from heterospawn.training.transactions import (
    FilePhaseTransactionStore,
    PhaseCommitManifest,
    PhaseRecoveryManifest,
    PhaseTransactionContext,
    PhaseTransactionEvidence,
    PhaseTransactionManager,
)

CrashPoint = Literal["before_pending", "after_pending", "before_commit", "after_commit"]


def _batch(
    phase: TrainingPhase,
    policy_id: PolicyId,
    revision: RolloutRevision,
) -> PolicyTrainingBatch:
    step = TrajectoryStep(
        task_id=TaskId("task-1"),
        episode_id=EpisodeId(f"episode-{phase}"),
        rollout_id=RolloutId(f"rollout-{phase}"),
        step_id=StepId(f"step-{phase}"),
        event_index=0,
        agent_role="sub" if phase == "sub_update" else "main",
        agent_instance_id=AgentInstanceId(f"agent-{phase}"),
        policy_id=policy_id,
        rollout_revision=revision,
        prompt_ids=(1, 2),
        response_ids=(3, 4),
        response_log_probs=(-0.2, -0.3),
        tokenizer_revision="tok@1",
        prompt_template_revision="template@1",
        stop_reason="length",
    )
    return TrainingBatchBuilder().build(
        batch_id=f"cycle-1:{phase}",
        phase=phase,
        target_policy_id=policy_id,
        expected_base_version=revision.weight_version,
        steps=(step,),
        episode_advantages={step.episode_id: 1.0},
    )


def _registry(backend: MockTrainingBackend, policy_id: PolicyId) -> PolicyRegistry:
    return PolicyRegistry(
        (RoleBinding(role="main", policy_id=policy_id, trainable=True),),
        ((policy_id, backend.rollout_revision(policy_id)),),
    )


def _manager(
    tmp_path: Path,
    backend: MockTrainingBackend,
    registry: PolicyRegistry,
) -> tuple[FilePhaseTransactionStore, PhaseTransactionManager]:
    store = FilePhaseTransactionStore(tmp_path / "transactions")
    manager = PhaseTransactionManager(
        store=store,
        backend=backend,
        registry=registry,
        context=PhaseTransactionContext(
            experiment_id="experiment-1",
            config_digest="config-digest",
            rng_state="base64-rng-state",
            sampler_state="base64-sampler-state",
            dataset_revision="dataset@1",
            environment_snapshot="environment@1",
            reward_revision="reward@1",
        ),
        evidence_provider=lambda phase: PhaseTransactionEvidence(
            cycle_id="cycle-1",
            task_ids=(TaskId("task-1"),),
            episode_ids=(EpisodeId(f"episode-{phase}"),),
            rollout_ids=(RolloutId(f"rollout-{phase}"),),
        ),
    )
    return store, manager


class _CrashHook:
    def __init__(self, manager: PhaseTransactionManager, crash_point: CrashPoint) -> None:
        self._manager = manager
        self._crash_point = crash_point

    async def prepare(
        self,
        phase: TrainingPhase,
        target: PolicyId,
        batch: PolicyTrainingBatch,
        snapshot: tuple[tuple[PolicyId, RolloutRevision], ...],
    ) -> None:
        await self._manager.prepare(phase, target, batch, snapshot)

    async def record_update(self, phase: TrainingPhase, update: UpdateResult) -> None:
        if self._crash_point == "before_pending":
            raise RuntimeError("injected crash before pending update")
        await self._manager.record_update(phase, update)
        if self._crash_point == "after_pending":
            raise RuntimeError("injected crash after pending update")

    async def commit(
        self,
        phase: TrainingPhase,
        update: UpdateResult | None,
        rollout_revision: RolloutRevision,
    ) -> None:
        if self._crash_point == "before_commit":
            raise RuntimeError("injected crash before manifest")
        await self._manager.commit(phase, update, rollout_revision)
        if self._crash_point == "after_commit":
            raise RuntimeError("injected crash after manifest")


class _ReplacementDeploymentBackend:
    def __init__(self, delegate: MockTrainingBackend) -> None:
        self._delegate = delegate

    async def update_policy(
        self,
        policy_id: PolicyId,
        batch: PolicyTrainingBatch,
        expected_base_version: WeightVersion,
    ) -> UpdateResult:
        return await self._delegate.update_policy(policy_id, batch, expected_base_version)

    async def sync_rollout_weights(
        self,
        policy_id: PolicyId,
        trained_version: WeightVersion,
    ) -> RolloutRevision:
        revision = await self._delegate.sync_rollout_weights(policy_id, trained_version)
        return revision.model_copy(
            update={
                "deployment_id": f"{revision.deployment_id}:replacement",
                "replica_set_revision": 0,
            }
        )

    async def save_checkpoint(self, policy_id: PolicyId) -> CheckpointRef:
        return await self._delegate.save_checkpoint(policy_id)

    async def restore_checkpoint(self, checkpoint: CheckpointRef) -> WeightVersion:
        return await self._delegate.restore_checkpoint(checkpoint)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "crash_point",
    ["before_pending", "after_pending", "before_commit", "after_commit"],
)
async def test_phase_recovery_never_advances_public_state_or_optimizer_twice(
    tmp_path: Path,
    crash_point: CrashPoint,
) -> None:
    policy_id = PolicyId("main")
    backend = MockTrainingBackend((policy_id,))
    registry = _registry(backend, policy_id)
    store, manager = _manager(tmp_path, backend, registry)
    initial_revision = registry.revision(policy_id)

    async def rollout(
        phase: TrainingPhase,
        snapshot: tuple[tuple[PolicyId, RolloutRevision], ...],
    ) -> PolicyTrainingBatch:
        return _batch(phase, policy_id, dict(snapshot)[policy_id])

    with pytest.raises(RuntimeError, match="injected crash"):
        await AlternatingCoordinator(
            registry,
            backend,
            _CrashHook(manager, crash_point),
        ).run_cycle(rollout)

    transaction_id = "experiment-1:cycle-1:main_update"
    assert store.load_input(transaction_id) is not None
    assert registry.revision(policy_id) == initial_revision
    assert backend.weight_version(policy_id).optimizer_step == 1

    manifest = await manager.recover(transaction_id)

    assert isinstance(manifest, PhaseCommitManifest)
    assert manifest.phase_completed == "main_update"
    assert manifest.rollout_revision == registry.revision(policy_id)
    assert backend.weight_version(policy_id).optimizer_step == 1
    assert backend.rollout_revision(policy_id).weight_version.optimizer_step == 1
    assert await manager.recover(transaction_id) == manifest
    assert backend.weight_version(policy_id).optimizer_step == 1


@pytest.mark.asyncio
async def test_empty_sub_batch_commits_without_version_change(tmp_path: Path) -> None:
    main_id = PolicyId("main")
    sub_id = PolicyId("sub")
    backend = MockTrainingBackend((main_id, sub_id))
    registry = PolicyRegistry(
        (
            RoleBinding(role="main", policy_id=main_id, trainable=True),
            RoleBinding(role="sub", policy_id=sub_id, trainable=True),
        ),
        (
            (main_id, backend.rollout_revision(main_id)),
            (sub_id, backend.rollout_revision(sub_id)),
        ),
    )
    store, manager = _manager(tmp_path, backend, registry)

    async def rollout(
        phase: TrainingPhase,
        snapshot: tuple[tuple[PolicyId, RolloutRevision], ...],
    ) -> PolicyTrainingBatch:
        policy_id = main_id if phase == "main_update" else sub_id
        batch = _batch(phase, policy_id, dict(snapshot)[policy_id])
        if phase == "main_update":
            return batch
        return TrainingBatchBuilder().build(
            batch_id=batch.batch_id,
            phase=phase,
            target_policy_id=policy_id,
            expected_base_version=batch.expected_base_version,
            steps=(),
            episode_advantages={},
        )

    result = await AlternatingCoordinator(registry, backend, manager).run_cycle(rollout)

    assert result.empty_sub_batch
    manifest = store.load_commit("experiment-1:cycle-1:sub_update")
    assert isinstance(manifest, PhaseCommitManifest)
    assert manifest.empty_sub_batch
    assert manifest.committed_checkpoint is None
    assert backend.weight_version(sub_id).optimizer_step == 0
    assert backend.rollout_revision(sub_id).replica_set_revision == 0


@pytest.mark.asyncio
async def test_transaction_store_is_idempotent_and_rejects_conflicts(tmp_path: Path) -> None:
    policy_id = PolicyId("main")
    backend = MockTrainingBackend((policy_id,))
    registry = _registry(backend, policy_id)
    store, manager = _manager(tmp_path, backend, registry)
    revision = registry.revision(policy_id)
    batch = _batch("main_update", policy_id, revision)

    await manager.prepare("main_update", policy_id, batch, registry.snapshot())
    transaction_id = "experiment-1:cycle-1:main_update"
    transaction_input = store.load_input(transaction_id)
    assert transaction_input is not None
    await asyncio.gather(
        *(asyncio.to_thread(store.persist_input, transaction_input) for _ in range(4))
    )

    conflicting_manager = PhaseTransactionManager(
        store=store,
        backend=backend,
        registry=registry,
        context=transaction_input.context.model_copy(update={"config_digest": "different"}),
        evidence_provider=lambda phase: transaction_input.evidence,
    )
    with pytest.raises(PhaseTransactionError, match="conflicting"):
        await conflicting_manager.prepare(
            "main_update",
            policy_id,
            batch,
            registry.snapshot(),
        )


@pytest.mark.asyncio
async def test_committed_weight_can_recover_to_a_new_deployment_revision(
    tmp_path: Path,
) -> None:
    policy_id = PolicyId("main")
    backend = MockTrainingBackend((policy_id,))
    registry = _registry(backend, policy_id)
    store, manager = _manager(tmp_path, backend, registry)

    async def rollout(
        phase: TrainingPhase,
        snapshot: tuple[tuple[PolicyId, RolloutRevision], ...],
    ) -> PolicyTrainingBatch:
        return _batch(phase, policy_id, dict(snapshot)[policy_id])

    await AlternatingCoordinator(registry, backend, manager).run_cycle(rollout)
    committed = store.load_commit("experiment-1:cycle-1:main_update")
    transaction_input = store.load_input("experiment-1:cycle-1:main_update")
    assert isinstance(committed, PhaseCommitManifest)
    assert transaction_input is not None

    replacement_registry = PolicyRegistry(
        (RoleBinding(role="main", policy_id=policy_id, trainable=True),),
        ((policy_id, committed.rollout_revision),),
    )
    replacement = PhaseTransactionManager(
        store=store,
        backend=_ReplacementDeploymentBackend(backend),
        registry=replacement_registry,
        context=transaction_input.context,
        evidence_provider=lambda phase: PhaseTransactionEvidence(
            cycle_id="unused",
            task_ids=(TaskId("unused"),),
            episode_ids=(EpisodeId("unused"),),
            rollout_ids=(RolloutId("unused"),),
        ),
    )

    recovered = await replacement.recover(committed.transaction_id)

    assert recovered == committed
    assert replacement_registry.revision(policy_id).weight_version == (
        committed.rollout_revision.weight_version
    )
    assert replacement_registry.revision(policy_id) != committed.rollout_revision
    assert len(replacement.recoveries) == 1
    assert isinstance(replacement.recoveries[0], PhaseRecoveryManifest)
