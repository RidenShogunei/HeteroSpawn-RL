from __future__ import annotations

import pytest

from heterospawn.domain.ids import AgentInstanceId, EpisodeId, PolicyId, RolloutId, StepId, TaskId
from heterospawn.domain.training import PolicyTrainingBatch, TrainingPhase, TrajectoryStep
from heterospawn.domain.versions import RoleBinding, RolloutRevision
from heterospawn.training import (
    AlternatingCoordinator,
    MockTrainingBackend,
    PolicyRegistry,
    TrainingBatchBuilder,
)


def _batch(
    *,
    phase: TrainingPhase,
    policy_id: PolicyId,
    revision: RolloutRevision,
    empty: bool = False,
) -> PolicyTrainingBatch:
    step = TrajectoryStep(
        task_id=TaskId("task"),
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
        batch_id=f"batch-{phase}",
        phase=phase,
        target_policy_id=policy_id,
        expected_base_version=revision.weight_version,
        steps=() if empty else (step,),
        episode_advantages={step.episode_id: 1.0},
    )


@pytest.mark.asyncio
async def test_coordinator_enforces_fresh_main_then_sub_rollout() -> None:
    main = PolicyId("main")
    sub = PolicyId("sub")
    backend = MockTrainingBackend((main, sub))
    registry = PolicyRegistry(
        (
            RoleBinding(role="main", policy_id=main, trainable=True),
            RoleBinding(role="sub", policy_id=sub, trainable=True),
        ),
        (
            (main, backend.rollout_revision(main)),
            (sub, backend.rollout_revision(sub)),
        ),
    )
    observed: list[tuple[TrainingPhase, dict[PolicyId, RolloutRevision]]] = []

    async def rollout(
        phase: TrainingPhase,
        revisions: tuple[tuple[PolicyId, RolloutRevision], ...],
    ) -> PolicyTrainingBatch:
        snapshot = dict(revisions)
        observed.append((phase, snapshot))
        policy_id = main if phase == "main_update" else sub
        return _batch(phase=phase, policy_id=policy_id, revision=snapshot[policy_id])

    result = await AlternatingCoordinator(registry, backend).run_cycle(rollout)

    assert result.main_update is not None
    assert result.sub_update is not None
    assert [phase for phase, _ in observed] == ["main_update", "sub_update"]
    assert observed[0][1][main].replica_set_revision == 0
    assert observed[1][1][main].replica_set_revision == 1
    assert observed[1][1][sub].replica_set_revision == 0
    assert registry.role_revision("main").replica_set_revision == 1
    assert registry.role_revision("sub").replica_set_revision == 1


@pytest.mark.asyncio
async def test_coordinator_skips_empty_sub_without_version_change() -> None:
    main = PolicyId("main")
    sub = PolicyId("sub")
    backend = MockTrainingBackend((main, sub))
    registry = PolicyRegistry(
        (
            RoleBinding(role="main", policy_id=main, trainable=True),
            RoleBinding(role="sub", policy_id=sub, trainable=True),
        ),
        (
            (main, backend.rollout_revision(main)),
            (sub, backend.rollout_revision(sub)),
        ),
    )

    async def rollout(
        phase: TrainingPhase,
        revisions: tuple[tuple[PolicyId, RolloutRevision], ...],
    ) -> PolicyTrainingBatch:
        policy_id = main if phase == "main_update" else sub
        return _batch(
            phase=phase,
            policy_id=policy_id,
            revision=dict(revisions)[policy_id],
            empty=phase == "sub_update",
        )

    result = await AlternatingCoordinator(registry, backend).run_cycle(rollout)

    assert result.empty_sub_batch
    assert result.sub_update is None
    assert backend.weight_version(sub).optimizer_step == 0
    assert backend.rollout_revision(sub).replica_set_revision == 0


@pytest.mark.asyncio
async def test_shared_policy_executes_one_joint_update() -> None:
    shared = PolicyId("shared")
    backend = MockTrainingBackend((shared,))
    registry = PolicyRegistry(
        (
            RoleBinding(role="main", policy_id=shared, trainable=True),
            RoleBinding(role="sub", policy_id=shared, trainable=True),
        ),
        ((shared, backend.rollout_revision(shared)),),
    )
    phases: list[TrainingPhase] = []

    async def rollout(
        phase: TrainingPhase,
        revisions: tuple[tuple[PolicyId, RolloutRevision], ...],
    ) -> PolicyTrainingBatch:
        phases.append(phase)
        return _batch(phase=phase, policy_id=shared, revision=dict(revisions)[shared])

    result = await AlternatingCoordinator(registry, backend).run_cycle(rollout)

    assert phases == ["joint_update"]
    assert result.joint_update is not None
    assert backend.weight_version(shared).optimizer_step == 1
    assert backend.rollout_revision(shared).replica_set_revision == 1
