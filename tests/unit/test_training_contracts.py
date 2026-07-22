from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from heterospawn.domain.ids import AgentInstanceId, EpisodeId, PolicyId, RolloutId, StepId, TaskId
from heterospawn.domain.training import GenerationRequest, GenerationResult, TrajectoryStep
from heterospawn.domain.versions import RoleBinding
from heterospawn.errors import RolloutRevisionMismatch, TrainingBatchError, WeightVersionMismatch
from heterospawn.training import (
    MockTrainingBackend,
    PolicyRegistry,
    TrainingBatchBuilder,
    normalize_outcome_advantages,
)


def _request(*, role: str = "main", request_id: str = "request-1") -> GenerationRequest:
    return GenerationRequest(
        task_id=TaskId("task-1"),
        episode_id=EpisodeId("episode-1"),
        rollout_id=RolloutId("rollout-1"),
        request_id=request_id,
        agent_role=role,  # type: ignore[arg-type]
        agent_instance_id=AgentInstanceId(f"{role}-0"),
        prompt_ids=(11, 12, 13),
        tokenizer_revision="tokenizer@1",
        prompt_template_revision="template@1",
        sampling_params=(("max_new_tokens", 2),),
    )


def _step(
    *,
    policy_id: PolicyId,
    backend: MockTrainingBackend,
    episode: str,
    agent: str,
    step: str,
    response: tuple[int, ...] = (21, 22),
) -> TrajectoryStep:
    revision = backend.rollout_revision(policy_id)
    return TrajectoryStep(
        task_id=TaskId("task-1"),
        episode_id=EpisodeId(episode),
        rollout_id=RolloutId(f"rollout-{episode}"),
        step_id=StepId(step),
        event_index=int(step.rsplit("-", 1)[-1]),
        agent_role="main" if str(policy_id) == "main" else "sub",
        agent_instance_id=AgentInstanceId(agent),
        policy_id=policy_id,
        rollout_revision=revision,
        prompt_ids=(1, 2, 3),
        response_ids=response,
        response_log_probs=tuple(-0.1 * (index + 1) for index in range(len(response))),
        tokenizer_revision="tokenizer@1",
        prompt_template_revision="template@1",
        sampling_params=(("temperature", 0.0),),
        stop_reason="eos",
    )


@pytest.mark.asyncio
async def test_generation_requires_expected_rollout_revision_and_exact_alignment() -> None:
    policy_id = PolicyId("main")
    backend = MockTrainingBackend((policy_id,))
    endpoint = backend.endpoint(policy_id)
    revision = backend.rollout_revision(policy_id)

    result = await endpoint.generate(_request(), revision)

    assert len(result.response_ids) == len(result.response_log_probs)
    assert result.rollout_revision == revision
    stale = revision.model_copy(update={"replica_set_revision": 9})
    with pytest.raises(RolloutRevisionMismatch):
        await endpoint.generate(_request(), stale)

    with pytest.raises(ValidationError, match="must align"):
        GenerationResult(
            request_id="bad",
            policy_id=policy_id,
            rollout_revision=revision,
            response_ids=(1, 2),
            response_log_probs=(-0.1,),
            stop_reason="length",
        )


def test_batch_builder_preserves_tokens_and_derives_episode_balancing() -> None:
    policy_id = PolicyId("main")
    backend = MockTrainingBackend((policy_id,))
    steps = (
        _step(policy_id=policy_id, backend=backend, episode="ep-1", agent="a", step="s-0"),
        _step(policy_id=policy_id, backend=backend, episode="ep-1", agent="a", step="s-1"),
        _step(policy_id=policy_id, backend=backend, episode="ep-1", agent="b", step="s-2"),
        _step(
            policy_id=policy_id,
            backend=backend,
            episode="ep-2",
            agent="c",
            step="s-3",
            response=(31,),
        ),
    )
    batch = TrainingBatchBuilder().build(
        batch_id="batch-1",
        phase="main_update",
        target_policy_id=policy_id,
        expected_base_version=backend.weight_version(policy_id),
        steps=steps,
        episode_advantages={EpisodeId("ep-1"): 1.5, EpisodeId("ep-2"): -0.5},
        loss_masks={StepId("s-1"): (0, 1)},
    )

    assert tuple(sample.prompt_ids for sample in batch.samples) == tuple(
        step.prompt_ids for step in steps
    )
    assert tuple(sample.response_ids for sample in batch.samples) == tuple(
        step.response_ids for step in steps
    )
    assert tuple(sample.old_log_probs for sample in batch.samples) == tuple(
        step.response_log_probs for step in steps
    )
    assert [sample.aggregation_weight for sample in batch.samples] == [0.25, 0.25, 0.5, 1.0]
    assert batch.samples[1].loss_mask == (0, 1)
    assert (
        sum(
            sample.aggregation_weight
            for sample in batch.samples
            if sample.episode_id == EpisodeId("ep-1")
        )
        == 1.0
    )


def test_zero_one_four_sub_group_uses_zero_spawn_in_baseline_only() -> None:
    sub = PolicyId("sub")
    backend = MockTrainingBackend((sub,))
    steps = tuple(
        [
            _step(
                policy_id=sub,
                backend=backend,
                episode="ep-1",
                agent="sub-1-0",
                step="s-0",
            )
        ]
        + [
            _step(
                policy_id=sub,
                backend=backend,
                episode="ep-4",
                agent=f"sub-4-{index}",
                step=f"s-{index + 1}",
            )
            for index in range(4)
        ]
    )
    rewards = {
        EpisodeId("ep-0"): 0.0,
        EpisodeId("ep-1"): 1.0,
        EpisodeId("ep-4"): 2.0,
    }

    batch, group = TrainingBatchBuilder().build_from_rewards(
        batch_id="sub-batch",
        phase="sub_update",
        target_policy_id=sub,
        expected_base_version=backend.weight_version(sub),
        steps=steps,
        episode_rewards=rewards,
    )

    advantages = group.as_mapping()
    assert EpisodeId("ep-0") in advantages
    assert all(sample.episode_id != EpisodeId("ep-0") for sample in batch.samples)
    assert len(batch.samples) == 5
    assert [sample.aggregation_weight for sample in batch.samples] == [1.0] + [0.25] * 4
    assert batch.samples[0].advantage == advantages[EpisodeId("ep-1")]
    assert batch.samples[-1].advantage == advantages[EpisodeId("ep-4")]


def test_zero_reward_variance_yields_zero_advantage_and_degenerate_metric() -> None:
    group = normalize_outcome_advantages({EpisodeId("ep-1"): 0.5, EpisodeId("ep-2"): 0.5})

    assert group.degenerate
    assert group.reward_std == 0.0
    assert set(group.as_mapping().values()) == {0.0}


@pytest.mark.asyncio
async def test_mock_update_sync_is_idempotent_and_isolates_partner() -> None:
    main = PolicyId("main")
    sub = PolicyId("sub")
    backend = MockTrainingBackend((main, sub))
    base = backend.weight_version(main)
    old_rollout = backend.rollout_revision(main)
    sub_hash = backend.parameter_hash(sub)
    step = _step(policy_id=main, backend=backend, episode="ep-1", agent="main-0", step="s-0")
    batch = TrainingBatchBuilder().build(
        batch_id="main-batch",
        phase="main_update",
        target_policy_id=main,
        expected_base_version=base,
        steps=(step,),
        episode_advantages={EpisodeId("ep-1"): 1.0},
    )

    update = await backend.update_policy(main, batch, base)
    replay = await backend.update_policy(main, batch, base)

    assert replay == update
    assert backend.rollout_revision(main) == old_rollout
    assert backend.parameter_hash(sub) == sub_hash
    synced = await backend.sync_rollout_weights(main, update.trained_version)
    assert synced.weight_version == update.trained_version
    assert synced.replica_set_revision == old_rollout.replica_set_revision + 1
    assert await backend.sync_rollout_weights(main, update.trained_version) == synced

    with pytest.raises(WeightVersionMismatch):
        await backend.update_policy(main, batch.model_copy(update={"batch_id": "new"}), base)


@pytest.mark.asyncio
async def test_batch_id_conflict_and_checkpoint_restore_are_strict() -> None:
    main = PolicyId("main")
    backend = MockTrainingBackend((main,))
    base = backend.weight_version(main)
    builder = TrainingBatchBuilder()
    first = builder.build(
        batch_id="same-id",
        phase="main_update",
        target_policy_id=main,
        expected_base_version=base,
        steps=(_step(policy_id=main, backend=backend, episode="ep-1", agent="a", step="s-0"),),
        episode_advantages={EpisodeId("ep-1"): 1.0},
    )
    second = builder.build(
        batch_id="same-id",
        phase="main_update",
        target_policy_id=main,
        expected_base_version=base,
        steps=(_step(policy_id=main, backend=backend, episode="ep-2", agent="a", step="s-1"),),
        episode_advantages={EpisodeId("ep-2"): -1.0},
    )
    update = await backend.update_policy(main, first, base)

    with pytest.raises(TrainingBatchError, match="another digest"):
        await backend.update_policy(main, second, base)

    restored = await backend.restore_checkpoint(update.checkpoint)
    assert restored == update.trained_version


def test_registry_represents_single_shared_heterogeneous_and_frozen() -> None:
    main = PolicyId("main")
    sub = PolicyId("sub")
    backend = MockTrainingBackend((main, sub))
    revisions = (
        (main, backend.rollout_revision(main)),
        (sub, backend.rollout_revision(sub)),
    )
    single = PolicyRegistry(
        (RoleBinding(role="main", policy_id=main, trainable=True),),
        ((main, backend.rollout_revision(main)),),
    )
    heterogeneous = PolicyRegistry(
        (
            RoleBinding(role="main", policy_id=main, trainable=True),
            RoleBinding(role="sub", policy_id=sub, trainable=True),
        ),
        revisions,
    )
    frozen = PolicyRegistry(
        (
            RoleBinding(role="main", policy_id=main, trainable=True),
            RoleBinding(role="sub", policy_id=sub, trainable=False),
        ),
        revisions,
    )
    shared = PolicyRegistry(
        (
            RoleBinding(role="main", policy_id=main, trainable=True),
            RoleBinding(role="sub", policy_id=main, trainable=True),
        ),
        ((main, backend.rollout_revision(main)),),
    )

    assert single.target_for_phase("sub_update") is None
    assert heterogeneous.target_for_phase("sub_update") == sub
    assert frozen.target_for_phase("sub_update") is None
    assert shared.is_shared_trainable
    assert shared.target_for_phase("joint_update") == main


def revisions_by_policy(
    revisions: tuple[tuple[PolicyId, object], ...],
) -> Mapping[PolicyId, object]:
    return dict(revisions)
