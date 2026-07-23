from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest

from heterospawn.benchmarks.xbench import load_xbench
from heterospawn.domain.ids import EpisodeId, TaskId
from heterospawn.errors import BenchmarkDataError
from heterospawn.evaluation.judges import (
    JudgeRequest,
    JudgeResult,
    JudgeRevision,
)
from heterospawn.orchestration.trainable_models import TrainableEpisodeTrace
from heterospawn.policies.base import TokenUsage
from heterospawn.rewards.xbench import XBenchOutcomeReward

FixtureFactory = Callable[[tuple[tuple[str, str, str], ...]], Path]


class _Judge:
    def __init__(self, *, correct: bool = True, official: bool = False) -> None:
        self.requests: list[JudgeRequest] = []
        self._correct = correct
        self._revision = JudgeRevision(
            mode="gemini-official" if official else "minimax-development",
            provider="fixture",
            model="fixture-judge",
            prompt_revision="fixture-prompt-v1",
            sampling_params=(("temperature", 0.0),),
            max_format_repair_attempts=0,
            format_repair_revision=None,
            comparable_to_official=official,
        )

    @property
    def revision(self) -> JudgeRevision:
        return self._revision

    async def judge(self, request: JudgeRequest) -> JudgeResult:
        self.requests.append(request)
        return JudgeResult(
            request_id=request.request_id,
            correct=self._correct,
            extracted_answer_digest="extracted-digest",
            reason_digest="reason-digest",
            provider_response_digest="response-digest",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            latency_ms=7,
        )


def _trace(task_id: TaskId, episode_id: str, answer: str) -> TrainableEpisodeTrace:
    return TrainableEpisodeTrace.model_construct(
        task_id=task_id,
        episode_id=EpisodeId(episode_id),
        status="success",
        answer=answer,
    )


@pytest.mark.asyncio
async def test_exact_training_reward_keeps_answer_inside_evaluator(
    xbench_fixture_factory: FixtureFactory,
) -> None:
    dataset = load_xbench(
        xbench_fixture_factory((("task-1", "private question", "alpha"),)),
        verify_official_digest=False,
    )
    task = dataset.tasks[0]
    reward = XBenchOutcomeReward(
        dataset,
        task_ids=(task.task_id,),
        mode="development-exact-only",
    )

    correct = await reward.score(
        task,
        _trace(task.task_id, "episode-correct", "alpha"),
    )
    incorrect = await reward.score(
        task,
        _trace(task.task_id, "episode-wrong", "beta"),
    )

    assert correct == 1.0
    assert incorrect == 0.0
    assert (await reward.audit(EpisodeId("episode-correct"))).outcome.resolution == "direct_exact"
    safe_json = (await reward.audit(EpisodeId("episode-wrong"))).model_dump_json()
    assert "private question" not in safe_json
    assert "alpha" not in safe_json


@pytest.mark.asyncio
async def test_development_judge_scores_exact_miss_and_caches_safe_audit(
    xbench_fixture_factory: FixtureFactory,
) -> None:
    dataset = load_xbench(
        xbench_fixture_factory((("task-1", "private question", "alpha"),)),
        verify_official_digest=False,
    )
    task = dataset.tasks[0]
    judge = _Judge(correct=True)
    reward = XBenchOutcomeReward(
        dataset,
        task_ids=(task.task_id,),
        mode="development-judge",
        judge=judge,
    )
    trace = _trace(task.task_id, "episode-judge", "no exact marker")

    assert await reward.score(task, trace) == 1.0
    assert await reward.score(task, trace) == 1.0
    assert len(judge.requests) == 1
    assert judge.requests[0].request_id == "episode-judge:xbench-training-reward"
    assert judge.requests[0].correct_answer == "alpha"

    audit = await reward.audit(EpisodeId("episode-judge"))
    assert audit.outcome.resolution == "development_judge"
    assert audit.outcome.judge_total_tokens == 12
    assert audit.outcome.judge_response_digest == "response-digest"
    assert "alpha" not in audit.model_dump_json()
    assert "private question" not in audit.model_dump_json()


@pytest.mark.asyncio
async def test_concurrent_reward_retry_is_single_flight_and_identity_checked(
    xbench_fixture_factory: FixtureFactory,
) -> None:
    dataset = load_xbench(
        xbench_fixture_factory((("task-1", "private question", "alpha"),)),
        verify_official_digest=False,
    )
    task = dataset.tasks[0]
    judge = _Judge(correct=True)
    reward = XBenchOutcomeReward(
        dataset,
        task_ids=(task.task_id,),
        mode="development-judge",
        judge=judge,
    )
    trace = _trace(task.task_id, "episode-retry", "first response")

    assert await asyncio.gather(reward.score(task, trace), reward.score(task, trace)) == [
        1.0,
        1.0,
    ]
    assert len(judge.requests) == 1
    with pytest.raises(BenchmarkDataError, match="another response"):
        await reward.score(
            task,
            _trace(task.task_id, "episode-retry", "different response"),
        )


@pytest.mark.asyncio
async def test_exact_hit_bypasses_judge_and_official_judge_is_rejected(
    xbench_fixture_factory: FixtureFactory,
) -> None:
    dataset = load_xbench(
        xbench_fixture_factory((("task-1", "private question", "alpha"),)),
        verify_official_digest=False,
    )
    task = dataset.tasks[0]
    judge = _Judge()
    reward = XBenchOutcomeReward(
        dataset,
        task_ids=(task.task_id,),
        mode="development-judge",
        judge=judge,
    )

    assert (
        await reward.score(
            task,
            _trace(task.task_id, "episode-exact", "alpha"),
        )
        == 1.0
    )
    assert judge.requests == []

    with pytest.raises(ValueError, match="must agree"):
        XBenchOutcomeReward(
            dataset,
            task_ids=(task.task_id,),
            mode="development-exact-only",
            judge=judge,
        )
    with pytest.raises(BenchmarkDataError, match="official-comparable"):
        XBenchOutcomeReward(
            dataset,
            task_ids=(task.task_id,),
            mode="development-judge",
            judge=_Judge(official=True),
        )


def test_reward_revision_changes_with_mode_and_selected_tasks(
    xbench_fixture_factory: FixtureFactory,
) -> None:
    dataset = load_xbench(
        xbench_fixture_factory(
            (
                ("task-1", "first private question", "alpha"),
                ("task-2", "second private question", "beta"),
            )
        ),
        verify_official_digest=False,
    )
    exact_one = XBenchOutcomeReward(
        dataset,
        task_ids=(TaskId("task-1"),),
        mode="development-exact-only",
    )
    exact_two = XBenchOutcomeReward(
        dataset,
        task_ids=(TaskId("task-2"),),
        mode="development-exact-only",
    )
    judged = XBenchOutcomeReward(
        dataset,
        task_ids=(TaskId("task-1"),),
        mode="development-judge",
        judge=_Judge(),
    )

    assert len({exact_one.revision, exact_two.revision, judged.revision}) == 3
