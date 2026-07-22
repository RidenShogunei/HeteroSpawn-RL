from __future__ import annotations

import hashlib

import pytest

from heterospawn.benchmarks.xbench import BenchmarkTask
from heterospawn.domain.ids import EpisodeId, PolicyId, TaskId
from heterospawn.policies.base import (
    EvaluationGenerationRequest,
    EvaluationGenerationResult,
    ExternalModelRevision,
    PolicyCapabilities,
    TokenUsage,
)
from heterospawn.search.base import SearchRequest
from heterospawn.search.mock import MockSearchService


class ScenarioPolicy:
    def __init__(self, subtask_count: int, *, invalid_first: bool = False) -> None:
        self._policy_id = PolicyId("shared-policy")
        self._revision = ExternalModelRevision(
            provider="scenario",
            model="v1",
            api_base="memory://scenario",
        )
        self._subtask_count = subtask_count
        self._invalid_first = invalid_first
        self.main_calls = 0

    @property
    def policy_id(self) -> PolicyId:
        return self._policy_id

    @property
    def revision(self) -> ExternalModelRevision:
        return self._revision

    async def generate(self, request: EvaluationGenerationRequest) -> EvaluationGenerationResult:
        if request.agent_role == "sub":
            content = f"evidence:{request.agent_instance_id}"
        else:
            self.main_calls += 1
            if self._invalid_first and self.main_calls == 1:
                content = "not json"
            elif self._subtask_count == 0 or self.main_calls > (2 if self._invalid_first else 1):
                content = '{"kind":"answer","answer":"final"}'
            else:
                subtasks = ",".join(f'"query-{index}"' for index in range(self._subtask_count))
                content = f'{{"kind":"spawn","subtasks":[{subtasks}]}}'
        digest = hashlib.sha256(content.encode()).hexdigest()
        return EvaluationGenerationResult(
            request_id=request.request_id,
            policy_id=self.policy_id,
            revision=self.revision,
            provider_request_id=f"scenario:{request.request_id}",
            content=content,
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            raw_response_digest=digest,
            capabilities=PolicyCapabilities(
                trainable=False,
                returns_token_ids=False,
                returns_old_log_probs=False,
            ),
        )


class PartiallyFailingSearch(MockSearchService):
    async def search(self, request: SearchRequest):  # type: ignore[no-untyped-def]
        if request.query == "query-1":
            raise RuntimeError("synthetic failure")
        return await super().search(request)


@pytest.mark.asyncio
@pytest.mark.parametrize("subtask_count", [0, 1, 4])
async def test_dynamic_sub_counts_have_stable_event_order(subtask_count: int) -> None:
    from heterospawn.orchestration.api_episode import ApiEpisodeOrchestrator

    trace = await ApiEpisodeOrchestrator(
        ScenarioPolicy(subtask_count),
        MockSearchService(),
        max_concurrency=2,
    ).run(
        BenchmarkTask(task_id=TaskId("task-1"), prompt="question"),
        EpisodeId(f"episode-{subtask_count}"),
    )

    assert trace.spawn_count == subtask_count
    assert trace.answer == "final"
    assert trace.trainable is False
    assert [event.event_index for event in trace.events] == list(range(len(trace.events)))
    assert [result.agent_instance_id for result in trace.sub_results] == [
        f"sub-{index}" for index in range(subtask_count)
    ]


@pytest.mark.asyncio
async def test_invalid_main_output_is_retained_before_repair() -> None:
    from heterospawn.orchestration.api_episode import ApiEpisodeOrchestrator

    trace = await ApiEpisodeOrchestrator(
        ScenarioPolicy(0, invalid_first=True),
        MockSearchService(),
    ).run(
        BenchmarkTask(task_id=TaskId("task-1"), prompt="question"),
        EpisodeId("episode-repair"),
    )

    assert [attempt.valid for attempt in trace.main_attempts] == [False, True]
    assert trace.main_attempts[0].content == "not json"
    assert [event.status for event in trace.events] == ["invalid", "valid"]
    assert trace.events[1].causal_event_indices == (0,)


@pytest.mark.asyncio
async def test_spawn_above_episode_limit_is_retained_and_repaired() -> None:
    from heterospawn.orchestration.api_episode import ApiEpisodeOrchestrator

    trace = await ApiEpisodeOrchestrator(
        ScenarioPolicy(5),
        MockSearchService(),
        max_spawn_per_episode=4,
    ).run(
        BenchmarkTask(task_id=TaskId("task-1"), prompt="question"),
        EpisodeId("episode-over-limit"),
    )

    assert trace.spawn_count == 0
    assert [attempt.valid for attempt in trace.main_attempts] == [False, True]
    assert [event.status for event in trace.events] == ["invalid", "valid"]


@pytest.mark.asyncio
async def test_one_sub_failure_does_not_cancel_siblings() -> None:
    from heterospawn.orchestration.api_episode import ApiEpisodeOrchestrator

    trace = await ApiEpisodeOrchestrator(
        ScenarioPolicy(4),
        PartiallyFailingSearch(),
        max_concurrency=2,
    ).run(
        BenchmarkTask(task_id=TaskId("task-1"), prompt="question"),
        EpisodeId("episode-partial-failure"),
    )

    assert [result.status for result in trace.sub_results] == [
        "success",
        "failed",
        "success",
        "success",
    ]
    assert trace.answer == "final"
