from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from heterospawn.benchmarks.xbench import load_xbench
from heterospawn.domain.ids import PolicyId, TaskId
from heterospawn.evaluation.api_pilot import ApiPilotConfig, ApiPilotRunner
from heterospawn.evaluation.judges import JudgeRequest, JudgeResult, JudgeRevision
from heterospawn.policies.base import (
    EvaluationGenerationRequest,
    EvaluationGenerationResult,
    ExternalModelRevision,
    PolicyCapabilities,
    TokenUsage,
)
from heterospawn.search.mock import MOCK_SEARCH_REVISION, MockSearchService

FixtureFactory = Callable[[tuple[tuple[str, str, str], ...]], Path]


class PilotPolicy:
    def __init__(
        self,
        *,
        fail_task_id: TaskId | None = None,
        invalid_task_id: TaskId | None = None,
    ) -> None:
        self._policy_id = PolicyId("pilot-policy")
        self._revision = ExternalModelRevision(
            provider="test",
            model="pilot-v1",
            api_base="memory://pilot",
        )
        self._fail_task_id = fail_task_id
        self._invalid_task_id = invalid_task_id
        self.requests: list[EvaluationGenerationRequest] = []

    @property
    def policy_id(self) -> PolicyId:
        return self._policy_id

    @property
    def revision(self) -> ExternalModelRevision:
        return self._revision

    async def generate(self, request: EvaluationGenerationRequest) -> EvaluationGenerationResult:
        self.requests.append(request)
        if request.task_id == self._fail_task_id:
            raise RuntimeError("private provider body")
        if request.task_id == self._invalid_task_id:
            content = "sensitive invalid action"
        else:
            answer = (
                "sensitive-correct-answer-x9"
                if request.task_id == TaskId("synthetic-1")
                else "wrong"
            )
            content = f'{{"kind":"answer","answer":"最终答案:{answer}"}}'
        return EvaluationGenerationResult(
            request_id=request.request_id,
            policy_id=self.policy_id,
            revision=self.revision,
            provider_request_id=f"test:{request.request_id}",
            content=content,
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            raw_response_digest=hashlib.sha256(content.encode()).hexdigest(),
            capabilities=PolicyCapabilities(
                trainable=False,
                returns_token_ids=False,
                returns_old_log_probs=False,
            ),
        )


class DevelopmentJudge:
    def __init__(self) -> None:
        self._revision = JudgeRevision(
            mode="minimax-development",
            provider="test",
            model="judge-v1",
            prompt_revision="prompt-v1",
            sampling_params=(),
            max_format_repair_attempts=0,
            format_repair_revision=None,
            comparable_to_official=False,
        )
        self.requests: list[JudgeRequest] = []

    @property
    def revision(self) -> JudgeRevision:
        return self._revision

    async def judge(self, request: JudgeRequest) -> JudgeResult:
        self.requests.append(request)
        return JudgeResult(
            request_id=request.request_id,
            correct=True,
            extracted_answer_digest="a" * 64,
            reason_digest="b" * 64,
            provider_response_digest="c" * 64,
            usage=TokenUsage(prompt_tokens=4, completion_tokens=5, total_tokens=9),
            latency_ms=10,
        )


@pytest.mark.asyncio
async def test_pilot_runs_fresh_repeats_and_emits_only_safe_aggregates(
    xbench_fixture_factory: FixtureFactory,
) -> None:
    fixture = xbench_fixture_factory(
        (
            ("synthetic-1", "first private prompt", "sensitive-correct-answer-x9"),
            ("synthetic-2", "second private prompt", "alpha"),
        )
    )
    dataset = load_xbench(fixture, verify_official_digest=False)
    policy = PilotPolicy()
    clock_values = iter((0.0, 0.1, 1.0, 1.3, 2.0, 2.2, 3.0, 3.4))
    progress = []

    report = await ApiPilotRunner(
        dataset,
        policy,
        MockSearchService(),
        ApiPilotConfig(
            run_id="unit-pilot",
            task_ids=(TaskId("synthetic-1"), TaskId("synthetic-2")),
            repeats_per_task=2,
            search_backend="mock",
            search_revision=MOCK_SEARCH_REVISION,
        ),
        clock=lambda: next(clock_values),
        progress_callback=progress.append,
    ).run()

    assert len({request.episode_id for request in policy.requests}) == 4
    assert all(request.sampling_params for request in policy.requests)
    assert report.completed_episodes == 4
    assert progress == list(report.episodes)
    assert report.failed_episodes == 0
    assert report.failure_counts == ()
    assert report.zero_spawn_episodes == 4
    assert report.total_tokens == 12
    assert report.successful_subs == 0
    assert report.failed_subs == 0
    assert report.invalid_main_attempts == 0
    assert report.latency_p50_ms == 200
    assert report.latency_p95_ms == 400
    assert report.latency_max_ms == 400
    assert [summary.task_id for summary in report.task_summaries] == [
        TaskId("synthetic-1"),
        TaskId("synthetic-2"),
    ]
    assert all(summary.attempted_episodes == 2 for summary in report.task_summaries)
    assert all(summary.completed_episodes == 2 for summary in report.task_summaries)
    assert report.task_summaries[0].latency_p50_ms == 100
    assert report.task_summaries[0].latency_p95_ms == 300
    assert report.task_summaries[1].latency_p50_ms == 200
    assert report.task_summaries[1].latency_p95_ms == 400
    assert report.score.exact_correct_episodes == 2
    assert report.score.average_exact_accuracy == 0.5
    assert report.score.best_of_n_exact_accuracy == 0.5
    serialized = report.model_dump_json()
    assert "private prompt" not in serialized
    assert "最终答案" not in serialized
    assert "sensitive-correct-answer-x9" not in serialized
    assert "alpha" not in serialized


@pytest.mark.asyncio
async def test_pilot_retains_safe_usage_for_exhausted_invalid_actions(
    xbench_fixture_factory: FixtureFactory,
) -> None:
    fixture = xbench_fixture_factory(
        (("synthetic-1", "private prompt", "sensitive-correct-answer-x9"),)
    )
    report = await ApiPilotRunner(
        load_xbench(fixture, verify_official_digest=False),
        PilotPolicy(invalid_task_id=TaskId("synthetic-1")),
        MockSearchService(),
        ApiPilotConfig(
            run_id="invalid-action-pilot",
            task_ids=(TaskId("synthetic-1"),),
            search_backend="mock",
            search_revision=MOCK_SEARCH_REVISION,
        ),
    ).run()

    assert report.completed_episodes == 0
    assert report.failed_episodes == 1
    assert report.failure_counts == (("InvalidActionError", 1),)
    assert report.episodes[0].error_code == "InvalidActionError"
    assert report.episodes[0].main_attempts == 2
    assert report.episodes[0].invalid_main_attempts == 2
    assert report.episodes[0].event_count == 2
    assert report.episodes[0].total_tokens == 6
    assert report.total_tokens == 6
    assert report.invalid_main_attempts == 2
    assert report.task_summaries[0].failure_counts == (("InvalidActionError", 1),)
    assert report.task_summaries[0].total_tokens == 6
    assert "sensitive invalid action" not in report.model_dump_json()


@pytest.mark.asyncio
async def test_pilot_isolates_episode_failures_without_persisting_error_text(
    xbench_fixture_factory: FixtureFactory,
) -> None:
    fixture = xbench_fixture_factory(
        (
            ("synthetic-1", "first private prompt", "sensitive-correct-answer-x9"),
            ("synthetic-2", "second private prompt", "alpha"),
        )
    )
    report = await ApiPilotRunner(
        load_xbench(fixture, verify_official_digest=False),
        PilotPolicy(fail_task_id=TaskId("synthetic-2")),
        MockSearchService(),
        ApiPilotConfig(
            run_id="failure-pilot",
            task_ids=(TaskId("synthetic-1"), TaskId("synthetic-2")),
            search_backend="mock",
            search_revision=MOCK_SEARCH_REVISION,
        ),
    ).run()

    assert report.completed_episodes == 1
    assert report.failed_episodes == 1
    assert report.failure_counts == (("RuntimeError", 1),)
    assert report.episodes[1].error_code == "RuntimeError"
    assert report.task_summaries[0].completed_episodes == 1
    assert report.task_summaries[1].failed_episodes == 1
    assert report.score.total_episodes == 2
    assert "private provider body" not in report.model_dump_json()


@pytest.mark.asyncio
async def test_pilot_short_circuits_exact_matches_before_development_judge(
    xbench_fixture_factory: FixtureFactory,
) -> None:
    fixture = xbench_fixture_factory(
        (
            ("synthetic-1", "first private prompt", "sensitive-correct-answer-x9"),
            ("synthetic-2", "second private prompt", "alpha"),
        )
    )
    judge = DevelopmentJudge()
    report = await ApiPilotRunner(
        load_xbench(fixture, verify_official_digest=False),
        PilotPolicy(),
        MockSearchService(),
        ApiPilotConfig(
            run_id="judged-pilot",
            task_ids=(TaskId("synthetic-1"), TaskId("synthetic-2")),
            repeats_per_task=2,
            search_backend="mock",
            search_revision=MOCK_SEARCH_REVISION,
            judge_mode="minimax-development",
        ),
        judge=judge,
    ).run()

    assert len(judge.requests) == 2
    assert all(request.task_id == TaskId("synthetic-2") for request in judge.requests)
    assert report.score.mode == "development-minimax-judge"
    assert report.score.comparable_to_official is False
    assert report.score.direct_exact_matches == 2
    assert report.score.judge_calls == 2
    assert report.score.judge_failures == 0
    assert report.score.judge_latency_ms == 20
    assert report.score.average_accuracy == 1.0
    serialized = report.model_dump_json()
    assert "first private prompt" not in serialized
    assert "second private prompt" not in serialized
    assert "sensitive-correct-answer-x9" not in serialized
    assert "alpha" not in serialized
