from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from heterospawn.benchmarks.xbench import load_xbench
from heterospawn.domain.ids import PolicyId, TaskId
from heterospawn.evaluation.api_pilot import ApiPilotConfig, ApiPilotRunner
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
    def __init__(self, *, fail_task_id: TaskId | None = None) -> None:
        self._policy_id = PolicyId("pilot-policy")
        self._revision = ExternalModelRevision(
            provider="test",
            model="pilot-v1",
            api_base="memory://pilot",
        )
        self._fail_task_id = fail_task_id
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
        answer = (
            "sensitive-correct-answer-x9" if request.task_id == TaskId("synthetic-1") else "wrong"
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
    clock_values = iter(index / 10 for index in range(8))
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
    assert report.zero_spawn_episodes == 4
    assert report.total_tokens == 12
    assert report.score.exact_correct_episodes == 2
    assert report.score.average_exact_accuracy == 0.5
    assert report.score.best_of_n_exact_accuracy == 0.5
    serialized = report.model_dump_json()
    assert "private prompt" not in serialized
    assert "最终答案" not in serialized
    assert "sensitive-correct-answer-x9" not in serialized
    assert "alpha" not in serialized


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
    assert report.episodes[1].error_code == "RuntimeError"
    assert report.score.total_episodes == 2
    assert "private provider body" not in report.model_dump_json()
