"""Repeatable, credential-safe API pilot for a fixed benchmark task set."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from heterospawn.benchmarks.xbench import (
    XBENCH_UPSTREAM_REVISION,
    JudgedRepeatScoreReport,
    RepeatExactScoreReport,
    XBenchDataset,
)
from heterospawn.domain.ids import EpisodeId, TaskId
from heterospawn.errors import EpisodeRunError
from heterospawn.evaluation.judges import JudgeRevision, JudgeService
from heterospawn.orchestration.api_episode import ApiEpisodeOrchestrator
from heterospawn.orchestration.models import EpisodeTrace
from heterospawn.policies.base import (
    EvaluationPolicyService,
    ExternalModelRevision,
    JsonScalar,
)
from heterospawn.search.base import SearchService

Clock = Callable[[], float]
ProgressCallback = Callable[["PilotEpisodeSummary"], None]


class ApiPilotConfig(BaseModel):
    """Inputs that define one reproducible pilot run."""

    model_config = ConfigDict(frozen=True, strict=True)

    run_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9._-]+$")
    task_ids: tuple[TaskId, ...] = Field(min_length=1)
    repeats_per_task: int = Field(default=1, ge=1, le=20)
    search_backend: str = Field(min_length=1)
    search_revision: str = Field(min_length=1)
    judge_mode: Literal["none", "minimax-development"] = "none"
    max_spawn_per_episode: int = Field(default=4, ge=1, le=32)
    max_concurrency: int = Field(default=4, ge=1, le=32)
    repair_attempts: int = Field(default=1, ge=0, le=4)
    sampling_params: tuple[tuple[str, JsonScalar], ...] = (
        ("temperature", 1.0),
        ("top_p", 0.95),
        ("max_completion_tokens", 4096),
        ("reasoning_split", True),
    )


class PilotManifest(BaseModel):
    """Revision-complete inputs; deliberately excludes timestamps and secrets."""

    model_config = ConfigDict(frozen=True, strict=True)

    run_id: str
    benchmark: Literal["xbench-DeepSearch-2510"] = "xbench-DeepSearch-2510"
    dataset_revision: str
    encrypted_source_sha256: str
    evaluator_revision: str
    evaluator_mode: Literal["development-repeat-exact-only", "development-minimax-judge"]
    official_repeats: int = 5
    task_ids: tuple[TaskId, ...]
    repeats_per_task: int
    execution_mode: Literal["sequential-fresh-episodes"] = "sequential-fresh-episodes"
    policy_revision: ExternalModelRevision
    search_backend: str
    search_revision: str
    judge_revision: JudgeRevision | None
    max_spawn_per_episode: int
    max_concurrency: int
    repair_attempts: int
    sampling_params: tuple[tuple[str, JsonScalar], ...]
    trainable: Literal[False] = False
    comparable_to_official: Literal[False] = False


class PilotEpisodeSummary(BaseModel):
    """Safe operational summary; no prompt, answer, evidence, or provider body."""

    model_config = ConfigDict(frozen=True, strict=True)

    task_id: TaskId
    repeat_index: int = Field(ge=0)
    episode_id: EpisodeId
    status: Literal["completed", "failed"]
    error_code: str | None = None
    spawn_count: int = Field(ge=0)
    successful_subs: int = Field(ge=0)
    failed_subs: int = Field(ge=0)
    main_attempts: int = Field(ge=0)
    invalid_main_attempts: int = Field(ge=0)
    event_count: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    trainable: Literal[False] = False


class ApiPilotReport(BaseModel):
    """One complete safe report for audit and comparison."""

    model_config = ConfigDict(frozen=True, strict=True)

    manifest: PilotManifest
    manifest_digest: str
    episodes: tuple[PilotEpisodeSummary, ...]
    score: RepeatExactScoreReport | JudgedRepeatScoreReport
    completed_episodes: int = Field(ge=0)
    failed_episodes: int = Field(ge=0)
    zero_spawn_episodes: int = Field(ge=0)
    total_spawn_count: int = Field(ge=0)
    total_prompt_tokens: int = Field(ge=0)
    total_completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    total_latency_ms: int = Field(ge=0)


class ApiPilotRunner:
    """Runs every configured repeat as a new episode and retains only safe summaries."""

    def __init__(
        self,
        dataset: XBenchDataset,
        policy: EvaluationPolicyService,
        search: SearchService,
        config: ApiPilotConfig,
        *,
        judge: JudgeService | None = None,
        clock: Clock = time.monotonic,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self._dataset = dataset
        self._policy = policy
        self._search = search
        self._config = config
        self._judge = judge
        self._clock = clock
        self._progress_callback = progress_callback
        if (config.judge_mode == "none") != (judge is None):
            raise ValueError("judge service and configured judge mode must agree")
        if judge is not None and judge.revision.mode != config.judge_mode:
            raise ValueError("judge service revision does not match configured judge mode")

    async def run(self) -> ApiPilotReport:
        tasks = self._dataset.select_tasks(self._config.task_ids)
        manifest = self._build_manifest()
        manifest_digest = _digest_model(manifest)
        summaries: list[PilotEpisodeSummary] = []
        predictions: dict[TaskId, list[str | None]] = {task.task_id: [] for task in tasks}

        for task in tasks:
            for repeat_index in range(self._config.repeats_per_task):
                episode_id = EpisodeId(
                    f"{self._config.run_id}:{task.task_id}:repeat-{repeat_index + 1}"
                )
                start = self._clock()
                try:
                    trace = await ApiEpisodeOrchestrator(
                        self._policy,
                        self._search,
                        max_concurrency=self._config.max_concurrency,
                        max_spawn_per_episode=self._config.max_spawn_per_episode,
                        repair_attempts=self._config.repair_attempts,
                        sampling_params=self._config.sampling_params,
                    ).run(task, episode_id)
                    latency_ms = _elapsed_ms(start, self._clock())
                    self._validate_search_revision(trace)
                    predictions[task.task_id].append(trace.answer)
                    summary = _completed_summary(task.task_id, repeat_index, trace, latency_ms)
                except EpisodeRunError as exc:
                    predictions[task.task_id].append(None)
                    summary = _failed_summary(
                        task.task_id,
                        repeat_index,
                        episode_id,
                        _elapsed_ms(start, self._clock()),
                        exc.error_code,
                        failure=exc,
                    )
                except Exception as exc:
                    predictions[task.task_id].append(None)
                    summary = _failed_summary(
                        task.task_id,
                        repeat_index,
                        episode_id,
                        _elapsed_ms(start, self._clock()),
                        type(exc).__name__,
                    )
                summaries.append(summary)
                if self._progress_callback is not None:
                    self._progress_callback(summary)

        frozen_predictions = {
            task_id: tuple(task_predictions) for task_id, task_predictions in predictions.items()
        }
        if self._judge is None:
            score: RepeatExactScoreReport | JudgedRepeatScoreReport = (
                self._dataset.evaluate_repeat_exact(
                    frozen_predictions,
                    task_ids=self._config.task_ids,
                    repeats_per_task=self._config.repeats_per_task,
                )
            )
        else:
            score = await self._dataset.evaluate_repeat_with_judge(
                frozen_predictions,
                task_ids=self._config.task_ids,
                repeats_per_task=self._config.repeats_per_task,
                judge=self._judge,
            )
        episodes = tuple(summaries)
        completed = tuple(item for item in episodes if item.status == "completed")
        return ApiPilotReport(
            manifest=manifest,
            manifest_digest=manifest_digest,
            episodes=episodes,
            score=score,
            completed_episodes=len(completed),
            failed_episodes=len(episodes) - len(completed),
            zero_spawn_episodes=sum(item.spawn_count == 0 for item in completed),
            total_spawn_count=sum(item.spawn_count for item in episodes),
            total_prompt_tokens=sum(item.prompt_tokens for item in episodes),
            total_completion_tokens=sum(item.completion_tokens for item in episodes),
            total_tokens=sum(item.total_tokens for item in episodes),
            total_latency_ms=sum(item.latency_ms for item in episodes),
        )

    def _build_manifest(self) -> PilotManifest:
        return PilotManifest(
            run_id=self._config.run_id,
            dataset_revision=XBENCH_UPSTREAM_REVISION,
            encrypted_source_sha256=self._dataset.source_digest,
            evaluator_revision=XBENCH_UPSTREAM_REVISION,
            evaluator_mode=(
                "development-minimax-judge"
                if self._judge is not None
                else "development-repeat-exact-only"
            ),
            task_ids=self._config.task_ids,
            repeats_per_task=self._config.repeats_per_task,
            policy_revision=self._policy.revision,
            search_backend=self._config.search_backend,
            search_revision=self._config.search_revision,
            judge_revision=self._judge.revision if self._judge is not None else None,
            max_spawn_per_episode=self._config.max_spawn_per_episode,
            max_concurrency=self._config.max_concurrency,
            repair_attempts=self._config.repair_attempts,
            sampling_params=self._config.sampling_params,
        )

    def _validate_search_revision(self, trace: EpisodeTrace) -> None:
        actual_revisions = {
            result.search_provider_revision
            for result in trace.sub_results
            if result.search_provider_revision is not None
        }
        if actual_revisions and actual_revisions != {self._config.search_revision}:
            raise ValueError("search provider revision does not match the pilot manifest")


def _completed_summary(
    task_id: TaskId,
    repeat_index: int,
    trace: EpisodeTrace,
    latency_ms: int,
) -> PilotEpisodeSummary:
    usages = [attempt.usage for attempt in trace.main_attempts]
    usages.extend(
        result.policy_usage for result in trace.sub_results if result.policy_usage is not None
    )
    return PilotEpisodeSummary(
        task_id=task_id,
        repeat_index=repeat_index,
        episode_id=trace.episode_id,
        status="completed",
        spawn_count=trace.spawn_count,
        successful_subs=sum(result.status == "success" for result in trace.sub_results),
        failed_subs=sum(result.status == "failed" for result in trace.sub_results),
        main_attempts=len(trace.main_attempts),
        invalid_main_attempts=sum(not attempt.valid for attempt in trace.main_attempts),
        event_count=len(trace.events),
        prompt_tokens=sum(usage.prompt_tokens for usage in usages),
        completion_tokens=sum(usage.completion_tokens for usage in usages),
        total_tokens=sum(usage.total_tokens for usage in usages),
        latency_ms=latency_ms,
        trainable=False,
    )


def _failed_summary(
    task_id: TaskId,
    repeat_index: int,
    episode_id: EpisodeId,
    latency_ms: int,
    error_code: str,
    *,
    failure: EpisodeRunError | None = None,
) -> PilotEpisodeSummary:
    return PilotEpisodeSummary(
        task_id=task_id,
        repeat_index=repeat_index,
        episode_id=episode_id,
        status="failed",
        error_code=error_code,
        spawn_count=failure.spawn_count if failure is not None else 0,
        successful_subs=failure.successful_subs if failure is not None else 0,
        failed_subs=failure.failed_subs if failure is not None else 0,
        main_attempts=failure.main_attempts if failure is not None else 0,
        invalid_main_attempts=failure.invalid_main_attempts if failure is not None else 0,
        event_count=failure.event_count if failure is not None else 0,
        prompt_tokens=failure.prompt_tokens if failure is not None else 0,
        completion_tokens=failure.completion_tokens if failure is not None else 0,
        total_tokens=failure.total_tokens if failure is not None else 0,
        latency_ms=latency_ms,
        trainable=False,
    )


def _elapsed_ms(start: float, end: float) -> int:
    return max(0, round((end - start) * 1000))


def _digest_model(model: BaseModel) -> str:
    payload = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
