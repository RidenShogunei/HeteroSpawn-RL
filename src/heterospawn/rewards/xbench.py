"""Ground-truth-safe xbench outcome reward for development training."""

from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import BaseModel, ConfigDict

from heterospawn.benchmarks.xbench import (
    XBENCH_UPSTREAM_REVISION,
    BenchmarkTask,
    XBenchDataset,
    XBenchTrainingOutcome,
)
from heterospawn.domain.ids import EpisodeId, TaskId
from heterospawn.domain.training import canonical_digest
from heterospawn.errors import BenchmarkDataError
from heterospawn.evaluation.judges import JudgeService
from heterospawn.orchestration.trainable_models import TrainableEpisodeTrace


class XBenchRewardAudit(BaseModel):
    """Safe cached reward evidence with no prompt, answer, or model text."""

    model_config = ConfigDict(frozen=True, strict=True)

    episode_id: EpisodeId
    response_digest: str
    outcome: XBenchTrainingOutcome


class XBenchOutcomeReward:
    """Binary exact/Judge correctness behind the generic outcome-reward protocol."""

    def __init__(
        self,
        dataset: XBenchDataset,
        *,
        task_ids: tuple[TaskId, ...],
        mode: Literal["development-exact-only", "development-judge"],
        judge: JudgeService | None = None,
    ) -> None:
        tasks = dataset.select_tasks(task_ids)
        if (mode == "development-judge") != (judge is not None):
            raise ValueError("development-judge mode and Judge service must agree")
        if judge is not None and judge.revision.comparable_to_official:
            raise BenchmarkDataError("official-comparable Judges cannot be used for training")
        self._dataset = dataset
        self._tasks = {task.task_id: task for task in tasks}
        self._judge = judge
        self._mode = mode
        self._audits: dict[EpisodeId, XBenchRewardAudit] = {}
        self._inflight: dict[
            EpisodeId,
            tuple[TaskId, str, asyncio.Task[XBenchTrainingOutcome]],
        ] = {}
        self._lock = asyncio.Lock()
        self._revision = canonical_digest(
            {
                "adapter": "xbench-binary-training-outcome-v1",
                "dataset_revision": XBENCH_UPSTREAM_REVISION,
                "encrypted_source_digest": dataset.source_digest,
                "task_ids": sorted(str(task_id) for task_id in task_ids),
                "mode": mode,
                "judge_revision": (
                    judge.revision.model_dump(mode="json") if judge is not None else None
                ),
            }
        )

    @property
    def revision(self) -> str:
        return self._revision

    async def score(self, task: BenchmarkTask, trace: TrainableEpisodeTrace) -> float:
        if trace.task_id != task.task_id:
            raise BenchmarkDataError("reward trace and task IDs do not match")
        expected = self._tasks.get(task.task_id)
        if expected is None or expected != task:
            raise BenchmarkDataError("reward task is outside the selected training task set")
        if trace.status != "success" or trace.answer is None:
            raise BenchmarkDataError("outcome reward requires a successful terminal answer")
        response_digest = canonical_digest({"response": trace.answer})

        async with self._lock:
            cached = self._audits.get(trace.episode_id)
            inflight = self._inflight.get(trace.episode_id)
        if cached is not None:
            self._validate_identity(
                cached.outcome.task_id, cached.response_digest, task, response_digest
            )
            return 1.0 if cached.outcome.correct else 0.0

        if inflight is None:
            evaluation = asyncio.create_task(
                self._dataset.evaluate_training_outcome(
                    task,
                    trace.answer,
                    judge=self._judge,
                    request_id=f"{trace.episode_id}:xbench-training-reward",
                )
            )
            inflight = (task.task_id, response_digest, evaluation)
            async with self._lock:
                existing_inflight = self._inflight.setdefault(trace.episode_id, inflight)
            if existing_inflight != inflight:
                evaluation.cancel()
                await asyncio.gather(evaluation, return_exceptions=True)
                inflight = existing_inflight
        inflight_task_id, inflight_response_digest, evaluation = inflight
        self._validate_identity(
            inflight_task_id,
            inflight_response_digest,
            task,
            response_digest,
        )
        try:
            outcome = await asyncio.shield(evaluation)
        finally:
            if evaluation.done():
                async with self._lock:
                    if self._inflight.get(trace.episode_id) == inflight:
                        del self._inflight[trace.episode_id]

        audit = XBenchRewardAudit(
            episode_id=trace.episode_id,
            response_digest=response_digest,
            outcome=outcome,
        )
        async with self._lock:
            existing = self._audits.setdefault(trace.episode_id, audit)
        self._validate_identity(
            existing.outcome.task_id,
            existing.response_digest,
            task,
            response_digest,
        )
        return 1.0 if existing.outcome.correct else 0.0

    async def audit(self, episode_id: EpisodeId) -> XBenchRewardAudit:
        async with self._lock:
            try:
                return self._audits[episode_id]
            except KeyError:
                raise BenchmarkDataError("reward audit does not exist") from None

    @property
    def mode(self) -> Literal["development-exact-only", "development-judge"]:
        return self._mode

    @staticmethod
    def _validate_identity(
        cached_task_id: TaskId,
        cached_response_digest: str,
        task: BenchmarkTask,
        response_digest: str,
    ) -> None:
        if cached_task_id != task.task_id:
            raise BenchmarkDataError("episode reward cache belongs to another task")
        if cached_response_digest != response_digest:
            raise BenchmarkDataError("episode reward cache belongs to another response")
