"""Rewarded full-system rollout groups integrated with fresh alternating updates."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from heterospawn.benchmarks.xbench import BenchmarkTask
from heterospawn.domain.ids import EpisodeId, PolicyId, RolloutId, TaskId
from heterospawn.domain.training import PolicyTrainingBatch, TrainingPhase, canonical_digest
from heterospawn.domain.versions import RolloutRevision
from heterospawn.errors import ConfigurationError
from heterospawn.training.base import TrainingBackend
from heterospawn.training.batch import (
    OutcomeAdvantageGroup,
    TrainingBatchBuilder,
    normalize_outcome_advantages,
)
from heterospawn.training.coordinator import AlternatingCoordinator, CycleResult
from heterospawn.training.registry import PolicyRegistry

if TYPE_CHECKING:
    from heterospawn.orchestration.trainable_episode import TrainableEpisodeOrchestrator
    from heterospawn.orchestration.trainable_models import TrainableEpisodeTrace


class OutcomeRewardService(Protocol):
    """Benchmark-owned outcome score; orchestration costs are composed separately."""

    @property
    def revision(self) -> str: ...

    async def score(self, task: BenchmarkTask, trace: TrainableEpisodeTrace) -> float: ...


class RewardConfig(BaseModel):
    """Explicit non-outcome reward terms for one experiment."""

    model_config = ConfigDict(frozen=True, strict=True)

    invalid_action_penalty: float = Field(default=0.0, ge=0.0)
    spawn_cost: float = Field(default=0.0, ge=0.0)
    sub_failure_penalty: float = Field(default=0.0, ge=0.0)
    failed_episode_outcome: float = -1.0


class EpisodeReward(BaseModel):
    """Auditable reward composition for one full system rollout."""

    model_config = ConfigDict(frozen=True, strict=True)

    task_id: TaskId
    episode_id: EpisodeId
    reward_revision: str = Field(min_length=1)
    outcome_reward: float
    invalid_action_component: float = Field(le=0.0)
    spawn_cost_component: float = Field(le=0.0)
    sub_failure_component: float = Field(le=0.0)
    total: float

    @model_validator(mode="after")
    def total_must_match_components(self) -> EpisodeReward:
        expected = (
            self.outcome_reward
            + self.invalid_action_component
            + self.spawn_cost_component
            + self.sub_failure_component
        )
        if not math.isclose(self.total, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("episode reward total does not match its components")
        if not all(
            math.isfinite(value)
            for value in (
                self.outcome_reward,
                self.invalid_action_component,
                self.spawn_cost_component,
                self.sub_failure_component,
                self.total,
            )
        ):
            raise ValueError("episode rewards must be finite")
        return self


class RewardComposer:
    """Combines a caller-owned outcome score with configured action costs."""

    def __init__(self, outcome: OutcomeRewardService, config: RewardConfig) -> None:
        if not outcome.revision:
            raise ValueError("outcome reward service revision cannot be empty")
        self._outcome = outcome
        self._config = config
        self._revision = (
            f"{outcome.revision}+costs-{canonical_digest(config.model_dump(mode='json'))[:12]}"
        )

    @property
    def revision(self) -> str:
        return self._revision

    async def score(
        self,
        task: BenchmarkTask,
        trace: TrainableEpisodeTrace,
    ) -> EpisodeReward:
        outcome_reward = (
            await self._outcome.score(task, trace)
            if trace.status == "success"
            else self._config.failed_episode_outcome
        )
        invalid_component = -self._config.invalid_action_penalty * trace.invalid_main_attempts
        spawn_component = -self._config.spawn_cost * trace.spawn_count
        failure_component = -self._config.sub_failure_penalty * trace.failed_subs
        return EpisodeReward(
            task_id=task.task_id,
            episode_id=trace.episode_id,
            reward_revision=self._revision,
            outcome_reward=outcome_reward,
            invalid_action_component=invalid_component,
            spawn_cost_component=spawn_component,
            sub_failure_component=failure_component,
            total=outcome_reward + invalid_component + spawn_component + failure_component,
        )


@dataclass(frozen=True)
class TaskRolloutGroup:
    """One task's complete rollout baseline and normalized advantages."""

    task_id: TaskId
    traces: tuple[TrainableEpisodeTrace, ...]
    rewards: tuple[EpisodeReward, ...]
    advantages: OutcomeAdvantageGroup


@dataclass(frozen=True)
class PhaseRolloutResult:
    """Durable evidence used to create one phase's training batch."""

    phase: TrainingPhase
    target_policy_id: PolicyId
    policy_revisions: tuple[tuple[PolicyId, RolloutRevision], ...]
    groups: tuple[TaskRolloutGroup, ...]
    batch: PolicyTrainingBatch

    @property
    def degenerate_groups(self) -> int:
        return sum(group.advantages.degenerate for group in self.groups)


@dataclass(frozen=True)
class TrainableCycleResult:
    """Alternating update result plus the exact phase rollout evidence."""

    updates: CycleResult
    phases: tuple[PhaseRolloutResult, ...]


class TrainableRolloutBatchFactory:
    """Creates fresh, task-normalized full-system rollout batches for a cycle."""

    def __init__(
        self,
        *,
        cycle_id: str,
        tasks: tuple[BenchmarkTask, ...],
        rollouts_per_task: int,
        orchestrator: TrainableEpisodeOrchestrator,
        reward: RewardComposer,
        registry: PolicyRegistry,
        batch_builder: TrainingBatchBuilder | None = None,
    ) -> None:
        if not cycle_id:
            raise ValueError("cycle_id cannot be empty")
        if not tasks:
            raise ValueError("at least one task is required")
        if len({task.task_id for task in tasks}) != len(tasks):
            raise ValueError("cycle tasks must be unique")
        if rollouts_per_task < 2:
            raise ValueError("each task/phase requires at least two full system rollouts")
        self._cycle_id = cycle_id
        self._tasks = tasks
        self._rollouts_per_task = rollouts_per_task
        self._orchestrator = orchestrator
        self._reward = reward
        self._registry = registry
        self._batch_builder = batch_builder or TrainingBatchBuilder()
        self._results: list[PhaseRolloutResult] = []
        self._completed_phases: set[TrainingPhase] = set()

    @property
    def results(self) -> tuple[PhaseRolloutResult, ...]:
        return tuple(self._results)

    async def build(
        self,
        phase: TrainingPhase,
        policy_revisions: tuple[tuple[PolicyId, RolloutRevision], ...],
    ) -> PolicyTrainingBatch:
        if phase in self._completed_phases:
            raise RuntimeError(f"{phase} rollout batch was already built")
        if policy_revisions != self._registry.snapshot():
            raise ConfigurationError("phase rollout must use the current registry snapshot")
        target = self._registry.target_for_phase(phase)
        if target is None:
            raise ConfigurationError(f"{phase} has no trainable target")
        revision_map = dict(policy_revisions)

        groups = tuple(
            await asyncio.gather(
                *(
                    self._run_task_group(
                        task=task,
                        phase=phase,
                        policy_revisions=policy_revisions,
                    )
                    for task in self._tasks
                )
            )
        )
        all_steps = tuple(
            step for group in groups for trace in group.traces for step in trace.model_steps
        )
        episode_advantages = {
            episode_id: advantage
            for group in groups
            for episode_id, advantage in group.advantages.advantages
        }
        batch = self._batch_builder.build(
            batch_id=f"{self._cycle_id}:{phase}",
            phase=phase,
            target_policy_id=target,
            expected_base_version=revision_map[target].weight_version,
            steps=all_steps,
            episode_advantages=episode_advantages,
        )
        result = PhaseRolloutResult(
            phase=phase,
            target_policy_id=target,
            policy_revisions=policy_revisions,
            groups=groups,
            batch=batch,
        )
        self._results.append(result)
        self._completed_phases.add(phase)
        return batch

    async def _run_task_group(
        self,
        *,
        task: BenchmarkTask,
        phase: TrainingPhase,
        policy_revisions: tuple[tuple[PolicyId, RolloutRevision], ...],
    ) -> TaskRolloutGroup:
        traces = tuple(
            await asyncio.gather(
                *(
                    self._orchestrator.run(
                        task,
                        episode_id=self._episode_id(phase, task.task_id, index),
                        rollout_id=self._rollout_id(phase, task.task_id, index),
                        policy_revisions=policy_revisions,
                    )
                    for index in range(self._rollouts_per_task)
                )
            )
        )
        rewards = tuple(
            await asyncio.gather(*(self._reward.score(task, trace) for trace in traces))
        )
        reward_map = {reward.episode_id: reward.total for reward in rewards}
        advantages = normalize_outcome_advantages(reward_map)
        return TaskRolloutGroup(
            task_id=task.task_id,
            traces=traces,
            rewards=rewards,
            advantages=advantages,
        )

    def _episode_id(
        self,
        phase: TrainingPhase,
        task_id: TaskId,
        index: int,
    ) -> EpisodeId:
        return EpisodeId(f"{self._cycle_id}:{phase}:{task_id}:episode-{index}")

    def _rollout_id(
        self,
        phase: TrainingPhase,
        task_id: TaskId,
        index: int,
    ) -> RolloutId:
        return RolloutId(f"{self._cycle_id}:{phase}:{task_id}:rollout-{index}")


class TrainableAlternatingCycleRunner:
    """Runs a complete fresh-rollout alternating cycle and exposes its evidence."""

    def __init__(
        self,
        registry: PolicyRegistry,
        backend: TrainingBackend,
        orchestrator: TrainableEpisodeOrchestrator,
        reward: RewardComposer,
        *,
        rollouts_per_task: int = 8,
    ) -> None:
        if rollouts_per_task < 2:
            raise ValueError("rollouts_per_task must be at least two")
        self._registry = registry
        self._coordinator = AlternatingCoordinator(registry, backend)
        self._orchestrator = orchestrator
        self._reward = reward
        self._rollouts_per_task = rollouts_per_task

    async def run_cycle(
        self,
        *,
        cycle_id: str,
        tasks: tuple[BenchmarkTask, ...],
    ) -> TrainableCycleResult:
        factory = TrainableRolloutBatchFactory(
            cycle_id=cycle_id,
            tasks=tasks,
            rollouts_per_task=self._rollouts_per_task,
            orchestrator=self._orchestrator,
            reward=self._reward,
            registry=self._registry,
        )
        updates = await self._coordinator.run_cycle(factory.build)
        return TrainableCycleResult(updates=updates, phases=factory.results)
