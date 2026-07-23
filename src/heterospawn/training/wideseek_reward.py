"""Auditable role-specific WideSeek reward composition."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, model_validator

from heterospawn.domain.ids import EpisodeId, TaskId
from heterospawn.domain.tasks import ResearchTask
from heterospawn.domain.training import TrainingPhase, canonical_digest
from heterospawn.evaluation.wideseek import WideSeekEvaluation, WideSeekEvaluator
from heterospawn.orchestration.trainable_models import TrainableEpisodeTrace


class WideSeekRewardConfig(BaseModel):
    """Versioned reward coefficients shared by both training topologies."""

    model_config = ConfigDict(frozen=True, strict=True)

    format_reward: float = Field(default=0.1, ge=0.0)
    access_credit: float = Field(default=0.1, ge=0.0)
    length_limit: int = Field(default=5000, ge=1)
    max_length_limit: int = Field(default=7000, ge=2)
    length_penalty: float = Field(default=0.1, ge=0.0)
    context_failure_penalty: float = Field(default=0.1, ge=0.0)
    spawn_cost: float = Field(default=0.0, ge=0.0)
    search_cost: float = Field(default=0.0, ge=0.0)
    token_cost: float = Field(default=0.0, ge=0.0)
    invalid_action_cost: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def length_limits_must_increase(self) -> WideSeekRewardConfig:
        if self.max_length_limit <= self.length_limit:
            raise ValueError("max_length_limit must exceed length_limit")
        return self


class RoleRewardTotals(BaseModel):
    """Training targets for shared and independent-policy topologies."""

    model_config = ConfigDict(frozen=True, strict=True)

    shared: float
    main: float
    sub: float


class WideSeekRewardBreakdown(BaseModel):
    """Per-episode reward facts with evaluator and role identity."""

    model_config = ConfigDict(frozen=True, strict=True)

    task_id: TaskId
    episode_id: EpisodeId
    reward_revision: str = Field(min_length=1)
    evaluator_revision: str = Field(min_length=1)
    outcome: float = Field(ge=0.0, le=1.0)
    format_component: float = Field(ge=0.0)
    search_credit_component: float = Field(ge=0.0)
    length_component: float = Field(le=0.0)
    context_component: float = Field(le=0.0)
    spawn_cost_component: float = Field(le=0.0)
    search_cost_component: float = Field(le=0.0)
    token_cost_component: float = Field(le=0.0)
    invalid_action_component: float = Field(le=0.0)
    spawn_count: int = Field(ge=0)
    search_calls: int = Field(ge=0)
    access_calls: int = Field(ge=0)
    generated_tokens: int = Field(ge=0)
    evaluation: WideSeekEvaluation
    role_totals: RoleRewardTotals

    @model_validator(mode="after")
    def role_totals_must_match_components(self) -> WideSeekRewardBreakdown:
        shared = (
            self.outcome
            + self.format_component
            + self.search_credit_component
            + self.length_component
            + self.context_component
        )
        main = (
            shared
            + self.spawn_cost_component
            + self.search_cost_component
            + self.token_cost_component
            + self.invalid_action_component
        )
        if not math.isclose(self.role_totals.shared, shared, abs_tol=1e-12):
            raise ValueError("shared WideSeek reward total does not match components")
        if not math.isclose(self.role_totals.main, main, abs_tol=1e-12):
            raise ValueError("Main WideSeek reward total does not match components")
        if not math.isclose(self.role_totals.sub, self.outcome, abs_tol=1e-12):
            raise ValueError("Sub WideSeek reward must equal system outcome in MVP")
        return self


class WideSeekRewardService:
    """Evaluates terminal answers and composes topology-specific reward totals."""

    def __init__(
        self,
        evaluator: WideSeekEvaluator,
        config: WideSeekRewardConfig,
    ) -> None:
        self._evaluator = evaluator
        self._config = config
        self._revision = canonical_digest(
            {
                "schema": "heterospawn-wideseek-reward-v1",
                "evaluator_revision": evaluator.revision,
                "judge_revision": (
                    evaluator.judge_revision.model_dump(mode="json")
                    if isinstance(evaluator.judge_revision, BaseModel)
                    else None
                ),
                "config": config.model_dump(mode="json"),
            }
        )

    @property
    def revision(self) -> str:
        return self._revision

    async def score(
        self,
        task: ResearchTask,
        trace: TrainableEpisodeTrace,
    ) -> float:
        """Compatibility score for independent Main updates."""

        return (await self.score_breakdown(task, trace)).role_totals.main

    async def score_for_phase(
        self,
        task: ResearchTask,
        trace: TrainableEpisodeTrace,
        phase: TrainingPhase,
    ) -> float:
        """Select the topology-specific total before task-level normalization."""

        totals = (await self.score_breakdown(task, trace)).role_totals
        if phase == "joint_update":
            return totals.shared
        if phase == "main_update":
            return totals.main
        return totals.sub

    async def score_breakdown(
        self,
        task: ResearchTask,
        trace: TrainableEpisodeTrace,
    ) -> WideSeekRewardBreakdown:
        response = trace.answer if trace.status == "success" and trace.answer else "unanswered"
        evaluation = await self._evaluator.evaluate(
            task,
            response,
            request_id=f"{trace.episode_id}:reward",
        )
        search_calls = sum(item.tool_name == "search" for item in trace.tool_outcomes)
        access_calls = sum(item.tool_name == "access" for item in trace.tool_outcomes)
        successful_access = any(
            item.tool_name == "access" and item.status == "success" for item in trace.tool_outcomes
        )
        generated_tokens = sum(len(step.response_ids) for step in trace.model_steps)
        max_response_tokens = max(
            (len(step.response_ids) for step in trace.model_steps),
            default=0,
        )
        length_ratio = max(
            0.0,
            min(
                1.0,
                (max_response_tokens - self._config.length_limit)
                / (self._config.max_length_limit - self._config.length_limit),
            ),
        )
        format_component = self._config.format_reward if evaluation.format_ok else 0.0
        search_credit_component = self._config.access_credit if successful_access else 0.0
        length_component = -length_ratio * self._config.length_penalty
        context_failed = any(step.stop_reason == "length" for step in trace.model_steps)
        context_component = -self._config.context_failure_penalty if context_failed else 0.0
        spawn_component = -self._config.spawn_cost * trace.spawn_count
        search_component = -self._config.search_cost * (search_calls + access_calls)
        token_component = -self._config.token_cost * generated_tokens
        invalid_component = -self._config.invalid_action_cost * trace.invalid_main_attempts
        shared_total = (
            evaluation.outcome_score
            + format_component
            + search_credit_component
            + length_component
            + context_component
        )
        main_total = (
            shared_total + spawn_component + search_component + token_component + invalid_component
        )
        return WideSeekRewardBreakdown(
            task_id=task.task_id,
            episode_id=trace.episode_id,
            reward_revision=self._revision,
            evaluator_revision=self._evaluator.revision,
            outcome=evaluation.outcome_score,
            format_component=format_component,
            search_credit_component=search_credit_component,
            length_component=length_component,
            context_component=context_component,
            spawn_cost_component=spawn_component,
            search_cost_component=search_component,
            token_cost_component=token_component,
            invalid_action_component=invalid_component,
            spawn_count=trace.spawn_count,
            search_calls=search_calls,
            access_calls=access_calls,
            generated_tokens=generated_tokens,
            evaluation=evaluation,
            role_totals=RoleRewardTotals(
                shared=shared_total,
                main=main_total,
                sub=evaluation.outcome_score,
            ),
        )
