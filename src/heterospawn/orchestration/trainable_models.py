"""Immutable event-sourced records for trainable Main/Sub episodes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from heterospawn.domain.ids import (
    AgentInstanceId,
    EpisodeId,
    PolicyId,
    RolloutId,
    StepId,
    TaskId,
)
from heterospawn.domain.training import TrajectoryStep
from heterospawn.domain.versions import AgentRole, RolloutRevision
from heterospawn.orchestration.models import BudgetSnapshot


class TrainableEnvironmentSnapshot(BaseModel):
    """Environment identity observed when an event became durable."""

    model_config = ConfigDict(frozen=True, strict=True)

    budget: BudgetSnapshot
    search_provider_revision: str | None = None
    search_response_digest: str | None = None


class TrainableEpisodeEvent(BaseModel):
    """One deterministic event in the episode fact stream."""

    model_config = ConfigDict(frozen=True, strict=True)

    event_index: int = Field(ge=0)
    step_id: StepId
    kind: Literal["model", "search", "sub_failure"]
    agent_role: AgentRole
    agent_instance_id: AgentInstanceId
    causal_step_ids: tuple[StepId, ...] = ()
    status: Literal["valid", "invalid", "success", "failed"]
    phase: Literal["initial", "sub", "final"]
    payload_digest: str = Field(min_length=1)
    environment: TrainableEnvironmentSnapshot


class TrainableMainAttempt(BaseModel):
    """Decoded Main output used for action interpretation alongside its raw MODEL step."""

    model_config = ConfigDict(frozen=True, strict=True)

    phase: Literal["initial", "final"]
    attempt_index: int = Field(ge=0)
    step_id: StepId
    content: str
    valid: bool
    action_kind: Literal["answer", "spawn"] | None = None
    error_code: str | None = None


class TrainableSubOutcome(BaseModel):
    """Structured Sub completion; failures never cancel sibling executions."""

    model_config = ConfigDict(frozen=True, strict=True)

    agent_instance_id: AgentInstanceId
    subtask: str = Field(min_length=1)
    status: Literal["success", "failed"]
    content: str
    search_step_id: StepId
    model_step_id: StepId | None = None
    error_code: str | None = None


class EvidenceRecord(BaseModel):
    """Evidence with explicit tool and optional model producers."""

    model_config = ConfigDict(frozen=True, strict=True)

    agent_instance_id: AgentInstanceId
    subtask: str = Field(min_length=1)
    content: str
    producer_tool_step_id: StepId
    producer_model_step_id: StepId | None = None


class TrainableEpisodeTrace(BaseModel):
    """A complete success or structured failure retaining all generated trajectories."""

    model_config = ConfigDict(frozen=True, strict=True)

    task_id: TaskId
    episode_id: EpisodeId
    rollout_id: RolloutId
    status: Literal["success", "failed"]
    answer: str | None = None
    failure_code: str | None = None
    spawn_count: int = Field(ge=0)
    main_attempts: tuple[TrainableMainAttempt, ...]
    sub_outcomes: tuple[TrainableSubOutcome, ...]
    evidence: tuple[EvidenceRecord, ...]
    model_steps: tuple[TrajectoryStep, ...]
    events: tuple[TrainableEpisodeEvent, ...]
    policy_revisions: tuple[tuple[PolicyId, RolloutRevision], ...]

    @model_validator(mode="after")
    def trace_must_be_internally_consistent(self) -> TrainableEpisodeTrace:
        if tuple(event.event_index for event in self.events) != tuple(range(len(self.events))):
            raise ValueError("trainable episode event indices must be contiguous")
        event_by_step = {event.step_id: event for event in self.events}
        if len(event_by_step) != len(self.events):
            raise ValueError("trainable episode step IDs must be unique")
        model_ids = {step.step_id for step in self.model_steps}
        if len(model_ids) != len(self.model_steps):
            raise ValueError("trainable MODEL step IDs must be unique")
        for step in self.model_steps:
            event = event_by_step.get(step.step_id)
            if event is None or event.kind != "model" or event.event_index != step.event_index:
                raise ValueError("every trajectory step must map to its MODEL event")
        for event in self.events:
            for cause in event.causal_step_ids:
                cause_event = event_by_step.get(cause)
                if cause_event is None or cause_event.event_index >= event.event_index:
                    raise ValueError("event causes must reference earlier retained steps")
        if any(attempt.step_id not in model_ids for attempt in self.main_attempts):
            raise ValueError("Main attempts must reference retained MODEL steps")
        if self.spawn_count != len(self.sub_outcomes):
            raise ValueError("spawn_count must equal the number of Sub outcomes")
        for outcome in self.sub_outcomes:
            search_event = event_by_step.get(outcome.search_step_id)
            if search_event is None or search_event.kind != "search":
                raise ValueError("Sub outcomes must reference retained SEARCH steps")
            if outcome.model_step_id is not None and outcome.model_step_id not in model_ids:
                raise ValueError("Sub outcomes must reference retained MODEL steps")
        for record in self.evidence:
            tool_event = event_by_step.get(record.producer_tool_step_id)
            if tool_event is None or tool_event.kind != "search":
                raise ValueError("evidence must reference its producer SEARCH step")
            if (
                record.producer_model_step_id is not None
                and record.producer_model_step_id not in model_ids
            ):
                raise ValueError("evidence model producer must reference a retained MODEL step")
        if self.status == "success" and not self.answer:
            raise ValueError("successful trainable episode requires an answer")
        if self.status == "failed":
            if self.failure_code is None:
                raise ValueError("failed trainable episode requires a failure code")
            if self.answer is not None:
                raise ValueError("failed trainable episode cannot contain an answer")
        revision_map = dict(self.policy_revisions)
        if len(revision_map) != len(self.policy_revisions):
            raise ValueError("trainable episode policy revisions must be unique")
        if any(
            revision_map.get(step.policy_id) != step.rollout_revision for step in self.model_steps
        ):
            raise ValueError("MODEL steps must match the episode policy revision map")
        return self

    @property
    def invalid_main_attempts(self) -> int:
        return sum(not attempt.valid for attempt in self.main_attempts)

    @property
    def failed_subs(self) -> int:
        return sum(outcome.status == "failed" for outcome in self.sub_outcomes)
