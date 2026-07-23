"""Immutable event-sourced records for trainable Main/Sub episodes."""

from __future__ import annotations

import json
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from heterospawn.domain.ids import (
    AgentInstanceId,
    EpisodeId,
    PolicyId,
    RolloutId,
    StepId,
    TaskId,
)
from heterospawn.domain.tasks import ResearchTask
from heterospawn.domain.training import TrajectoryStep
from heterospawn.domain.versions import AgentRole, RolloutRevision
from heterospawn.orchestration.models import BudgetSnapshot


class TrainableEnvironmentSnapshot(BaseModel):
    """Environment identity observed when an event became durable."""

    model_config = ConfigDict(frozen=True, strict=True)

    budget: BudgetSnapshot
    search_provider_revision: str | None = None
    search_response_digest: str | None = None
    prompt_revision: str | None = None
    tool_schema_revision: str | None = None
    parser_revision: str | None = None


class TrainableEpisodeEvent(BaseModel):
    """One deterministic event in the episode fact stream."""

    model_config = ConfigDict(frozen=True, strict=True)

    event_index: int = Field(ge=0)
    step_id: StepId
    kind: Literal["model", "search", "access", "sub_failure"]
    agent_role: AgentRole
    agent_instance_id: AgentInstanceId
    causal_step_ids: tuple[StepId, ...] = ()
    status: Literal["valid", "invalid", "success", "failed"]
    phase: Literal["initial", "main", "sub", "final"]
    payload_digest: str = Field(min_length=1)
    environment: TrainableEnvironmentSnapshot


class TrainableMainAttempt(BaseModel):
    """Decoded Main output used for action interpretation alongside its raw MODEL step."""

    model_config = ConfigDict(frozen=True, strict=True)

    phase: Literal["initial", "main", "final"]
    round_index: int = Field(default=0, ge=0)
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
    spawn_round: int = Field(default=0, ge=0)
    status: Literal["success", "failed"]
    content: str
    search_step_id: StepId | None = None
    model_step_id: StepId | None = None
    tool_step_ids: tuple[StepId, ...] = ()
    model_step_ids: tuple[StepId, ...] = ()
    error_code: str | None = None


class SpawnRoundRecord(BaseModel):
    """One accepted Main delegation decision and its child instances."""

    model_config = ConfigDict(frozen=True, strict=True)

    round_index: int = Field(ge=0)
    main_step_id: StepId
    agent_instance_ids: tuple[AgentInstanceId, ...] = Field(min_length=1, max_length=4)


class ToolOutcomeRecord(BaseModel):
    """Auditable Search/Access result in deterministic request order."""

    model_config = ConfigDict(frozen=True, strict=True)

    step_id: StepId
    agent_instance_id: AgentInstanceId
    spawn_round: int = Field(ge=0)
    sub_turn: int = Field(ge=0)
    request_index: int = Field(ge=0)
    tool_name: Literal["search", "access"]
    status: Literal["success", "failed"]
    request_json: str = Field(min_length=2)
    request_digest: str = Field(min_length=1)
    result_json: str = Field(min_length=2)
    result_digest: str = Field(min_length=1)
    provider_response_digest: str | None = None
    query: str | None = None
    url: str | None = None
    source_search_step_id: StepId | None = None
    provider_revision: str | None = None
    error_code: str | None = None

    @model_validator(mode="after")
    def payload_digests_must_match(self) -> ToolOutcomeRecord:
        from heterospawn.domain.training import canonical_digest

        for payload, digest, label in (
            (self.request_json, self.request_digest, "request"),
            (self.result_json, self.result_digest, "result"),
        ):
            try:
                value = json.loads(payload)
            except json.JSONDecodeError:
                raise ValueError(f"tool {label} payload must be valid JSON") from None
            canonical = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if canonical != payload:
                raise ValueError(f"tool {label} payload must use canonical JSON")
            if canonical_digest(value) != digest:
                raise ValueError(f"tool {label} digest does not match payload")
        return self


class EvidenceRecord(BaseModel):
    """Evidence with explicit tool and optional model producers."""

    model_config = ConfigDict(frozen=True, strict=True)

    agent_instance_id: AgentInstanceId
    subtask: str = Field(min_length=1)
    content: str
    producer_tool_step_id: StepId | None = None
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
    spawn_rounds: tuple[SpawnRoundRecord, ...] = ()
    main_attempts: tuple[TrainableMainAttempt, ...]
    sub_outcomes: tuple[TrainableSubOutcome, ...]
    tool_outcomes: tuple[ToolOutcomeRecord, ...] = ()
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
        if sum(len(round_.agent_instance_ids) for round_ in self.spawn_rounds) not in (
            0,
            self.spawn_count,
        ):
            raise ValueError("spawn rounds must account for every Sub outcome")
        if any(round_.main_step_id not in model_ids for round_ in self.spawn_rounds):
            raise ValueError("spawn rounds must reference retained Main MODEL steps")
        tool_ids = {outcome.step_id for outcome in self.tool_outcomes}
        if len(tool_ids) != len(self.tool_outcomes):
            raise ValueError("tool outcome step IDs must be unique")
        for outcome in self.sub_outcomes:
            if outcome.search_step_id is not None:
                search_event = event_by_step.get(outcome.search_step_id)
                if search_event is None or search_event.kind != "search":
                    raise ValueError("Sub outcomes must reference retained SEARCH steps")
            if outcome.model_step_id is not None and outcome.model_step_id not in model_ids:
                raise ValueError("Sub outcomes must reference retained MODEL steps")
            if any(step_id not in tool_ids for step_id in outcome.tool_step_ids):
                raise ValueError("Sub outcomes must reference retained tool outcomes")
            if any(step_id not in model_ids for step_id in outcome.model_step_ids):
                raise ValueError("Sub outcomes must reference retained MODEL steps")
        for tool_outcome in self.tool_outcomes:
            event = event_by_step.get(tool_outcome.step_id)
            if event is None or event.kind != tool_outcome.tool_name:
                raise ValueError("tool outcomes must map to matching retained tool events")
            if tool_outcome.source_search_step_id is not None:
                source = event_by_step.get(tool_outcome.source_search_step_id)
                if source is None or source.kind != "search":
                    raise ValueError("Access provenance must reference a retained SEARCH step")
        for record in self.evidence:
            if record.producer_tool_step_id is not None:
                tool_event = event_by_step.get(record.producer_tool_step_id)
                if tool_event is None or tool_event.kind not in ("search", "access"):
                    raise ValueError("evidence must reference its producer tool step")
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


class TrainableEpisodeRunner(Protocol):
    """Common rollout surface used by one-round and WideSeek multi-round environments."""

    async def run(
        self,
        task: ResearchTask,
        episode_id: EpisodeId,
        rollout_id: RolloutId,
        policy_revisions: tuple[tuple[PolicyId, RolloutRevision], ...],
    ) -> TrainableEpisodeTrace: ...
