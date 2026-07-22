"""Strict action, event, and trace models for the API validation slice."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from heterospawn.domain.ids import AgentInstanceId, EpisodeId, PolicyId, TaskId
from heterospawn.errors import InvalidActionError
from heterospawn.policies.base import ExternalModelRevision


class AnswerAction(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    kind: Literal["answer"]
    answer: str = Field(min_length=1)


class SpawnAction(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    kind: Literal["spawn"]
    subtasks: tuple[str, ...] = Field(min_length=1)


MainAction = Annotated[AnswerAction | SpawnAction, Field(discriminator="kind")]
_ACTION_ADAPTER: TypeAdapter[MainAction] = TypeAdapter(MainAction)


class BudgetSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    capacity: int = Field(ge=1)
    reserved: int = Field(ge=0)
    committed: int = Field(ge=0)


class MainAttempt(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    phase: Literal["initial", "final"]
    attempt_index: int = Field(ge=0)
    content: str
    raw_response_digest: str
    valid: bool
    error_code: str | None = None


class SubResult(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    agent_instance_id: AgentInstanceId
    subtask: str
    status: Literal["success", "failed"]
    content: str
    search_provider_request_id: str | None = None
    policy_provider_request_id: str | None = None
    error_code: str | None = None


class EpisodeEvent(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    event_index: int = Field(ge=0)
    kind: Literal["main_output", "sub_result"]
    agent_instance_id: AgentInstanceId
    causal_event_indices: tuple[int, ...]
    status: Literal["valid", "invalid", "success", "failed"]
    phase: Literal["initial", "sub", "final"]
    detail: str
    environment: BudgetSnapshot


class EpisodeTrace(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    task_id: TaskId
    episode_id: EpisodeId
    answer: str
    spawn_count: int = Field(ge=0)
    main_attempts: tuple[MainAttempt, ...]
    sub_results: tuple[SubResult, ...]
    events: tuple[EpisodeEvent, ...]
    policy_revisions: tuple[tuple[PolicyId, ExternalModelRevision], ...]
    trainable: Literal[False] = False


def parse_main_action(content: str) -> MainAction:
    """Parse a strict JSON action; an empty spawn is rejected by schema."""

    try:
        return _ACTION_ADAPTER.validate_json(content, strict=True)
    except ValidationError:
        raise InvalidActionError("main output is not a valid ANSWER or non-empty SPAWN") from None
