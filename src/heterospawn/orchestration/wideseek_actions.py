"""WideSeek-compatible tool-call schemas and strict turn parsing."""

from __future__ import annotations

import json
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from heterospawn.domain.training import canonical_digest
from heterospawn.errors import InvalidActionError
from heterospawn.policies.trainable import ToolDefinition

WIDESEEK_UPSTREAM_REVISION = "d9f3d8a9db4d7aad1d641029293295503dd3eb2c"
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_ANY_TOOL_TAG_RE = re.compile(r"</?tool_call>")

SUBTASK_TOOL = ToolDefinition(
    name="subtask",
    description="Delegate one independent research subtask to a worker.",
    parameters_json=(
        '{"type":"object","properties":{"subtask":{"type":"string"}},'
        '"required":["subtask"],"additionalProperties":false}'
    ),
)
SEARCH_TOOL = ToolDefinition(
    name="search",
    description="Search the pinned offline corpus for relevant documents.",
    parameters_json=(
        '{"type":"object","properties":{"query":{"type":"string"},'
        '"topk":{"type":"integer","minimum":1,"maximum":20}},'
        '"required":["query"],"additionalProperties":false}'
    ),
)
ACCESS_TOOL = ToolDefinition(
    name="access",
    description="Read one URL previously returned by this worker's search.",
    parameters_json=(
        '{"type":"object","properties":{"url":{"type":"string"},'
        '"info_to_extract":{"type":"string"}},'
        '"required":["url","info_to_extract"],"additionalProperties":false}'
    ),
)

MAIN_TOOLS = (SUBTASK_TOOL,)
SUB_TOOLS = (SEARCH_TOOL, ACCESS_TOOL)
WIDESEEK_TOOL_SCHEMA_REVISION = canonical_digest(
    {
        "upstream_revision": WIDESEEK_UPSTREAM_REVISION,
        "main_tools": [tool.model_dump(mode="json") for tool in MAIN_TOOLS],
        "sub_tools": [tool.model_dump(mode="json") for tool in SUB_TOOLS],
    }
)
WIDESEEK_PARSER_REVISION = canonical_digest(
    {
        "upstream_revision": WIDESEEK_UPSTREAM_REVISION,
        "tool_envelope": "<tool_call>{json}</tool_call>",
        "main_semantics": "no-call-answer|1-4-subtask",
        "sub_semantics": "no-call-summary|1-3-search-access",
        "strict_arguments": True,
    }
)


class _SubtaskArguments(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    subtask: str = Field(min_length=1, max_length=2000)


class _SearchArguments(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    query: str = Field(min_length=1, max_length=400)
    topk: int = Field(default=5, ge=1, le=20)


class _AccessArguments(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    url: str = Field(min_length=1)
    info_to_extract: str = Field(min_length=1, max_length=1000)


class _SubtaskCall(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    name: Literal["subtask"]
    arguments: _SubtaskArguments


class SearchToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    name: Literal["search"]
    arguments: _SearchArguments


class AccessToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    name: Literal["access"]
    arguments: _AccessArguments


SubToolCall = Annotated[SearchToolCall | AccessToolCall, Field(discriminator="name")]
_SUB_TOOL_ADAPTER: TypeAdapter[SubToolCall] = TypeAdapter(SubToolCall)


class MainAnswerTurn(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    kind: Literal["answer"] = "answer"
    answer: str = Field(min_length=1)


class MainSpawnTurn(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    kind: Literal["spawn"] = "spawn"
    subtasks: tuple[str, ...] = Field(min_length=1, max_length=4)


MainTurn = Annotated[MainAnswerTurn | MainSpawnTurn, Field(discriminator="kind")]


class SubSummaryTurn(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    kind: Literal["summary"] = "summary"
    summary: str = Field(min_length=1)


class SubToolsTurn(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    kind: Literal["tools"] = "tools"
    calls: tuple[SubToolCall, ...] = Field(min_length=1, max_length=3)


SubTurn = Annotated[SubSummaryTurn | SubToolsTurn, Field(discriminator="kind")]


def _extract_calls(content: str) -> tuple[object, ...]:
    matches = _TOOL_CALL_RE.findall(content)
    if not matches:
        if _ANY_TOOL_TAG_RE.search(content):
            raise InvalidActionError("malformed tool-call envelope")
        return ()
    remainder = _TOOL_CALL_RE.sub("", content)
    if _ANY_TOOL_TAG_RE.search(remainder):
        raise InvalidActionError("malformed tool-call envelope")
    calls: list[object] = []
    for payload in matches:
        try:
            calls.append(json.loads(payload))
        except json.JSONDecodeError:
            raise InvalidActionError("tool call is not valid JSON") from None
    return tuple(calls)


def parse_main_turn(content: str) -> MainTurn:
    calls = _extract_calls(content)
    if not calls:
        answer = content.strip()
        if not answer:
            raise InvalidActionError("empty Main output is neither ANSWER nor SPAWN")
        return MainAnswerTurn(answer=answer)
    try:
        parsed = tuple(_SubtaskCall.model_validate(call, strict=True) for call in calls)
        return MainSpawnTurn(subtasks=tuple(call.arguments.subtask for call in parsed))
    except ValidationError:
        raise InvalidActionError("Main tool calls must be 1-4 valid subtask calls") from None


def parse_sub_turn(content: str) -> SubTurn:
    calls = _extract_calls(content)
    if not calls:
        summary = content.strip()
        if not summary:
            raise InvalidActionError("empty Sub output is neither evidence nor a tool request")
        return SubSummaryTurn(summary=summary)
    try:
        parsed = tuple(_SUB_TOOL_ADAPTER.validate_python(call, strict=True) for call in calls)
        return SubToolsTurn(calls=parsed)
    except ValidationError:
        raise InvalidActionError("Sub tool calls must be 1-3 valid search/access calls") from None
