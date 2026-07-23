from __future__ import annotations

import json

import pytest

from heterospawn.domain.ids import TaskId
from heterospawn.errors import JudgeRequestError
from heterospawn.evaluation.semantic_judge import (
    MiniMaxSemanticJudge,
    SemanticJudgeCache,
    SemanticJudgeRequest,
)
from heterospawn.policies.base import ExternalModelRevision, TokenUsage
from heterospawn.policies.minimax import (
    MiniMaxChatRequest,
    MiniMaxChatResult,
)


class _Chat:
    revision = ExternalModelRevision(
        provider="fake-minimax",
        model="judge",
        api_base="https://example.invalid/v1",
    )

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.requests: list[MiniMaxChatRequest] = []

    async def complete(self, request: MiniMaxChatRequest) -> MiniMaxChatResult:
        self.requests.append(request)
        content = self._responses.pop(0)
        return MiniMaxChatResult(
            provider_request_id=f"request-{len(self.requests)}",
            content=content,
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
            raw_response_digest=f"digest-{len(self.requests)}",
        )


def _request(request_id: str) -> SemanticJudgeRequest:
    return SemanticJudgeRequest(
        request_id=request_id,
        task_id=TaskId("task"),
        operation="cell_equivalence",
        question="question",
        candidates=("one", "two"),
        references=("1", "2"),
    )


@pytest.mark.asyncio
async def test_minimax_semantic_judge_is_temperature_zero_and_exact_revision_cached() -> None:
    chat = _Chat([json.dumps({"scores": [1, 0]})])
    judge = MiniMaxSemanticJudge(chat, cache=SemanticJudgeCache())  # type: ignore[arg-type]

    first = await judge.judge(_request("first"))
    second = await judge.judge(_request("second"))

    assert first.scores == (1, 0)
    assert first.cache_hit is False
    assert second.scores == first.scores
    assert second.cache_hit is True
    assert second.cache_key == first.cache_key
    assert len(chat.requests) == 1
    assert dict(chat.requests[0].sampling_params)["temperature"] == 0.0
    cached_payload = second.model_dump_json()
    assert "question" not in cached_payload
    assert "one" not in cached_payload


@pytest.mark.asyncio
async def test_invalid_judge_schema_repairs_once_then_fails_phase() -> None:
    chat = _Chat(["not-json", '{"scores":[1]}'])
    judge = MiniMaxSemanticJudge(  # type: ignore[arg-type]
        chat,
        max_format_attempts=2,
    )
    with pytest.raises(JudgeRequestError, match="invalid output"):
        await judge.judge(_request("bad"))
    assert len(chat.requests) == 2


def test_cache_rejects_conflicting_digest_identity() -> None:
    cache = SemanticJudgeCache()
    cache.put("a" * 64, (1,), "digest-1")
    with pytest.raises(JudgeRequestError, match="conflicting"):
        cache.put("a" * 64, (0,), "digest-2")
