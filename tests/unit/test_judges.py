from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from heterospawn.domain.ids import TaskId
from heterospawn.errors import JudgeRequestError
from heterospawn.evaluation.judges import (
    XBENCH_JUDGE_PROMPT_REVISION,
    JudgeRequest,
    MiniMaxDevelopmentJudge,
    parse_xbench_judge_response,
)
from heterospawn.policies.minimax import MiniMaxChatClient, MiniMaxConfig


def _request() -> JudgeRequest:
    return JudgeRequest(
        request_id="judge-1",
        task_id=TaskId("task-1"),
        question="sensitive question",
        correct_answer="sensitive reference",
        response="sensitive prediction",
    )


@pytest.mark.asyncio
async def test_minimax_development_judge_uses_pinned_prompt_and_returns_only_digests() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        prompt = payload["messages"][0]["content"]
        assert "sensitive question" in prompt
        assert "sensitive reference" in prompt
        assert "sensitive prediction" in prompt
        return httpx.Response(
            200,
            json={
                "id": "judge-provider-1",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "最终答案:候选答案\n解释:两者一致\n结论:正确"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        judge = MiniMaxDevelopmentJudge(
            MiniMaxChatClient(
                MiniMaxConfig(api_key=SecretStr("test-key")),
                client=client,
            ),
            clock=iter((0.0, 0.25)).__next__,
        )
        result = await judge.judge(_request())

    assert judge.revision.mode == "minimax-development"
    assert judge.revision.comparable_to_official is False
    assert judge.revision.prompt_revision == XBENCH_JUDGE_PROMPT_REVISION
    assert result.correct is True
    assert result.usage.total_tokens == 15
    assert result.latency_ms == 250
    serialized = result.model_dump_json()
    assert "候选答案" not in serialized
    assert "两者一致" not in serialized
    assert "sensitive" not in serialized


def test_xbench_judge_parser_requires_all_upstream_fields() -> None:
    assert XBENCH_JUDGE_PROMPT_REVISION.endswith(
        "e3422231ab04e701dc551d28983572407970ed4ddd83c373cae7bb5a51cd2559"
    )
    assert parse_xbench_judge_response("最终答案:alpha\n解释:matches\n结论: 正确") == (
        "alpha",
        "matches",
        True,
    )
    with pytest.raises(JudgeRequestError, match="required fields"):
        parse_xbench_judge_response("sensitive malformed response")


@pytest.mark.asyncio
async def test_minimax_judge_sanitizes_invalid_provider_verdict() -> None:
    attempts = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            200,
            json={
                "id": "judge-provider-2",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "sensitive malformed verdict"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        judge = MiniMaxDevelopmentJudge(
            MiniMaxChatClient(
                MiniMaxConfig(api_key=SecretStr("test-key")),
                client=client,
            ),
            clock=iter((0.0, 0.5)).__next__,
        )
        with pytest.raises(JudgeRequestError) as raised:
            await judge.judge(_request())

    assert "sensitive malformed verdict" not in str(raised.value)
    assert attempts == 2


@pytest.mark.asyncio
async def test_minimax_judge_repairs_format_once_and_accounts_for_both_calls() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        payload = json.loads(request.content)
        if attempts == 1:
            content = "sensitive malformed verdict"
            assert len(payload["messages"]) == 1
        else:
            content = "最终答案:alpha\n解释:matches\n结论: 正确"
            assert len(payload["messages"]) == 3
        return httpx.Response(
            200,
            json={
                "id": f"judge-provider-{attempts}",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": content},
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        judge = MiniMaxDevelopmentJudge(
            MiniMaxChatClient(
                MiniMaxConfig(api_key=SecretStr("test-key")),
                client=client,
            ),
            clock=iter((0.0, 0.75)).__next__,
        )
        result = await judge.judge(_request())

    assert attempts == 2
    assert result.correct is True
    assert result.usage.prompt_tokens == 6
    assert result.usage.completion_tokens == 4
    assert result.usage.total_tokens == 10
    assert result.latency_ms == 750
    assert "sensitive malformed verdict" not in result.model_dump_json()
