import json

import httpx
import pytest
from pydantic import SecretStr

from heterospawn.domain.ids import AgentInstanceId, EpisodeId, PolicyId, RolloutId, TaskId
from heterospawn.errors import ProviderRequestError
from heterospawn.policies.base import EvaluationGenerationRequest, Message
from heterospawn.policies.minimax import MiniMaxConfig, MiniMaxEvaluationPolicy


def _request() -> EvaluationGenerationRequest:
    return EvaluationGenerationRequest(
        request_id="request-1",
        task_id=TaskId("task-1"),
        episode_id=EpisodeId("episode-1"),
        rollout_id=RolloutId("rollout-1"),
        agent_role="main",
        agent_instance_id=AgentInstanceId("main-0"),
        messages=(Message(role="user", content="question"),),
    )


@pytest.mark.asyncio
async def test_minimax_adapter_uses_current_endpoint_and_is_not_trainable() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.minimaxi.com/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "MiniMax-M2.7"
        assert payload["max_completion_tokens"] == 4096
        assert payload["reasoning_split"] is True
        return httpx.Response(
            200,
            json={
                "id": "provider-request-1",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "<think>reason</think>\nanswer"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "total_tokens": 13,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        policy = MiniMaxEvaluationPolicy(
            PolicyId("main-policy"),
            MiniMaxConfig(api_key=SecretStr("test-key")),
            client=client,
        )
        result = await policy.generate(_request())

    assert result.content == "answer"
    assert result.reasoning_content == "reason"
    assert result.provider_request_id == "provider-request-1"
    assert result.capabilities.trainable is False
    assert result.capabilities.returns_token_ids is False
    assert result.capabilities.returns_old_log_probs is False


@pytest.mark.asyncio
async def test_minimax_adapter_retries_without_exposing_response_body() -> None:
    attempts = 0
    delays: list[float] = []

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, text="sensitive-provider-body")

    async def record_delay(delay: float) -> None:
        delays.append(delay)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        policy = MiniMaxEvaluationPolicy(
            PolicyId("main-policy"),
            MiniMaxConfig(api_key=SecretStr("test-key"), max_attempts=2),
            client=client,
            sleeper=record_delay,
        )
        with pytest.raises(ProviderRequestError) as raised:
            await policy.generate(_request())

    assert attempts == 2
    assert delays == [0.5]
    assert "sensitive-provider-body" not in str(raised.value)
    assert "test-key" not in str(raised.value)


@pytest.mark.asyncio
async def test_minimax_adapter_retries_http_200_with_invalid_schema() -> None:
    attempts = 0
    delays: list[float] = []

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(200, json={"unexpected": "sensitive-provider-body"})
        return httpx.Response(
            200,
            json={
                "id": "provider-request-2",
                "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    async def record_delay(delay: float) -> None:
        delays.append(delay)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        policy = MiniMaxEvaluationPolicy(
            PolicyId("main-policy"),
            MiniMaxConfig(api_key=SecretStr("test-key"), max_attempts=2),
            client=client,
            sleeper=record_delay,
        )
        result = await policy.generate(_request())

    assert attempts == 2
    assert delays == [0.5]
    assert result.content == "answer"
