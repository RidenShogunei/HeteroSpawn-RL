"""MiniMax OpenAI-compatible evaluation policy adapter."""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import Awaitable, Callable

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from heterospawn.domain.ids import PolicyId
from heterospawn.errors import ConfigurationError, ProviderRequestError
from heterospawn.policies.base import (
    EvaluationGenerationRequest,
    EvaluationGenerationResult,
    ExternalModelRevision,
    PolicyCapabilities,
    TokenUsage,
)

DEFAULT_MINIMAX_BASE_URL = "https://api.minimaxi.com/v1"
DEFAULT_MINIMAX_MODEL = "MiniMax-M2.7"
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})

Sleeper = Callable[[float], Awaitable[None]]


class MiniMaxConfig(BaseModel):
    """Secret-safe MiniMax connection and retry configuration."""

    model_config = ConfigDict(frozen=True, strict=True)

    api_key: SecretStr
    base_url: str = Field(default=DEFAULT_MINIMAX_BASE_URL, min_length=1)
    model: str = Field(default=DEFAULT_MINIMAX_MODEL, min_length=1)
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=8)

    @classmethod
    def from_environment(cls) -> MiniMaxConfig:
        api_key = os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            raise ConfigurationError("MINIMAX_API_KEY is required for live MiniMax calls")
        return cls(
            api_key=SecretStr(api_key),
            base_url=os.environ.get("MINIMAX_BASE_URL", DEFAULT_MINIMAX_BASE_URL),
            model=os.environ.get("MINIMAX_MODEL", DEFAULT_MINIMAX_MODEL),
        )


class _ResponseUsage(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class _ResponseMessage(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    content: str
    reasoning_content: str | None = None
    reasoning: str | None = None


class _ResponseChoice(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    finish_reason: str
    message: _ResponseMessage


class _ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    id: str
    choices: tuple[_ResponseChoice, ...] = Field(min_length=1)
    usage: _ResponseUsage


class MiniMaxEvaluationPolicy:
    """Calls MiniMax for text evaluation without claiming trainable provenance."""

    def __init__(
        self,
        policy_id: PolicyId,
        config: MiniMaxConfig,
        *,
        client: httpx.AsyncClient | None = None,
        sleeper: Sleeper = asyncio.sleep,
    ) -> None:
        self._policy_id = policy_id
        self._config = config
        self._client = client
        self._sleeper = sleeper
        self._revision = ExternalModelRevision(
            provider="minimax",
            model=config.model,
            api_base=config.base_url.rstrip("/"),
        )

    @property
    def policy_id(self) -> PolicyId:
        return self._policy_id

    @property
    def revision(self) -> ExternalModelRevision:
        return self._revision

    async def generate(self, request: EvaluationGenerationRequest) -> EvaluationGenerationResult:
        sampling_params = dict(request.sampling_params)
        sampling_params.setdefault("temperature", 1.0)
        sampling_params.setdefault("top_p", 0.95)
        sampling_params.setdefault("max_completion_tokens", 4096)
        payload: dict[str, object] = {
            "model": self._config.model,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            **sampling_params,
        }
        headers = {
            "Authorization": f"Bearer {self._config.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        endpoint = f"{self._config.base_url.rstrip('/')}/chat/completions"

        if self._client is not None:
            response = await self._request_with_retries(self._client, endpoint, headers, payload)
        else:
            async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
                response = await self._request_with_retries(client, endpoint, headers, payload)

        try:
            parsed = _ChatCompletionResponse.model_validate_json(response.content, strict=True)
        except (ValueError, ValidationError) as exc:
            raise ProviderRequestError("MiniMax returned an invalid response schema") from exc

        choice = parsed.choices[0]
        return EvaluationGenerationResult(
            request_id=request.request_id,
            policy_id=self.policy_id,
            revision=self.revision,
            provider_request_id=parsed.id,
            content=choice.message.content,
            reasoning_content=choice.message.reasoning_content or choice.message.reasoning,
            finish_reason=choice.finish_reason,
            usage=TokenUsage(
                prompt_tokens=parsed.usage.prompt_tokens,
                completion_tokens=parsed.usage.completion_tokens,
                total_tokens=parsed.usage.total_tokens,
            ),
            raw_response_digest=hashlib.sha256(response.content).hexdigest(),
            capabilities=PolicyCapabilities(
                trainable=False,
                returns_token_ids=False,
                returns_old_log_probs=False,
            ),
        )

    async def _request_with_retries(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        headers: dict[str, str],
        payload: dict[str, object],
    ) -> httpx.Response:
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                response = await client.post(endpoint, headers=headers, json=payload)
            except httpx.RequestError as exc:
                if attempt == self._config.max_attempts:
                    raise ProviderRequestError(
                        "MiniMax request failed after bounded retries"
                    ) from exc
                await self._sleeper(_retry_delay(attempt))
                continue

            if response.status_code < 400:
                return response
            if (
                response.status_code not in _RETRYABLE_STATUS_CODES
                or attempt == self._config.max_attempts
            ):
                raise ProviderRequestError(
                    f"MiniMax request failed with HTTP {response.status_code}"
                )
            await self._sleeper(_retry_delay(attempt))

        raise AssertionError("retry loop must return or raise")


def _retry_delay(attempt: int) -> float:
    delay = 0.5 * (2.0 ** (attempt - 1))
    return min(delay, 4.0)
