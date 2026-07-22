"""Tavily Search API adapter."""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import Awaitable, Callable

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from heterospawn.errors import ConfigurationError, SearchRequestError
from heterospawn.search.base import SearchItem, SearchRequest, SearchResponse

DEFAULT_TAVILY_BASE_URL = "https://api.tavily.com"
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
Sleeper = Callable[[float], Awaitable[None]]


class TavilyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    api_key: SecretStr
    base_url: str = Field(default=DEFAULT_TAVILY_BASE_URL, min_length=1)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=8)

    @classmethod
    def from_environment(cls) -> TavilyConfig:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise ConfigurationError("TAVILY_API_KEY is required for live Tavily calls")
        return cls(api_key=SecretStr(api_key))


class _TavilyItem(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    title: str
    url: str
    content: str
    score: float | None = None


class _TavilyUsage(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    credits: int = Field(ge=0)


class _TavilyResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    query: str
    results: tuple[_TavilyItem, ...]
    request_id: str
    usage: _TavilyUsage | None = None


class TavilySearchService:
    def __init__(
        self,
        config: TavilyConfig,
        *,
        client: httpx.AsyncClient | None = None,
        sleeper: Sleeper = asyncio.sleep,
    ) -> None:
        self._config = config
        self._client = client
        self._sleeper = sleeper

    async def search(self, request: SearchRequest) -> SearchResponse:
        endpoint = f"{self._config.base_url.rstrip('/')}/search"
        headers = {
            "Authorization": f"Bearer {self._config.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": request.query,
            "search_depth": "basic",
            "max_results": request.max_results,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
        if self._client is not None:
            response = await self._request_with_retries(self._client, endpoint, headers, payload)
        else:
            async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
                response = await self._request_with_retries(client, endpoint, headers, payload)

        try:
            parsed = _TavilyResponse.model_validate_json(response.content, strict=True)
        except (ValueError, ValidationError):
            raise SearchRequestError("Tavily returned an invalid response schema") from None

        return SearchResponse(
            request_id=request.request_id,
            provider="tavily",
            provider_revision="search-v1-basic",
            provider_request_id=parsed.request_id,
            query=parsed.query,
            results=tuple(
                SearchItem(
                    title=item.title,
                    url=item.url,
                    content=item.content,
                    score=item.score,
                )
                for item in parsed.results
            ),
            credits=parsed.usage.credits if parsed.usage else None,
            raw_response_digest=hashlib.sha256(response.content).hexdigest(),
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
                    raise SearchRequestError("Tavily request failed after bounded retries") from exc
                await self._sleeper(_retry_delay(attempt))
                continue

            if response.status_code < 400:
                return response
            if (
                response.status_code not in _RETRYABLE_STATUS_CODES
                or attempt == self._config.max_attempts
            ):
                raise SearchRequestError(f"Tavily request failed with HTTP {response.status_code}")
            await self._sleeper(_retry_delay(attempt))

        raise AssertionError("retry loop must return or raise")


def _retry_delay(attempt: int) -> float:
    delay = 0.5 * (2.0 ** (attempt - 1))
    return min(delay, 4.0)
