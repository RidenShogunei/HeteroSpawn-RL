"""Serper Search + Jina Reader Access for WideSearch-style online evaluation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Awaitable, Callable

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from heterospawn.errors import ConfigurationError, SearchRequestError
from heterospawn.search.base import (
    AccessRequest,
    AccessResponse,
    SearchItem,
    SearchRequest,
    SearchResponse,
)

DEFAULT_SERPER_URL = "https://google.serper.dev/search"
DEFAULT_SERPER_SCRAPE_URL = "https://scrape.serper.dev"
DEFAULT_JINA_READER_BASE = "https://r.jina.ai"
SERPER_JINA_TOOL_REVISION = "heterospawn-serper-jina-v1"
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
Sleeper = Callable[[float], Awaitable[None]]


class SerperJinaConfig(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    serper_api_key: SecretStr
    jina_api_key: SecretStr
    serper_url: str = Field(default=DEFAULT_SERPER_URL, min_length=1)
    serper_scrape_url: str = Field(default=DEFAULT_SERPER_SCRAPE_URL, min_length=1)
    jina_reader_base: str = Field(default=DEFAULT_JINA_READER_BASE, min_length=1)
    timeout_seconds: float = Field(default=20.0, gt=0)
    jina_timeout_seconds: float = Field(default=8.0, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=8)
    allow_serper_scrape_fallback: bool = True

    @classmethod
    def from_environment(cls) -> SerperJinaConfig:
        serper = os.environ.get("SERPER_API_KEY")
        jina = os.environ.get("JINA_API_KEY")
        if not serper:
            raise ConfigurationError("SERPER_API_KEY is required for Serper+Jina tools")
        if not jina:
            raise ConfigurationError("JINA_API_KEY is required for Serper+Jina tools")
        return cls(serper_api_key=SecretStr(serper), jina_api_key=SecretStr(jina))


class SerperJinaResearchTools:
    """ResearchToolService backed by Serper web search and Jina Reader access.

    When Jina is unreachable from the host, Access falls back to Serper scrape so
    smoke/eval can proceed; reports should note the fallback provider.
    """

    def __init__(
        self,
        config: SerperJinaConfig | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        sleeper: Sleeper = asyncio.sleep,
    ) -> None:
        self._config = config or SerperJinaConfig.from_environment()
        self._client = client
        self._sleeper = sleeper
        self.provider_revision = SERPER_JINA_TOOL_REVISION

    async def search(self, request: SearchRequest) -> SearchResponse:
        headers = {
            "X-API-KEY": self._config.serper_api_key.get_secret_value(),
            "Content-Type": "application/json",
        }
        payload = {"q": request.query, "num": request.max_results}
        response = await self._request(
            "POST",
            self._config.serper_url,
            headers=headers,
            json=payload,
            timeout=self._config.timeout_seconds,
        )
        try:
            body = response.json()
        except ValueError as exc:
            raise SearchRequestError("Serper returned non-JSON response") from exc
        organic = body.get("organic") if isinstance(body, dict) else None
        if not isinstance(organic, list):
            organic = []
        results: list[SearchItem] = []
        for item in organic[: request.max_results]:
            if not isinstance(item, dict):
                continue
            url = item.get("link") or item.get("url")
            if not isinstance(url, str) or not url.strip():
                continue
            title = item.get("title") if isinstance(item.get("title"), str) else ""
            snippet = item.get("snippet") if isinstance(item.get("snippet"), str) else ""
            results.append(SearchItem(title=title, url=url.strip(), content=snippet, score=None))
        digest = hashlib.sha256(response.content).hexdigest()
        return SearchResponse(
            request_id=request.request_id,
            provider="serper",
            provider_revision=self.provider_revision,
            provider_request_id=digest[:16],
            query=request.query,
            results=tuple(results),
            credits=None,
            raw_response_digest=digest,
        )

    async def access(self, request: AccessRequest) -> AccessResponse:
        try:
            return await self._access_jina(request)
        except SearchRequestError:
            if not self._config.allow_serper_scrape_fallback:
                raise
            return await self._access_serper_scrape(request)

    async def _access_jina(self, request: AccessRequest) -> AccessResponse:
        reader_url = f"{self._config.jina_reader_base.rstrip('/')}/{request.url}"
        headers = {
            "Authorization": f"Bearer {self._config.jina_api_key.get_secret_value()}",
            "Accept": "text/plain",
            "X-Return-Format": "markdown",
        }
        response = await self._request(
            "GET",
            reader_url,
            headers=headers,
            timeout=self._config.jina_timeout_seconds,
        )
        return self._access_response(
            request, provider="jina-reader", status=response.status_code, text=response.text
        )

    async def _access_serper_scrape(self, request: AccessRequest) -> AccessResponse:
        headers = {
            "X-API-KEY": self._config.serper_api_key.get_secret_value(),
            "Content-Type": "application/json",
        }
        response = await self._request(
            "POST",
            self._config.serper_scrape_url,
            headers=headers,
            json={"url": request.url},
            timeout=self._config.timeout_seconds,
        )
        try:
            body = response.json()
        except ValueError as exc:
            raise SearchRequestError("Serper scrape returned non-JSON response") from exc
        if isinstance(body, dict):
            text = body.get("text") or body.get("markdown") or json.dumps(body, ensure_ascii=False)
        else:
            text = str(body)
        if not isinstance(text, str):
            text = str(text)
        return self._access_response(
            request,
            provider="serper-scrape-fallback",
            status=response.status_code,
            text=text,
        )

    def _access_response(
        self, request: AccessRequest, *, provider: str, status: int, text: str
    ) -> AccessResponse:
        digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        truncated = len(text) > request.max_characters
        return AccessResponse(
            request_id=request.request_id,
            provider=provider,
            provider_revision=self.provider_revision,
            provider_request_id=f"{provider}:{status}:{digest[:16]}",
            url=request.url,
            content=text[: request.max_characters],
            truncated=truncated,
            raw_response_digest=digest,
        )

    async def _request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object] | None = None,
        timeout: float,
    ) -> httpx.Response:
        if self._client is not None:
            return await self._request_with_retries(
                self._client, method, url, headers=headers, json=json
            )
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            return await self._request_with_retries(
                client, method, url, headers=headers, json=json
            )

    async def _request_with_retries(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object] | None = None,
    ) -> httpx.Response:
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                response = await client.request(method, url, headers=headers, json=json)
            except httpx.RequestError as exc:
                if attempt == self._config.max_attempts:
                    raise SearchRequestError(
                        f"Serper/Jina request failed after retries: {type(exc).__name__}"
                    ) from exc
                await self._sleeper(_retry_delay(attempt))
                continue
            if response.status_code < 400:
                return response
            if (
                response.status_code not in _RETRYABLE_STATUS_CODES
                or attempt == self._config.max_attempts
            ):
                raise SearchRequestError(
                    f"Serper/Jina request failed with HTTP {response.status_code}"
                )
            await self._sleeper(_retry_delay(attempt))
        raise AssertionError("retry loop must return or raise")


def _retry_delay(attempt: int) -> float:
    return min(0.5 * (2.0 ** (attempt - 1)), 4.0)
