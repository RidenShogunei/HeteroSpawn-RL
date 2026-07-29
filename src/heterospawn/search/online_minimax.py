"""Online MiniMax search + HTTP page access for WideSearch-style evaluation."""

from __future__ import annotations

import hashlib
import re
from html import unescape
from urllib.parse import urlsplit

import httpx

from heterospawn.errors import SearchRequestError
from heterospawn.search.base import (
    AccessRequest,
    AccessResponse,
    SearchRequest,
    SearchResponse,
)
from heterospawn.search.minimax_mcp import MiniMaxMcpConfig, MiniMaxMcpSearchService

ONLINE_TOOL_REVISION = "heterospawn-minimax-online-http-v1"
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


class OnlineMiniMaxResearchTools:
    """ResearchToolService backed by MiniMax web_search and direct HTTP Access."""

    def __init__(
        self,
        search: MiniMaxMcpSearchService | None = None,
        *,
        timeout_seconds: float = 30.0,
        user_agent: str = "HeteroSpawn-RL-WideSearchEval/0.1",
    ) -> None:
        self._search = search or MiniMaxMcpSearchService(MiniMaxMcpConfig.from_environment())
        self._timeout = timeout_seconds
        self._user_agent = user_agent
        self.provider_revision = ONLINE_TOOL_REVISION

    async def search(self, request: SearchRequest) -> SearchResponse:
        response = await self._search.search(request)
        return SearchResponse(
            request_id=response.request_id,
            provider="minimax-online-search",
            provider_revision=self.provider_revision,
            provider_request_id=response.provider_request_id,
            query=response.query,
            results=response.results,
            credits=response.credits,
            raw_response_digest=response.raw_response_digest,
        )

    async def access(self, request: AccessRequest) -> AccessResponse:
        parsed = urlsplit(request.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise SearchRequestError("Access URL must be an absolute http(s) URL")
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": self._user_agent},
            ) as client:
                response = await client.get(request.url)
        except httpx.HTTPError as exc:
            raise SearchRequestError(f"Access request failed: {type(exc).__name__}") from exc
        raw = response.text
        digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()
        text = _html_to_text(raw)
        truncated = len(text) > request.max_characters
        content = text[: request.max_characters]
        return AccessResponse(
            request_id=request.request_id,
            provider="httpx-page-access",
            provider_revision=self.provider_revision,
            provider_request_id=f"http:{response.status_code}:{digest[:16]}",
            url=request.url,
            content=content,
            truncated=truncated,
            raw_response_digest=digest,
        )


def _html_to_text(raw: str) -> str:
    without_scripts = re.sub(
        r"(?is)<(script|style|noscript).*?>.*?</\1>",
        " ",
        raw,
    )
    text = _TAG_RE.sub(" ", without_scripts)
    text = unescape(text)
    return _WS_RE.sub(" ", text).strip()
