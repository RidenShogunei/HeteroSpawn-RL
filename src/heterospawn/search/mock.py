"""Deterministic search service for orchestration tests."""

from __future__ import annotations

import hashlib

from heterospawn.search.base import SearchItem, SearchRequest, SearchResponse

MOCK_SEARCH_REVISION = "deterministic-v1"


class MockSearchService:
    def __init__(self, content_by_query: dict[str, str] | None = None) -> None:
        self._content_by_query = dict(content_by_query or {})

    async def search(self, request: SearchRequest) -> SearchResponse:
        content = self._content_by_query.get(request.query, f"evidence for {request.query}")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return SearchResponse(
            request_id=request.request_id,
            provider="mock-search",
            provider_revision=MOCK_SEARCH_REVISION,
            provider_request_id=f"mock:{request.request_id}",
            query=request.query,
            results=(
                SearchItem(
                    title="mock result",
                    url="memory://mock-search",
                    content=content,
                    score=1.0,
                ),
            ),
            credits=0,
            raw_response_digest=digest,
        )
