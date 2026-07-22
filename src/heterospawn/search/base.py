"""Search contracts kept independent from policy providers."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class SearchRequest(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    request_id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=400)
    max_results: int = Field(default=5, ge=1, le=20)


class SearchItem(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    title: str
    url: str = Field(min_length=1)
    content: str
    score: float | None = None


class SearchResponse(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    request_id: str
    provider: str
    provider_revision: str
    provider_request_id: str
    query: str
    results: tuple[SearchItem, ...]
    credits: int | None = Field(default=None, ge=0)
    raw_response_digest: str


class SearchService(Protocol):
    async def search(self, request: SearchRequest) -> SearchResponse: ...
