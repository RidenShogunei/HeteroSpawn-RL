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


class AccessRequest(BaseModel):
    """Fetch one previously discovered document by canonical URL."""

    model_config = ConfigDict(frozen=True, strict=True)

    request_id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    info_to_extract: str = Field(min_length=1, max_length=1000)
    max_characters: int = Field(default=5000, ge=1, le=100000)


class AccessResponse(BaseModel):
    """Immutable page-access result with provider identity."""

    model_config = ConfigDict(frozen=True, strict=True)

    request_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    provider_revision: str = Field(min_length=1)
    provider_request_id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    content: str
    truncated: bool
    raw_response_digest: str = Field(min_length=1)


class ResearchToolService(SearchService, Protocol):
    """Search plus document access used by deep-research environments."""

    async def access(self, request: AccessRequest) -> AccessResponse: ...
