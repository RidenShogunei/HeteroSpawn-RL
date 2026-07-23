"""Pinned HTTP adapter for the WideSeek Qdrant/E5 offline tool service."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from heterospawn.domain.training import canonical_digest
from heterospawn.errors import ConfigurationError, SearchRequestError
from heterospawn.search.base import (
    AccessRequest,
    AccessResponse,
    SearchItem,
    SearchRequest,
    SearchResponse,
)

WIDESEEK_UPSTREAM_REVISION: Literal["d9f3d8a9db4d7aad1d641029293295503dd3eb2c"] = (
    "d9f3d8a9db4d7aad1d641029293295503dd3eb2c"
)
WIDESEEK_CORPUS_REVISION: Literal["178d7d037f661be3159b0c3a8a4119b974f01880"] = (
    "178d7d037f661be3159b0c3a8a4119b974f01880"
)
WIDESEEK_CORPUS_MANIFEST_DIGEST: Literal[
    "7f22b16e05f90d2fd7d0ff724effc1eb7cc543d5207526125e26d458cb8a4aa5"
] = "7f22b16e05f90d2fd7d0ff724effc1eb7cc543d5207526125e26d458cb8a4aa5"
WIDESEEK_E5_REVISION: Literal["f52bf8ec8c7124536f0efb74aca902b2995e5bcd"] = (
    "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
)
WIDESEEK_E5_MANIFEST_DIGEST: Literal[
    "5877db5cb6f70f910ee862f852515faedb391a15f4558ddb2ed3f2b86f8c88be"
] = "5877db5cb6f70f910ee862f852515faedb391a15f4558ddb2ed3f2b86f8c88be"
WIDESEEK_COLLECTION: Literal["wiki_collection_m32_cef512"] = "wiki_collection_m32_cef512"
WIDESEEK_OFFLINE_PROTOCOL_REVISION: Literal["heterospawn-wideseek-offline-http-v1"] = (
    "heterospawn-wideseek-offline-http-v1"
)
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
Sleeper = Callable[[float], Awaitable[None]]


class WideSeekEnvironmentIdentity(BaseModel):
    """Complete immutable identity for the official-shaped offline environment."""

    model_config = ConfigDict(frozen=True, strict=True)

    upstream_revision: Literal["d9f3d8a9db4d7aad1d641029293295503dd3eb2c"] = (
        WIDESEEK_UPSTREAM_REVISION
    )
    corpus_repo: Literal["RLinf/Wiki-2018-Corpus"] = "RLinf/Wiki-2018-Corpus"
    corpus_revision: Literal["178d7d037f661be3159b0c3a8a4119b974f01880"] = WIDESEEK_CORPUS_REVISION
    corpus_manifest_digest: Literal[
        "7f22b16e05f90d2fd7d0ff724effc1eb7cc543d5207526125e26d458cb8a4aa5"
    ] = WIDESEEK_CORPUS_MANIFEST_DIGEST
    retriever_repo: Literal["intfloat/e5-base-v2"] = "intfloat/e5-base-v2"
    retriever_revision: Literal["f52bf8ec8c7124536f0efb74aca902b2995e5bcd"] = WIDESEEK_E5_REVISION
    retriever_manifest_digest: Literal[
        "5877db5cb6f70f910ee862f852515faedb391a15f4558ddb2ed3f2b86f8c88be"
    ] = WIDESEEK_E5_MANIFEST_DIGEST
    collection_name: Literal["wiki_collection_m32_cef512"] = WIDESEEK_COLLECTION
    vector_size: Literal[768] = 768
    distance: Literal["Cosine"] = "Cosine"
    hnsw_m: Literal[32] = 32
    hnsw_ef_construct: Literal[512] = 512
    search_hnsw_ef: Literal[256] = 256
    protocol_revision: Literal["heterospawn-wideseek-offline-http-v1"] = (
        WIDESEEK_OFFLINE_PROTOCOL_REVISION
    )

    @property
    def digest(self) -> str:
        return canonical_digest(self)


class WideSeekLocalConfig(BaseModel):
    """Connection policy; URLs cannot contain credentials or fragments."""

    model_config = ConfigDict(frozen=True, strict=True)

    service_url: str = Field(default="http://127.0.0.1:8000", min_length=1)
    qdrant_url: str = Field(default="http://127.0.0.1:6333", min_length=1)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=10)
    identity: WideSeekEnvironmentIdentity = Field(default_factory=WideSeekEnvironmentIdentity)

    @model_validator(mode="after")
    def urls_must_be_safe_http_endpoints(self) -> WideSeekLocalConfig:
        for label, value in (
            ("service_url", self.service_url),
            ("qdrant_url", self.qdrant_url),
        ):
            parsed = urlsplit(value)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(f"{label} must be a credential-free HTTP endpoint")
        return self


class _Document(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    url: str = Field(min_length=1)
    contents: str
    title: str | None = None


class _ScoredDocument(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    document: _Document
    score: float


class _RetrieveResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    result: tuple[tuple[_ScoredDocument, ...], ...] = Field(min_length=1)


class _AccessResponsePayload(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    result: tuple[_Document | None, ...] = Field(min_length=1)


class _QdrantVectors(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    size: int
    distance: str


class _QdrantParams(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    vectors: _QdrantVectors


class _QdrantHnsw(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    m: int
    ef_construct: int


class _QdrantConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    params: _QdrantParams
    hnsw_config: _QdrantHnsw


class _QdrantCollection(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    status: str
    points_count: int = Field(ge=0)
    config: _QdrantConfig


class _QdrantResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    result: _QdrantCollection
    status: str


class WideSeekEnvironmentReport(BaseModel):
    """Safe readiness evidence; queries, URLs, pages, and host addresses are omitted."""

    model_config = ConfigDict(frozen=True, strict=True)

    environment_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    retriever_manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    collection_name: str
    qdrant_status: str
    points_count: int = Field(ge=0)
    vector_size: int = Field(gt=0)
    retrieved_items: int = Field(ge=1)
    access_nonempty: bool
    passed: Literal[True] = True


class WideSeekLocalToolService:
    """Exact adapter for upstream WideSeek `/retrieve` and `/access` shapes."""

    def __init__(
        self,
        config: WideSeekLocalConfig,
        *,
        client: httpx.AsyncClient | None = None,
        sleeper: Sleeper = asyncio.sleep,
    ) -> None:
        self._config = config
        self._client = client
        self._sleeper = sleeper

    @property
    def provider_revision(self) -> str:
        return self._config.identity.digest

    @property
    def identity(self) -> WideSeekEnvironmentIdentity:
        return self._config.identity

    async def search(self, request: SearchRequest) -> SearchResponse:
        content = await self._post(
            "/retrieve",
            {
                "queries": [request.query],
                "topk": request.max_results,
                "return_scores": True,
            },
        )
        try:
            parsed = _RetrieveResponse.model_validate_json(content, strict=True)
        except (ValueError, ValidationError):
            raise SearchRequestError("WideSeek retrieve response has invalid schema") from None
        if len(parsed.result) != 1:
            raise SearchRequestError("WideSeek retrieve response does not align with query batch")
        digest = hashlib.sha256(content).hexdigest()
        results = parsed.result[0]
        return SearchResponse(
            request_id=request.request_id,
            provider="wideseek-qdrant-e5",
            provider_revision=self.provider_revision,
            provider_request_id=f"offline:{digest[:24]}",
            query=request.query,
            results=tuple(
                SearchItem(
                    title=item.document.title or item.document.url,
                    url=item.document.url,
                    content=item.document.contents,
                    score=item.score,
                )
                for item in results
            ),
            credits=0,
            raw_response_digest=digest,
        )

    async def access(self, request: AccessRequest) -> AccessResponse:
        content = await self._post("/access", {"urls": [request.url]})
        try:
            parsed = _AccessResponsePayload.model_validate_json(content, strict=True)
        except (ValueError, ValidationError):
            raise SearchRequestError("WideSeek access response has invalid schema") from None
        if len(parsed.result) != 1:
            raise SearchRequestError("WideSeek access response does not align with URL batch")
        document = parsed.result[0]
        if document is None or document.url != request.url:
            raise SearchRequestError("WideSeek access did not return the requested URL")
        full_content = document.contents
        output = full_content[: request.max_characters]
        digest = hashlib.sha256(content).hexdigest()
        return AccessResponse(
            request_id=request.request_id,
            provider="wideseek-qdrant-e5",
            provider_revision=self.provider_revision,
            provider_request_id=f"offline:{digest[:24]}",
            url=request.url,
            content=output,
            truncated=len(output) != len(full_content),
            raw_response_digest=digest,
        )

    async def check_environment(self, *, probe_query: str) -> WideSeekEnvironmentReport:
        qdrant_content = await self._get_qdrant_collection()
        try:
            qdrant = _QdrantResponse.model_validate_json(qdrant_content, strict=True)
        except (ValueError, ValidationError):
            raise ConfigurationError("Qdrant collection response has invalid schema") from None
        identity = self._config.identity
        collection = qdrant.result
        if qdrant.status.lower() != "ok" or collection.status.lower() != "green":
            raise ConfigurationError("Qdrant collection is not ready")
        vectors = collection.config.params.vectors
        hnsw = collection.config.hnsw_config
        if (
            vectors.size != identity.vector_size
            or vectors.distance.casefold() != identity.distance.casefold()
            or hnsw.m != identity.hnsw_m
            or hnsw.ef_construct != identity.hnsw_ef_construct
        ):
            raise ConfigurationError("Qdrant collection configuration differs from pinned identity")

        search = await self.search(
            SearchRequest(request_id="environment-probe-search", query=probe_query, max_results=1)
        )
        if not search.results:
            raise ConfigurationError("WideSeek environment probe returned no search results")
        access = await self.access(
            AccessRequest(
                request_id="environment-probe-access",
                url=search.results[0].url,
                info_to_extract="environment readiness probe",
                max_characters=256,
            )
        )
        return WideSeekEnvironmentReport(
            environment_revision=identity.digest,
            corpus_manifest_digest=identity.corpus_manifest_digest,
            retriever_manifest_digest=identity.retriever_manifest_digest,
            collection_name=identity.collection_name,
            qdrant_status=collection.status,
            points_count=collection.points_count,
            vector_size=vectors.size,
            retrieved_items=len(search.results),
            access_nonempty=bool(access.content),
        )

    async def _post(self, route: str, payload: dict[str, object]) -> bytes:
        endpoint = f"{self._config.service_url.rstrip('/')}{route}"
        return await self._request("POST", endpoint, payload)

    async def _get_qdrant_collection(self) -> bytes:
        endpoint = (
            f"{self._config.qdrant_url.rstrip('/')}/collections/"
            f"{self._config.identity.collection_name}"
        )
        return await self._request("GET", endpoint, None)

    async def _request(
        self,
        method: Literal["GET", "POST"],
        endpoint: str,
        payload: dict[str, object] | None,
    ) -> bytes:
        if self._client is not None:
            return await self._request_with_retries(
                self._client,
                method,
                endpoint,
                payload,
            )
        async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
            return await self._request_with_retries(client, method, endpoint, payload)

    async def _request_with_retries(
        self,
        client: httpx.AsyncClient,
        method: Literal["GET", "POST"],
        endpoint: str,
        payload: dict[str, object] | None,
    ) -> bytes:
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                response = await client.request(method, endpoint, json=payload)
            except httpx.RequestError as exc:
                if attempt == self._config.max_attempts:
                    raise SearchRequestError(
                        "WideSeek offline service failed after bounded retries"
                    ) from exc
                await self._sleeper(_retry_delay(attempt))
                continue
            if response.status_code < 400:
                return response.content
            if (
                response.status_code not in _RETRYABLE_STATUS_CODES
                or attempt == self._config.max_attempts
            ):
                raise SearchRequestError(
                    f"WideSeek offline service failed with HTTP {response.status_code}"
                )
            await self._sleeper(_retry_delay(attempt))
        raise AssertionError("retry loop must return or raise")


def _retry_delay(attempt: int) -> float:
    return min(0.5 * (2.0 ** (attempt - 1)), 4.0)
