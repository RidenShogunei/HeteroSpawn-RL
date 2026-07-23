from __future__ import annotations

import json

import httpx
import pytest

from heterospawn.errors import ConfigurationError, SearchRequestError
from heterospawn.search.base import AccessRequest, SearchRequest
from heterospawn.search.wideseek_local import (
    WIDESEEK_COLLECTION,
    WideSeekLocalConfig,
    WideSeekLocalToolService,
)


def _transport(*, bad_qdrant: bool = False) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/retrieve":
            payload = json.loads(request.content)
            assert payload == {
                "queries": ["red bull"],
                "topk": 2,
                "return_scores": True,
            } or payload == {
                "queries": ["probe"],
                "topk": 1,
                "return_scores": True,
            }
            return httpx.Response(
                200,
                json={
                    "result": [
                        [
                            {
                                "document": {
                                    "title": "Red Bull",
                                    "url": "https://en.wikipedia.org/wiki/Red_Bull",
                                    "contents": "Red Bull is an energy drink.",
                                },
                                "score": 0.97,
                            }
                        ]
                    ]
                },
            )
        if request.url.path == "/access":
            assert json.loads(request.content) == {
                "urls": ["https://en.wikipedia.org/wiki/Red_Bull"]
            }
            return httpx.Response(
                200,
                json={
                    "result": [
                        {
                            "url": "https://en.wikipedia.org/wiki/Red_Bull",
                            "contents": "A" * 300,
                        }
                    ]
                },
            )
        if request.url.path == f"/collections/{WIDESEEK_COLLECTION}":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "result": {
                        "status": "green",
                        "points_count": 21_000_000,
                        "config": {
                            "params": {
                                "vectors": {
                                    "size": 1 if bad_qdrant else 768,
                                    "distance": "Cosine",
                                }
                            },
                            "hnsw_config": {"m": 32, "ef_construct": 512},
                        },
                    },
                },
            )
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_offline_search_and_access_match_upstream_http_shape() -> None:
    async with httpx.AsyncClient(transport=_transport()) as client:
        service = WideSeekLocalToolService(WideSeekLocalConfig(), client=client)
        search = await service.search(
            SearchRequest(request_id="search-1", query="red bull", max_results=2)
        )
        access = await service.access(
            AccessRequest(
                request_id="access-1",
                url=search.results[0].url,
                info_to_extract="ingredients",
                max_characters=128,
            )
        )

    assert search.provider == "wideseek-qdrant-e5"
    assert search.provider_revision == service.provider_revision
    assert search.results[0].score == pytest.approx(0.97)
    assert access.content == "A" * 128
    assert access.truncated is True
    assert access.provider_revision == service.provider_revision


@pytest.mark.asyncio
async def test_environment_probe_checks_qdrant_search_and_access_without_leaking_payloads() -> None:
    async with httpx.AsyncClient(transport=_transport()) as client:
        service = WideSeekLocalToolService(WideSeekLocalConfig(), client=client)
        report = await service.check_environment(probe_query="probe")

    assert report.passed is True
    assert report.points_count == 21_000_000
    assert report.retrieved_items == 1
    assert report.access_nonempty is True
    safe_json = report.model_dump_json()
    assert "probe" not in safe_json
    assert "wikipedia" not in safe_json
    assert "A" * 20 not in safe_json


@pytest.mark.asyncio
async def test_environment_probe_rejects_qdrant_configuration_drift() -> None:
    async with httpx.AsyncClient(transport=_transport(bad_qdrant=True)) as client:
        service = WideSeekLocalToolService(WideSeekLocalConfig(), client=client)
        with pytest.raises(ConfigurationError, match="differs"):
            await service.check_environment(probe_query="probe")


@pytest.mark.asyncio
async def test_offline_service_retries_only_bounded_retryable_failures() -> None:
    calls = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "result": [
                    [
                        {
                            "document": {
                                "url": "memory://one",
                                "contents": "one",
                            },
                            "score": 1.0,
                        }
                    ]
                ]
            },
        )

    async def sleeper(delay: float) -> None:
        delays.append(delay)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = WideSeekLocalToolService(
            WideSeekLocalConfig(max_attempts=3),
            client=client,
            sleeper=sleeper,
        )
        result = await service.search(
            SearchRequest(request_id="retry", query="red bull", max_results=2)
        )

    assert len(result.results) == 1
    assert calls == 3
    assert delays == [0.5, 1.0]


@pytest.mark.asyncio
async def test_offline_service_rejects_invalid_schema() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"result": []}))
    ) as client:
        service = WideSeekLocalToolService(WideSeekLocalConfig(), client=client)
        with pytest.raises(SearchRequestError, match="invalid schema"):
            await service.search(
                SearchRequest(request_id="invalid", query="red bull", max_results=2)
            )
