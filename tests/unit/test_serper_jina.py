"""Unit tests for Serper + Jina research tools."""

from __future__ import annotations

import httpx
import pytest
from pydantic import SecretStr

from heterospawn.search.base import AccessRequest, SearchRequest
from heterospawn.search.serper_jina import SerperJinaConfig, SerperJinaResearchTools


@pytest.mark.asyncio
async def test_serper_search_maps_organic_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "google.serper.dev"
        return httpx.Response(
            200,
            json={
                "organic": [
                    {
                        "title": "Example",
                        "link": "https://example.com/a",
                        "snippet": "hello",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        tools = SerperJinaResearchTools(
            SerperJinaConfig(
                serper_api_key=SecretStr("serper-test"),
                jina_api_key=SecretStr("jina-test"),
            ),
            client=client,
        )
        response = await tools.search(
            SearchRequest(request_id="s1", query="wide search", max_results=3)
        )
    assert response.provider == "serper"
    assert len(response.results) == 1
    assert response.results[0].url == "https://example.com/a"


@pytest.mark.asyncio
async def test_jina_access_truncates_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "r.jina.ai" in str(request.url)
        return httpx.Response(200, text="x" * 50)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        tools = SerperJinaResearchTools(
            SerperJinaConfig(
                serper_api_key=SecretStr("serper-test"),
                jina_api_key=SecretStr("jina-test"),
            ),
            client=client,
        )
        response = await tools.access(
            AccessRequest(
                request_id="a1",
                url="https://example.com/page",
                info_to_extract="title",
                max_characters=10,
            )
        )
    assert response.provider == "jina-reader"
    assert response.content == "x" * 10
    assert response.truncated is True
