import json

import httpx
import pytest
from pydantic import SecretStr

from heterospawn.search.base import SearchRequest
from heterospawn.search.tavily import TavilyConfig, TavilySearchService


@pytest.mark.asyncio
async def test_tavily_adapter_uses_separate_search_contract() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.tavily.com/search"
        assert request.headers["Authorization"] == "Bearer test-search-key"
        payload = json.loads(request.content)
        assert payload == {
            "query": "bounded query",
            "search_depth": "basic",
            "max_results": 3,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
        return httpx.Response(
            200,
            json={
                "query": "bounded query",
                "results": [
                    {
                        "title": "source",
                        "url": "https://example.test/source",
                        "content": "evidence",
                        "score": 0.9,
                    }
                ],
                "request_id": "search-provider-1",
                "usage": {"credits": 1},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = TavilySearchService(
            TavilyConfig(api_key=SecretStr("test-search-key")),
            client=client,
        )
        response = await service.search(
            SearchRequest(request_id="search-1", query="bounded query", max_results=3)
        )

    assert response.provider == "tavily"
    assert response.provider_request_id == "search-provider-1"
    assert response.credits == 1
    assert response.results[0].content == "evidence"
