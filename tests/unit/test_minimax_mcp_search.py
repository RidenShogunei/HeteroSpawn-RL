from __future__ import annotations

import json

import pytest
from mcp import types
from pydantic import SecretStr

from heterospawn.errors import SearchRequestError
from heterospawn.search.base import SearchRequest
from heterospawn.search.minimax_mcp import (
    MiniMaxMcpConfig,
    MiniMaxMcpSearchService,
    _build_mcp_environment,
    _extract_tool_text,
)


class FakeTransport:
    def __init__(self, response: str) -> None:
        self.response = response
        self.queries: list[str] = []

    async def web_search(self, query: str) -> str:
        self.queries.append(query)
        return self.response


def _payload(*, status_code: int = 0) -> str:
    return json.dumps(
        {
            "organic": [
                {
                    "title": f"result-{index}",
                    "link": f"https://example.test/{index}",
                    "snippet": f"evidence-{index}",
                }
                for index in range(3)
            ],
            "related_searches": [],
            "base_resp": {"status_code": status_code, "status_msg": ""},
        }
    )


@pytest.mark.asyncio
async def test_minimax_mcp_search_maps_and_limits_results() -> None:
    transport = FakeTransport(_payload())
    service = MiniMaxMcpSearchService(
        MiniMaxMcpConfig(api_key=SecretStr("test-key")),
        transport=transport,
    )

    response = await service.search(
        SearchRequest(request_id="search-1", query="short query", max_results=2)
    )

    assert transport.queries == ["short query"]
    assert response.provider == "minimax-mcp-search"
    assert response.provider_revision == "minimax-coding-plan-mcp@0.0.4"
    assert len(response.results) == 2
    assert response.results[0].url == "https://example.test/0"


@pytest.mark.asyncio
async def test_minimax_mcp_search_rejects_invalid_payload_and_provider_error() -> None:
    config = MiniMaxMcpConfig(api_key=SecretStr("test-key"))
    invalid_service = MiniMaxMcpSearchService(config, transport=FakeTransport("not-json"))
    failed_service = MiniMaxMcpSearchService(
        config,
        transport=FakeTransport(_payload(status_code=1001)),
    )
    request = SearchRequest(request_id="search-1", query="short query")

    with pytest.raises(SearchRequestError, match="invalid search schema"):
        await invalid_service.search(request)
    with pytest.raises(SearchRequestError, match="provider status 1001"):
        await failed_service.search(request)


def test_mcp_tool_contract_rejects_missing_and_error_results_without_body() -> None:
    success = types.CallToolResult(content=[types.TextContent(type="text", text="{}")])
    error = types.CallToolResult(
        content=[types.TextContent(type="text", text="sensitive-provider-body")],
        isError=True,
    )

    with pytest.raises(SearchRequestError, match="did not advertise"):
        _extract_tool_text(("understand_image",), success)
    with pytest.raises(SearchRequestError) as raised:
        _extract_tool_text(("web_search",), error)
    assert "sensitive-provider-body" not in str(raised.value)


def test_mcp_subprocess_environment_does_not_inherit_unrelated_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UNRELATED_API_KEY", "must-not-be-inherited")
    monkeypatch.setenv("PATH", "test-path")

    environment = _build_mcp_environment("minimax-test-key", "https://example.test")

    assert environment["MINIMAX_API_KEY"] == "minimax-test-key"
    assert environment["MINIMAX_API_HOST"] == "https://example.test"
    assert environment["PATH"] == "test-path"
    assert "UNRELATED_API_KEY" not in environment
