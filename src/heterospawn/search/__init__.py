"""Provider-neutral search services."""

from heterospawn.search.base import SearchItem, SearchRequest, SearchResponse, SearchService
from heterospawn.search.minimax_mcp import (
    MiniMaxMcpConfig,
    MiniMaxMcpSearchService,
    StdioMiniMaxMcpTransport,
)
from heterospawn.search.mock import MockSearchService
from heterospawn.search.tavily import TavilyConfig, TavilySearchService

__all__ = [
    "MiniMaxMcpConfig",
    "MiniMaxMcpSearchService",
    "MockSearchService",
    "SearchItem",
    "SearchRequest",
    "SearchResponse",
    "SearchService",
    "StdioMiniMaxMcpTransport",
    "TavilyConfig",
    "TavilySearchService",
]
