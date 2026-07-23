"""Provider-neutral search services."""

from heterospawn.search.base import (
    AccessRequest,
    AccessResponse,
    ResearchToolService,
    SearchItem,
    SearchRequest,
    SearchResponse,
    SearchService,
)
from heterospawn.search.minimax_mcp import (
    MiniMaxMcpConfig,
    MiniMaxMcpSearchService,
    StdioMiniMaxMcpTransport,
)
from heterospawn.search.mock import MockSearchService
from heterospawn.search.tavily import TavilyConfig, TavilySearchService

__all__ = [
    "AccessRequest",
    "AccessResponse",
    "MiniMaxMcpConfig",
    "MiniMaxMcpSearchService",
    "MockSearchService",
    "ResearchToolService",
    "SearchItem",
    "SearchRequest",
    "SearchResponse",
    "SearchService",
    "StdioMiniMaxMcpTransport",
    "TavilyConfig",
    "TavilySearchService",
]
