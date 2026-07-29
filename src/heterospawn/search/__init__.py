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
from heterospawn.search.browsecomp_plus_local import BrowseCompPlusLocalTools
from heterospawn.search.minimax_mcp import (
    MiniMaxMcpConfig,
    MiniMaxMcpSearchService,
    StdioMiniMaxMcpTransport,
)
from heterospawn.search.mock import MockSearchService
from heterospawn.search.serper_jina import SerperJinaConfig, SerperJinaResearchTools
from heterospawn.search.tavily import TavilyConfig, TavilySearchService
from heterospawn.search.wideseek_local import (
    WideSeekEnvironmentIdentity,
    WideSeekEnvironmentReport,
    WideSeekLocalConfig,
    WideSeekLocalToolService,
)

__all__ = [
    "AccessRequest",
    "AccessResponse",
    "BrowseCompPlusLocalTools",
    "MiniMaxMcpConfig",
    "MiniMaxMcpSearchService",
    "MockSearchService",
    "ResearchToolService",
    "SearchItem",
    "SearchRequest",
    "SearchResponse",
    "SearchService",
    "SerperJinaConfig",
    "SerperJinaResearchTools",
    "StdioMiniMaxMcpTransport",
    "TavilyConfig",
    "TavilySearchService",
    "WideSeekEnvironmentIdentity",
    "WideSeekEnvironmentReport",
    "WideSeekLocalConfig",
    "WideSeekLocalToolService",
]
