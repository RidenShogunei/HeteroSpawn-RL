"""MiniMax Token Plan MCP `web_search` adapter."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Protocol

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from heterospawn.errors import ConfigurationError, SearchRequestError
from heterospawn.search.base import SearchItem, SearchRequest, SearchResponse

DEFAULT_MINIMAX_MCP_API_HOST = "https://api.minimaxi.com"
MINIMAX_MCP_PACKAGE = "minimax-coding-plan-mcp==0.0.4"
MINIMAX_MCP_REVISION = "minimax-coding-plan-mcp@0.0.4"
_PASSTHROUGH_ENVIRONMENT_NAMES = (
    "APPDATA",
    "COMSPEC",
    "HOME",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "LOCALAPPDATA",
    "NO_PROXY",
    "PATH",
    "PATHEXT",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USERPROFILE",
    "UV_CACHE_DIR",
    "WINDIR",
    "XDG_CACHE_HOME",
)


class MiniMaxMcpConfig(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    api_key: SecretStr
    api_host: str = Field(default=DEFAULT_MINIMAX_MCP_API_HOST, min_length=1)
    command: str = Field(default="uvx", min_length=1)
    timeout_seconds: float = Field(default=60.0, gt=0)

    @classmethod
    def from_environment(cls) -> MiniMaxMcpConfig:
        api_key = os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            raise ConfigurationError("MINIMAX_API_KEY is required for MiniMax MCP search")
        return cls(
            api_key=SecretStr(api_key),
            api_host=os.environ.get("MINIMAX_API_HOST", DEFAULT_MINIMAX_MCP_API_HOST),
        )


class MiniMaxMcpTransport(Protocol):
    async def web_search(self, query: str) -> str: ...


class StdioMiniMaxMcpTransport:
    """Starts the pinned official MCP server for one isolated tool call."""

    def __init__(self, config: MiniMaxMcpConfig) -> None:
        self._config = config

    async def web_search(self, query: str) -> str:
        environment = _build_mcp_environment(
            self._config.api_key.get_secret_value(), self._config.api_host
        )
        parameters = StdioServerParameters(
            command=self._config.command,
            args=["--from", MINIMAX_MCP_PACKAGE, "minimax-coding-plan-mcp"],
            env=environment,
        )
        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                with Path(os.devnull).open("w", encoding="utf-8") as error_log:  # noqa: ASYNC230
                    async with stdio_client(parameters, errlog=error_log) as (
                        read_stream,
                        write_stream,
                    ):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            tools = await session.list_tools()
                            tool_names = tuple(tool.name for tool in tools.tools)
                            result = await session.call_tool(
                                "web_search", arguments={"query": query}
                            )
            return _extract_tool_text(tool_names, result)
        except TimeoutError:
            raise SearchRequestError("MiniMax MCP web_search timed out") from None
        except SearchRequestError:
            raise
        except Exception:
            raise SearchRequestError("MiniMax MCP web_search failed") from None


class _OrganicResult(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    title: str
    link: str = Field(min_length=1)
    snippet: str
    date: str | None = None


class _BaseResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    status_code: int


class _SearchPayload(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    organic: tuple[_OrganicResult, ...]
    base_resp: _BaseResponse


class MiniMaxMcpSearchService:
    def __init__(
        self,
        config: MiniMaxMcpConfig,
        *,
        transport: MiniMaxMcpTransport | None = None,
    ) -> None:
        self._transport = transport or StdioMiniMaxMcpTransport(config)

    async def search(self, request: SearchRequest) -> SearchResponse:
        raw_text = await self._transport.web_search(request.query)
        try:
            payload = _SearchPayload.model_validate_json(raw_text, strict=True)
        except (ValueError, ValidationError):
            raise SearchRequestError("MiniMax MCP returned an invalid search schema") from None
        if payload.base_resp.status_code != 0:
            raise SearchRequestError(
                f"MiniMax MCP search failed with provider status {payload.base_resp.status_code}"
            )

        digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        return SearchResponse(
            request_id=request.request_id,
            provider="minimax-mcp-search",
            provider_revision=MINIMAX_MCP_REVISION,
            provider_request_id=f"mcp:{digest[:16]}",
            query=request.query,
            results=tuple(
                SearchItem(
                    title=item.title,
                    url=item.link,
                    content=item.snippet,
                    score=None,
                )
                for item in payload.organic[: request.max_results]
            ),
            credits=None,
            raw_response_digest=digest,
        )


def _extract_tool_text(tool_names: tuple[str, ...], result: types.CallToolResult) -> str:
    if "web_search" not in tool_names:
        raise SearchRequestError("MiniMax MCP did not advertise web_search")
    if result.isError:
        raise SearchRequestError("MiniMax MCP web_search returned a tool error")
    text_blocks = tuple(
        block.text for block in result.content if isinstance(block, types.TextContent)
    )
    if len(text_blocks) != 1:
        raise SearchRequestError("MiniMax MCP web_search returned an unexpected content shape")
    return text_blocks[0]


def _build_mcp_environment(api_key: str, api_host: str) -> dict[str, str]:
    environment = {
        name: value
        for name in _PASSTHROUGH_ENVIRONMENT_NAMES
        if (value := os.environ.get(name)) is not None
    }
    environment.update(
        {
            "MINIMAX_API_KEY": api_key,
            "MINIMAX_API_HOST": api_host,
            "FASTMCP_LOG_LEVEL": "ERROR",
        }
    )
    return environment
