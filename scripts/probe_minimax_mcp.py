"""Credential-safe live probe for MiniMax Token Plan MCP web search."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from heterospawn.search.minimax_mcp import (
    MINIMAX_MCP_PACKAGE,
    _build_mcp_environment,
)


async def probe() -> dict[str, object]:
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY is required")

    environment = _build_mcp_environment(api_key, "https://api.minimaxi.com")
    parameters = StdioServerParameters(
        command="uvx",
        args=[
            "--from",
            MINIMAX_MCP_PACKAGE,
            "minimax-coding-plan-mcp",
        ],
        env=environment,
    )
    with Path(os.devnull).open("w", encoding="utf-8") as error_log:  # noqa: ASYNC230
        async with stdio_client(parameters, errlog=error_log) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                tool_names = tuple(sorted(tool.name for tool in tools.tools))
                if "web_search" not in tool_names:
                    raise RuntimeError("MiniMax MCP did not advertise web_search")
                result = await session.call_tool(
                    "web_search",
                    arguments={"query": "Python official documentation"},
                )

    if result.isError:
        raise RuntimeError("MiniMax MCP web_search returned a tool error")
    text_blocks = [block.text for block in result.content if isinstance(block, types.TextContent)]
    if len(text_blocks) != 1:
        raise RuntimeError("MiniMax MCP web_search returned an unexpected content shape")
    payload = json.loads(text_blocks[0])
    organic = payload.get("organic")
    related = payload.get("related_searches", [])
    base_response = payload.get("base_resp", {})
    if not isinstance(organic, list) or not isinstance(related, list):
        raise RuntimeError("MiniMax MCP web_search returned an invalid JSON schema")

    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "server": "minimax-coding-plan-mcp",
        "tools": tool_names,
        "organic_result_count": len(organic),
        "related_search_count": len(related),
        "provider_status_code": base_response.get("status_code"),
        "response_digest": digest,
    }


def main() -> int:
    print(json.dumps(asyncio.run(probe()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
