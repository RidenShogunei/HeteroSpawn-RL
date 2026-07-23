from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from heterospawn.search.wideseek_local import (
    WideSeekLocalConfig,
    WideSeekLocalToolService,
)
from heterospawn.training.wideseek_smoke import run_wideseek_rollout_smoke


@pytest.mark.asyncio
async def test_rollout_smoke_forces_real_shape_search_then_access_without_plaintext_report(
    tmp_path: Path,
) -> None:
    url = "https://en.wikipedia.org/wiki/Red_Bull"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/retrieve":
            return httpx.Response(
                200,
                json={
                    "result": [
                        [
                            {
                                "document": {
                                    "url": url,
                                    "contents": "retrieved private snippet",
                                },
                                "score": 1.0,
                            }
                        ]
                    ]
                },
            )
        if request.url.path == "/access":
            assert json.loads(request.content) == {"urls": [url]}
            return httpx.Response(
                200,
                json={
                    "result": [
                        {
                            "url": url,
                            "contents": "accessed private full page",
                        }
                    ]
                },
            )
        return httpx.Response(404)

    report_path = tmp_path / "report.json"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tools = WideSeekLocalToolService(WideSeekLocalConfig(), client=client)
        report = await run_wideseek_rollout_smoke(
            service_url="http://unused.invalid",
            qdrant_url="http://unused.invalid",
            report_path=report_path,
            tool_service=tools,
        )

    assert report["status"] == "passed"
    assert report["environment_mode"] == "controlled-fixture"
    assert report["tool_sequence"] == ["search", "access"]
    assert report["access_has_search_provenance"] is True
    persisted = report_path.read_text(encoding="utf-8")
    assert "private" not in persisted
    assert "wikipedia" not in persisted
    assert "Red Bull" not in persisted
