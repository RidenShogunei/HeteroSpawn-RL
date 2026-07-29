"""Unit tests for BrowseComp-Plus local BM25 tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from heterospawn.search.base import AccessRequest, SearchRequest
from heterospawn.search.browsecomp_plus_local import BrowseCompPlusLocalTools


@pytest.mark.asyncio
async def test_bm25_search_and_access(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    docs = [
        {"docid": "a", "text": "The capital of France is Paris."},
        {"docid": "b", "text": "Tokyo is the capital of Japan."},
        {"docid": "c", "text": "Unrelated gardening tips."},
    ]
    corpus.write_text("\n".join(json.dumps(d) for d in docs) + "\n", encoding="utf-8")
    tools = BrowseCompPlusLocalTools(corpus)
    response = await tools.search(
        SearchRequest(request_id="s1", query="capital of France", max_results=2)
    )
    assert response.results
    assert "bcplus://doc/" in response.results[0].url
    access = await tools.access(
        AccessRequest(
            request_id="a1",
            url=response.results[0].url,
            info_to_extract="capital",
            max_characters=200,
        )
    )
    assert "Paris" in access.content or "France" in access.content
