"""Unit tests for BrowseComp-Plus official run export helpers."""

from __future__ import annotations

from types import SimpleNamespace

from heterospawn.benchmarks.browsecomp_plus_export import (
    collect_retrieved_docids,
    collect_tool_call_counts,
    to_official_run,
)


def test_collect_docids_from_search_and_access() -> None:
    outcomes = [
        SimpleNamespace(
            tool_name="search",
            url=None,
            result_json='{"results":[{"url":"bcplus://doc/111","title":"t"},{"docid":"222"}]}',
        ),
        SimpleNamespace(
            tool_name="access",
            url="bcplus://doc/111",
            result_json='{"url":"bcplus://doc/111","content":"x"}',
        ),
        SimpleNamespace(
            tool_name="access",
            url="bcplus://doc/333",
            result_json='{"url":"bcplus://doc/333"}',
        ),
    ]
    assert collect_retrieved_docids(outcomes) == ["111", "222", "333"]
    assert collect_tool_call_counts(outcomes) == {"search": 1, "access": 2}


def test_to_official_run_completed() -> None:
    outcomes = [
        SimpleNamespace(
            tool_name="search",
            url=None,
            result_json='{"results":[{"url":"bcplus://doc/9"}]}',
        )
    ]
    run = to_official_run(
        query_id="42",
        answer="Paris",
        status="success",
        tool_outcomes=outcomes,
        model="heterospawn-sft-1024",
        retriever="BM25-official-pyserini",
    )
    assert run["status"] == "completed"
    assert run["retrieved_docids"] == ["9"]
    assert run["result"][0]["output"] == "Paris"
    assert run["metadata"]["retriever"] == "BM25-official-pyserini"


def test_to_official_run_failed_without_answer() -> None:
    run = to_official_run(
        query_id="1",
        answer="",
        status="success",
        tool_outcomes=[],
        model="m",
        retriever="r",
    )
    assert run["status"] == "failed"
