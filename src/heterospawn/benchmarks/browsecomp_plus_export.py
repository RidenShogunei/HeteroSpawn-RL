"""Helpers to export HeteroSpawn BrowseComp-Plus traces into official run schema."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

_DOC_URL_RE = re.compile(r"^bcplus://doc/(.+)$")


def docid_from_url(url: str | None) -> str | None:
    if not url:
        return None
    match = _DOC_URL_RE.match(url)
    return match.group(1) if match else None


def collect_retrieved_docids(tool_outcomes: Iterable[Any]) -> list[str]:
    """Union of docids seen via Search hits or Access URLs."""
    docids: set[str] = set()
    for outcome in tool_outcomes:
        url = getattr(outcome, "url", None)
        docid = docid_from_url(url)
        if docid:
            docids.add(docid)
        raw = getattr(outcome, "result_json", None)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        for item in payload.get("results") or []:
            if not isinstance(item, dict):
                continue
            item_docid = item.get("docid")
            if item_docid:
                docids.add(str(item_docid))
            nested = docid_from_url(item.get("url"))
            if nested:
                docids.add(nested)
        # Access-style payloads may only have url.
        nested = docid_from_url(payload.get("url"))
        if nested:
            docids.add(nested)
    return sorted(docids)


def collect_tool_call_counts(tool_outcomes: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for outcome in tool_outcomes:
        name = getattr(outcome, "tool_name", None) or "unknown"
        counts[name] = counts.get(name, 0) + 1
    return counts


def to_official_run(
    *,
    query_id: str,
    answer: str | None,
    status: str,
    tool_outcomes: Iterable[Any],
    model: str,
    retriever: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    answer_text = answer or ""
    official_status = "completed" if status == "success" and answer_text else "failed"
    run: dict[str, Any] = {
        "query_id": str(query_id),
        "tool_call_counts": collect_tool_call_counts(tool_outcomes),
        "status": official_status,
        "retrieved_docids": collect_retrieved_docids(tool_outcomes),
        "result": [{"type": "output_text", "output": answer_text}],
        "metadata": {
            "model": model,
            "retriever": retriever,
            "source": "heterospawn",
            "source_status": status,
        },
    }
    if extra:
        run["metadata"].update(extra)
    return run
