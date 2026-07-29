"""BrowseComp-Plus query loader (decrypted JSONL)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from heterospawn.domain.ids import TaskId
from heterospawn.domain.tasks import ResearchTask
from heterospawn.errors import BenchmarkDataError

BCPLUS_ANSWER_FORMAT = "boxed"


class BrowseCompPlusRecord(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    evidence_docs: tuple[str, ...] = ()
    gold_docs: tuple[str, ...] = ()


class BrowseCompPlusDataset(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    source_digest: str
    records: tuple[BrowseCompPlusRecord, ...]

    @property
    def tasks(self) -> tuple[ResearchTask, ...]:
        return tuple(
            ResearchTask(
                task_id=TaskId(f"browsecomp-plus:{record.query_id}"),
                prompt=record.query,
                answer_format="boxed",
                dataset_revision="browsecomp-plus-decrypted",
            )
            for record in self.records
        )


def load_browsecomp_plus_dataset(path: Path) -> BrowseCompPlusDataset:
    if not path.is_file():
        raise BenchmarkDataError(f"BrowseComp-Plus dataset missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    records: list[BrowseCompPlusRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BenchmarkDataError(
                    f"BrowseComp-Plus JSONL line {line_index + 1} is invalid"
                ) from exc
            records.append(_parse_record(row, line_index))
    if not records:
        raise BenchmarkDataError("BrowseComp-Plus dataset is empty")
    return BrowseCompPlusDataset(source_digest=digest, records=tuple(records))


def _parse_record(row: object, line_index: int) -> BrowseCompPlusRecord:
    if not isinstance(row, dict):
        raise BenchmarkDataError(f"BrowseComp-Plus line {line_index + 1} must be an object")
    query_id = row.get("query_id")
    query = row.get("query")
    answer = row.get("answer")
    if not isinstance(query_id, str) or not query_id.strip():
        raise BenchmarkDataError(f"BrowseComp-Plus line {line_index + 1} missing query_id")
    if not isinstance(query, str) or not query.strip():
        raise BenchmarkDataError(f"BrowseComp-Plus line {line_index + 1} missing query")
    if not isinstance(answer, str) or not answer.strip():
        raise BenchmarkDataError(f"BrowseComp-Plus line {line_index + 1} missing answer")
    evidence = row.get("evidence_docs") or row.get("evidence") or []
    gold = row.get("gold_docs") or row.get("gold") or []
    if not isinstance(evidence, list):
        evidence = []
    if not isinstance(gold, list):
        gold = []

    def _ids(items: list[object]) -> tuple[str, ...]:
        out: list[str] = []
        for item in items:
            if isinstance(item, dict):
                docid = item.get("docid") or item.get("id")
                if docid is not None:
                    out.append(str(docid))
            elif item is not None:
                out.append(str(item))
        return tuple(out)

    return BrowseCompPlusRecord(
        query_id=query_id.strip(),
        query=query.strip(),
        answer=answer.strip(),
        evidence_docs=_ids(evidence),
        gold_docs=_ids(gold),
    )
