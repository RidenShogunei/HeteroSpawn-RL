"""WideSearch / WideSeek-R1 official test-set loader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from heterospawn.benchmarks.wideseek import (
    WideSeekDataset,
    _WideSeekEvaluatorRecord,
)
from heterospawn.domain.ids import TaskId
from heterospawn.domain.tasks import ResearchTask
from heterospawn.errors import BenchmarkDataError

WIDESEARCH_TEST_REPO: Literal["RLinf/WideSeek-R1-test-data"] = "RLinf/WideSeek-R1-test-data"
WIDESEARCH_TEST_DIGEST = "402285fc1bc28b662c363d0ea04103f8be79b6a38f6035471985dc45b4ada143"


class WideSearchDatasetSummary(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    dataset: Literal["RLinf/WideSeek-R1-test-data"] = WIDESEARCH_TEST_REPO
    source_digest: str
    tasks: int = Field(ge=0)
    english_tasks: int = Field(ge=0)
    chinese_tasks: int = Field(ge=0)


def load_widesearch_test_dataset(
    path: Path,
    *,
    expected_sha256: str = WIDESEARCH_TEST_DIGEST,
    revision: str = "widesearch-test-jsonl",
) -> WideSeekDataset:
    """Load the pinned 200-task WideSearch test split as markdown WideSeek records."""

    source_digest = _sha256(path)
    if source_digest != expected_sha256:
        raise BenchmarkDataError("WideSearch test digest does not match the trusted digest")

    records: list[_WideSeekEvaluatorRecord] = []
    seen: set[TaskId] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                if not line.strip():
                    raise BenchmarkDataError("WideSearch JSONL contains an empty line")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BenchmarkDataError(
                        f"WideSearch JSONL line {line_index + 1} is invalid"
                    ) from exc
                record = _parse_widesearch_record(value, line_index, revision)
                if record.task.task_id in seen:
                    raise BenchmarkDataError("WideSearch dataset contains duplicate task IDs")
                seen.add(record.task.task_id)
                records.append(record)
    except OSError as exc:
        raise BenchmarkDataError(f"WideSearch dataset cannot be read: {type(exc).__name__}") from None
    if len(records) != 200:
        raise BenchmarkDataError(f"WideSearch test set must contain 200 tasks, got {len(records)}")
    return WideSeekDataset("hybrid_20k", tuple(records), source_digest, revision)


def summarize_widesearch(dataset: WideSeekDataset) -> WideSearchDatasetSummary:
    english = sum(1 for task in dataset.tasks if str(task.task_id).startswith("widesearch:ws_en"))
    return WideSearchDatasetSummary(
        source_digest=dataset.source_digest,
        tasks=len(dataset.tasks),
        english_tasks=english,
        chinese_tasks=len(dataset.tasks) - english,
    )


def _parse_widesearch_record(
    value: object,
    line_index: int,
    revision: str,
) -> _WideSeekEvaluatorRecord:
    if not isinstance(value, dict):
        raise BenchmarkDataError("WideSearch record must be a JSON object")
    allowed = {
        "instance_id",
        "query",
        "evaluation",
        "language",
        "answer",
        "unique_columns",
    }
    unknown = set(value) - allowed
    if unknown:
        raise BenchmarkDataError(f"WideSearch record has unsupported fields: {sorted(unknown)}")

    instance_id = value.get("instance_id")
    query = value.get("query")
    answer = value.get("answer")
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise BenchmarkDataError("WideSearch instance_id must be a non-empty string")
    if not isinstance(query, str) or not query.strip():
        raise BenchmarkDataError("WideSearch query must be a non-empty string")
    if not isinstance(answer, str) or not answer.strip():
        raise BenchmarkDataError("WideSearch answer must be a non-empty string")

    unique_value = value.get("unique_columns", ())
    if not isinstance(unique_value, list) or not unique_value:
        raise BenchmarkDataError("WideSearch unique_columns must be a non-empty list")
    unique_columns = tuple(
        item.strip() for item in unique_value if isinstance(item, str) and item.strip()
    )
    if len(unique_columns) != len(unique_value) or len(set(unique_columns)) != len(unique_columns):
        raise BenchmarkDataError("WideSearch unique_columns must contain unique non-empty strings")

    required_columns: tuple[str, ...] = ()
    evaluation = value.get("evaluation")
    if isinstance(evaluation, str):
        try:
            evaluation = json.loads(evaluation)
        except json.JSONDecodeError as exc:
            raise BenchmarkDataError("WideSearch evaluation field contains invalid JSON") from exc
    if isinstance(evaluation, dict):
        required_raw = evaluation.get("required", ())
        if isinstance(required_raw, list):
            required_columns = tuple(
                item.strip() for item in required_raw if isinstance(item, str) and item.strip()
            )

    language_value = value.get("language", "en")
    language: Literal["en", "zh"] = "zh" if language_value == "zh" else "en"
    task = ResearchTask(
        task_id=TaskId(f"widesearch:{instance_id}"),
        prompt=query,
        dataset_revision=revision,
        answer_format="markdown_table",
        language=language,
        metadata=(
            ("line_index", line_index),
            ("split", "widesearch_test"),
            ("is_markdown", True),
            ("instance_id", instance_id),
        ),
    )
    return _WideSeekEvaluatorRecord(
        task=task,
        answers=(answer,),
        is_markdown=True,
        unique_columns=unique_columns,
        required_columns=required_columns,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
