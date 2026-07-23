"""Pinned WideSeek training data with evaluator-private reference records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from heterospawn.domain.ids import TaskId
from heterospawn.domain.tasks import ResearchTask
from heterospawn.errors import BenchmarkDataError

WIDESEEK_TRAIN_REPO: Literal["RLinf/WideSeek-R1-train-data"] = "RLinf/WideSeek-R1-train-data"
WIDESEEK_TRAIN_REVISION = "47832ea20581f78d32cd6b32b4b37b985cbbc9df"
WideSeekSplit = Literal["width_20k", "depth_20k", "hybrid_20k"]


class WideSeekDatasetSummary(BaseModel):
    """Credential- and answer-safe inspection output."""

    model_config = ConfigDict(frozen=True, strict=True)

    dataset: Literal["RLinf/WideSeek-R1-train-data"] = WIDESEEK_TRAIN_REPO
    revision: str
    split: WideSeekSplit
    source_digest: str
    tasks: int = Field(ge=0)
    markdown_tasks: int = Field(ge=0)
    boxed_tasks: int = Field(ge=0)


@dataclass(frozen=True, repr=False)
class _WideSeekEvaluatorRecord:
    task: ResearchTask
    answers: tuple[str, ...]
    is_markdown: bool
    unique_columns: tuple[str, ...]
    required_columns: tuple[str, ...]


class WideSeekDataset:
    """Policy-visible tasks plus evaluator-only reference answers and schema."""

    def __init__(
        self,
        split: WideSeekSplit,
        records: tuple[_WideSeekEvaluatorRecord, ...],
        source_digest: str,
        revision: str,
    ) -> None:
        self._split = split
        self._records = records
        self._record_by_id = {record.task.task_id: record for record in records}
        self._source_digest = source_digest
        self._revision = revision

    @property
    def tasks(self) -> tuple[ResearchTask, ...]:
        return tuple(record.task for record in self._records)

    @property
    def split(self) -> WideSeekSplit:
        return self._split

    @property
    def source_digest(self) -> str:
        return self._source_digest

    @property
    def revision(self) -> str:
        return self._revision

    def select_tasks(self, task_ids: tuple[TaskId, ...]) -> tuple[ResearchTask, ...]:
        if not task_ids:
            raise BenchmarkDataError("WideSeek task selection cannot be empty")
        if len(set(task_ids)) != len(task_ids):
            raise BenchmarkDataError("WideSeek task selection contains duplicate IDs")
        try:
            return tuple(self._record_by_id[task_id].task for task_id in task_ids)
        except KeyError:
            raise BenchmarkDataError("WideSeek task selection contains an unknown ID") from None

    def summary(self) -> WideSeekDatasetSummary:
        markdown = sum(record.is_markdown for record in self._records)
        return WideSeekDatasetSummary(
            revision=self._revision,
            split=self._split,
            source_digest=self._source_digest,
            tasks=len(self._records),
            markdown_tasks=markdown,
            boxed_tasks=len(self._records) - markdown,
        )

    def evaluator_record(self, task: ResearchTask) -> _WideSeekEvaluatorRecord:
        try:
            record = self._record_by_id[task.task_id]
        except KeyError:
            raise BenchmarkDataError("WideSeek evaluator received an unknown task") from None
        if record.task != task:
            raise BenchmarkDataError("WideSeek task differs from its evaluator-private record")
        return record


def load_wideseek_dataset(
    path: Path,
    *,
    split: WideSeekSplit,
    expected_sha256: str,
    revision: str = WIDESEEK_TRAIN_REVISION,
) -> WideSeekDataset:
    """Load one fully verified JSONL file without exposing references in tasks."""

    if revision != WIDESEEK_TRAIN_REVISION:
        raise BenchmarkDataError("WideSeek dataset revision is not the pinned trust root")
    source_digest = _sha256(path)
    if source_digest != expected_sha256:
        raise BenchmarkDataError("WideSeek dataset digest does not match trusted manifest")

    records: list[_WideSeekEvaluatorRecord] = []
    seen_ids: set[TaskId] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                if not line.strip():
                    raise BenchmarkDataError("WideSeek JSONL contains an empty line")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    raise BenchmarkDataError(
                        f"WideSeek JSONL line {line_index + 1} is invalid"
                    ) from None
                record = _parse_record(value, split, line_index, revision)
                if record.task.task_id in seen_ids:
                    raise BenchmarkDataError("WideSeek dataset contains duplicate task IDs")
                seen_ids.add(record.task.task_id)
                records.append(record)
    except OSError as exc:
        raise BenchmarkDataError(f"WideSeek dataset cannot be read: {type(exc).__name__}") from None
    if not records:
        raise BenchmarkDataError("WideSeek dataset is empty")
    return WideSeekDataset(split, tuple(records), source_digest, revision)


def _parse_record(
    value: object,
    split: WideSeekSplit,
    line_index: int,
    revision: str,
) -> _WideSeekEvaluatorRecord:
    if not isinstance(value, dict):
        raise BenchmarkDataError("WideSeek record must be a JSON object")
    question = value.get("question")
    answer = value.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise BenchmarkDataError("WideSeek record question must be a non-empty string")
    answers = (
        tuple(item for item in answer if isinstance(item, str) and item.strip())
        if isinstance(answer, list)
        else (answer,)
        if isinstance(answer, str) and answer.strip()
        else ()
    )
    if not answers:
        raise BenchmarkDataError("WideSeek record answer must contain non-empty strings")

    if split == "width_20k":
        allowed = {"question", "answer", "unique_columns", "evaluation", "language"}
        is_markdown = True
        identity = line_index
    elif split == "depth_20k":
        allowed = {
            "question",
            "answer",
            "source",
            "aug_answer",
            "qid",
            "language",
        }
        is_markdown = False
        identity = value.get("qid", line_index)
    else:
        allowed = {
            "question",
            "answer",
            "unique_columns",
            "is_markdown",
            "instance_id",
            "evaluation",
            "language",
        }
        marker = value.get("is_markdown")
        if not isinstance(marker, bool):
            raise BenchmarkDataError("hybrid WideSeek record requires boolean is_markdown")
        is_markdown = marker
        identity = value.get("instance_id", line_index)
    unknown = set(value) - allowed
    if unknown:
        raise BenchmarkDataError(
            f"WideSeek {split} record has unsupported fields: {sorted(unknown)}"
        )

    unique_value = value.get("unique_columns", ())
    unique_columns = (
        ()
        if unique_value is None and not is_markdown
        else _string_tuple(unique_value, "unique_columns")
    )
    required_columns: tuple[str, ...] = ()
    evaluation = value.get("evaluation")
    if isinstance(evaluation, str):
        try:
            evaluation = json.loads(evaluation)
        except json.JSONDecodeError:
            raise BenchmarkDataError("WideSeek evaluation field contains invalid JSON") from None
    if evaluation is not None:
        if not isinstance(evaluation, dict):
            raise BenchmarkDataError("WideSeek evaluation field must be an object")
        required_columns = _string_tuple(evaluation.get("required", ()), "required")
    if is_markdown and not unique_columns:
        raise BenchmarkDataError("markdown WideSeek record requires unique_columns")

    language_value = value.get("language", "en")
    language: Literal["en", "zh"] = "zh" if language_value == "zh" else "en"
    task = ResearchTask(
        task_id=TaskId(f"{split}:{identity}"),
        prompt=question,
        dataset_revision=revision,
        answer_format="markdown_table" if is_markdown else "boxed",
        language=language,
        metadata=(
            ("line_index", line_index),
            ("split", split),
            ("is_markdown", is_markdown),
        ),
    )
    return _WideSeekEvaluatorRecord(
        task=task,
        answers=answers,
        is_markdown=is_markdown,
        unique_columns=unique_columns,
        required_columns=required_columns,
    )


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise BenchmarkDataError(f"WideSeek {label} must be a string list")
    result = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    if len(result) != len(value) or len(set(result)) != len(result):
        raise BenchmarkDataError(f"WideSeek {label} must contain unique non-empty strings")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BenchmarkDataError(f"WideSeek dataset cannot be read: {type(exc).__name__}") from None
    return digest.hexdigest()
