"""Secure, pinned adapter for xbench-DeepSearch-2510.

The upstream dataset is encrypted at rest. Decryption happens only in memory,
and answers remain behind the evaluator surface.
"""

from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from heterospawn.domain.ids import TaskId
from heterospawn.errors import BenchmarkDataError

XBENCH_UPSTREAM_REVISION = "17c562192cc7e62215bfb98b65e9f8806fb95504"
XBENCH_DATASET_PATH = "data/DeepSearch-2510.csv"
XBENCH_DATASET_SHA256 = "a9378e56b05ec8f007b8ecc8f6ac74900abafd558267acd5839d0d05fbc6977a"


class BenchmarkTask(BaseModel):
    """Policy-visible task data. Ground truth is intentionally absent."""

    model_config = ConfigDict(frozen=True, strict=True)

    task_id: TaskId
    prompt: str = Field(min_length=1)


class ExactScoreReport(BaseModel):
    """Development score that does not claim official leaderboard parity."""

    model_config = ConfigDict(frozen=True, strict=True)

    benchmark: str
    dataset_revision: str
    evaluator_revision: str
    mode: str
    comparable_to_official: bool
    correct: int = Field(ge=0)
    total: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)


class RepeatExactScoreReport(BaseModel):
    """Repeat-aware development metrics without the upstream Gemini judge."""

    model_config = ConfigDict(frozen=True, strict=True)

    benchmark: str
    dataset_revision: str
    evaluator_revision: str
    mode: str
    comparable_to_official: Literal[False] = False
    tasks: int = Field(ge=0)
    repeats_per_task: int = Field(ge=1)
    completed_episodes: int = Field(ge=0)
    exact_correct_episodes: int = Field(ge=0)
    total_episodes: int = Field(ge=0)
    average_exact_accuracy: float = Field(ge=0.0, le=1.0)
    best_of_n_correct_tasks: int = Field(ge=0)
    best_of_n_exact_accuracy: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True, repr=False)
class _EvaluatorRecord:
    task: BenchmarkTask
    answer: str


class XBenchDataset:
    """Task collection with evaluator-only access to decrypted answers."""

    def __init__(self, records: tuple[_EvaluatorRecord, ...], source_digest: str) -> None:
        self._records = records
        self._source_digest = source_digest

    @property
    def tasks(self) -> tuple[BenchmarkTask, ...]:
        return tuple(record.task for record in self._records)

    @property
    def source_digest(self) -> str:
        return self._source_digest

    def select_tasks(self, task_ids: tuple[TaskId, ...]) -> tuple[BenchmarkTask, ...]:
        """Return tasks in caller order after rejecting duplicates and unknown IDs."""

        if not task_ids:
            raise BenchmarkDataError("task selection cannot be empty")
        if len(frozenset(task_ids)) != len(task_ids):
            raise BenchmarkDataError("task selection contains duplicate task ids")
        task_by_id = {record.task.task_id: record.task for record in self._records}
        try:
            return tuple(task_by_id[task_id] for task_id in task_ids)
        except KeyError:
            raise BenchmarkDataError("task selection contains an unknown task id") from None

    def evaluate_exact(
        self,
        predictions: dict[TaskId, str],
        *,
        task_ids: tuple[TaskId, ...] | None = None,
    ) -> ExactScoreReport:
        """Run only xbench's deterministic exact-match shortcut.

        Upstream sends non-exact responses to a Gemini judge. Because that
        fallback is absent here, this mode is always marked non-comparable.
        """

        selected_records = self._records
        if task_ids is not None:
            requested = frozenset(task_ids)
            selected_records = tuple(
                record for record in self._records if record.task.task_id in requested
            )
            if len(selected_records) != len(requested):
                raise BenchmarkDataError("score scope contains an unknown task id")

        correct = 0
        for record in selected_records:
            prediction = predictions.get(record.task.task_id)
            if prediction is not None and parse_final_answer(prediction) == record.answer:
                correct += 1
        total = len(selected_records)
        return ExactScoreReport(
            benchmark="xbench-DeepSearch-2510",
            dataset_revision=XBENCH_UPSTREAM_REVISION,
            evaluator_revision=XBENCH_UPSTREAM_REVISION,
            mode="development-exact-only",
            comparable_to_official=False,
            correct=correct,
            total=total,
            accuracy=(correct / total) if total else 0.0,
        )

    def evaluate_repeat_exact(
        self,
        predictions: dict[TaskId, tuple[str | None, ...]],
        *,
        task_ids: tuple[TaskId, ...],
        repeats_per_task: int,
    ) -> RepeatExactScoreReport:
        """Score repeated episodes using only the upstream direct-match shortcut.

        Missing or failed episodes are counted as incorrect. The upstream judge,
        random tie-breaking majority vote, and official aggregation are deliberately
        absent, so this report is never official-comparable.
        """

        if repeats_per_task < 1:
            raise BenchmarkDataError("repeats_per_task must be at least one")
        selected_tasks = self.select_tasks(task_ids)
        if frozenset(predictions) != frozenset(task_ids):
            raise BenchmarkDataError("repeat score predictions do not match task selection")
        record_by_id = {record.task.task_id: record for record in self._records}

        completed = 0
        exact_correct = 0
        best_correct = 0
        for task in selected_tasks:
            task_predictions = predictions[task.task_id]
            if len(task_predictions) != repeats_per_task:
                raise BenchmarkDataError("repeat score prediction count does not match manifest")
            task_has_exact = False
            for prediction in task_predictions:
                if prediction is None:
                    continue
                completed += 1
                if parse_final_answer(prediction) == record_by_id[task.task_id].answer:
                    exact_correct += 1
                    task_has_exact = True
            best_correct += int(task_has_exact)

        total_episodes = len(selected_tasks) * repeats_per_task
        task_count = len(selected_tasks)
        return RepeatExactScoreReport(
            benchmark="xbench-DeepSearch-2510",
            dataset_revision=XBENCH_UPSTREAM_REVISION,
            evaluator_revision=XBENCH_UPSTREAM_REVISION,
            mode="development-repeat-exact-only",
            tasks=task_count,
            repeats_per_task=repeats_per_task,
            completed_episodes=completed,
            exact_correct_episodes=exact_correct,
            total_episodes=total_episodes,
            average_exact_accuracy=(exact_correct / total_episodes) if total_episodes else 0.0,
            best_of_n_correct_tasks=best_correct,
            best_of_n_exact_accuracy=(best_correct / task_count) if task_count else 0.0,
        )


def parse_final_answer(response: str) -> str | None:
    """Mirror the deterministic `最终答案` extraction in pinned upstream code."""

    match = re.search(r"最终答案:*(.*)", response)
    if match is None:
        return None
    matched_text = match.group(0)
    try:
        return matched_text.split(":")[1].strip()
    except IndexError:
        return matched_text


def load_xbench(path: Path, *, verify_official_digest: bool = True) -> XBenchDataset:
    """Load an encrypted xbench CSV without persisting decrypted fields."""

    try:
        encrypted_bytes = path.read_bytes()
    except OSError as exc:
        raise BenchmarkDataError(f"cannot read encrypted benchmark file: {path.name}") from exc

    digest = hashlib.sha256(encrypted_bytes).hexdigest()
    if verify_official_digest and digest != XBENCH_DATASET_SHA256:
        raise BenchmarkDataError("encrypted benchmark digest does not match pinned revision")

    try:
        text = encrypted_bytes.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        required = {"id", "prompt", "answer", "canary"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise BenchmarkDataError("encrypted benchmark has an invalid header")

        records: list[_EvaluatorRecord] = []
        for row_number, row in enumerate(reader, start=2):
            task_id = TaskId(_required_cell(row, "id", row_number))
            key = _required_cell(row, "canary", row_number)
            prompt = _decrypt_cell(_required_cell(row, "prompt", row_number), key, row_number)
            answer = _decrypt_cell(_required_cell(row, "answer", row_number), key, row_number)
            records.append(
                _EvaluatorRecord(
                    task=BenchmarkTask(task_id=task_id, prompt=prompt),
                    answer=answer,
                )
            )
    except (UnicodeDecodeError, csv.Error) as exc:
        raise BenchmarkDataError("encrypted benchmark is not a valid UTF-8 CSV") from exc

    return XBenchDataset(tuple(records), digest)


def _required_cell(row: dict[str, str | None], name: str, row_number: int) -> str:
    value = row.get(name)
    if value is None or value == "":
        raise BenchmarkDataError(f"encrypted benchmark row {row_number} is missing {name}")
    return value


def _decrypt_cell(encoded: str, key: str, row_number: int) -> str:
    if not key:
        raise BenchmarkDataError(f"encrypted benchmark row {row_number} has an empty canary")
    try:
        ciphertext = base64.b64decode(encoded, validate=True)
        key_bytes = key.encode("utf-8")
        plaintext = bytes(
            value ^ key_bytes[index % len(key_bytes)] for index, value in enumerate(ciphertext)
        )
        return plaintext.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise BenchmarkDataError(
            f"encrypted benchmark row {row_number} cannot be decrypted"
        ) from exc
