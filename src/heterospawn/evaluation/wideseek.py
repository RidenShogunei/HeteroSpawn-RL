"""WideSeek boxed/Markdown parsing and item-level semantic evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, ConfigDict, Field

from heterospawn.benchmarks.wideseek import WideSeekDataset
from heterospawn.domain.ids import TaskId
from heterospawn.domain.tasks import ResearchTask
from heterospawn.domain.training import canonical_digest
from heterospawn.evaluation.semantic_judge import (
    SemanticJudge,
    SemanticJudgeOperation,
    SemanticJudgeRequest,
)

WIDESEEK_EVALUATOR_UPSTREAM = "d9f3d8a9db4d7aad1d641029293295503dd3eb2c"


class MarkdownTable(BaseModel):
    """Small immutable table representation independent of pandas."""

    model_config = ConfigDict(frozen=True, strict=True)

    columns: tuple[str, ...] = Field(min_length=1)
    rows: tuple[tuple[str, ...], ...] = Field(min_length=1)

    def model_post_init(self, __context: object) -> None:
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("Markdown table columns must be unique")
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("Markdown table rows must align with columns")


class WideSeekEvaluation(BaseModel):
    """Safe evaluator result with no prediction, reference, or Judge response text."""

    model_config = ConfigDict(frozen=True, strict=True)

    task_id: TaskId
    evaluator_revision: str = Field(min_length=1)
    format_ok: bool
    outcome_score: float = Field(ge=0.0, le=1.0)
    true_positive_items: float = Field(ge=0.0)
    predicted_items: int = Field(ge=0)
    reference_items: int = Field(ge=0)
    judge_calls: int = Field(ge=0)
    judge_cache_hits: int = Field(ge=0)
    judge_response_digests: tuple[str, ...]


@dataclass
class _JudgeAudit:
    calls: int = 0
    cache_hits: int = 0
    response_digests: list[str] | None = None

    def __post_init__(self) -> None:
        if self.response_digests is None:
            self.response_digests = []


class WideSeekEvaluator:
    """Evaluator-private reference access with optional semantic Judge."""

    def __init__(
        self,
        dataset: WideSeekDataset,
        judge: SemanticJudge | None = None,
    ) -> None:
        self._dataset = dataset
        self._judge = judge
        self._revision = canonical_digest(
            {
                "upstream": WIDESEEK_EVALUATOR_UPSTREAM,
                "dataset_revision": dataset.revision,
                "dataset_digest": dataset.source_digest,
                "parser": "heterospawn-wideseek-parser-v1",
                "judge": (judge.revision.model_dump(mode="json") if judge is not None else None),
            }
        )

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def judge_revision(self) -> object | None:
        return self._judge.revision if self._judge is not None else None

    async def evaluate(
        self,
        task: ResearchTask,
        response: str,
        *,
        request_id: str,
    ) -> WideSeekEvaluation:
        record = self._dataset.evaluator_record(task)
        audit = _JudgeAudit()
        if record.is_markdown:
            prediction = parse_markdown_table(response, strict=True)
            reference = parse_markdown_table(record.answers[0], strict=False)
            if prediction is None or reference is None:
                return self._result(task.task_id, False, 0.0, 0.0, 0, 0, audit)
            return await self._evaluate_markdown(
                task,
                prediction,
                reference,
                record.unique_columns,
                record.required_columns,
                request_id,
                audit,
            )

        boxed_prediction = parse_boxed_answer(response)
        if boxed_prediction is None:
            return self._result(task.task_id, False, 0.0, 0.0, 0, 1, audit)
        if any(
            _normalize_value(boxed_prediction) == _normalize_value(item) for item in record.answers
        ):
            return self._result(task.task_id, True, 1.0, 1.0, 1, 1, audit)
        score = 0
        if self._judge is not None:
            result = await self._judge_pairs(
                task=task,
                operation="answer_equivalence",
                candidates=tuple(boxed_prediction for _ in record.answers),
                references=record.answers,
                request_id=f"{request_id}:answer",
                audit=audit,
            )
            score = max(result, default=0)
        return self._result(task.task_id, True, float(score), float(score), 1, 1, audit)

    async def _evaluate_markdown(
        self,
        task: ResearchTask,
        prediction: MarkdownTable,
        reference: MarkdownTable,
        unique_columns: tuple[str, ...],
        configured_required: tuple[str, ...],
        request_id: str,
        audit: _JudgeAudit,
    ) -> WideSeekEvaluation:
        reference_columns = {_normalize_column(column): column for column in reference.columns}
        required = configured_required or reference.columns
        required_norm = tuple(_normalize_column(column) for column in required)
        unique_norm = tuple(_normalize_column(column) for column in unique_columns)
        if not set(required_norm).issubset(reference_columns):
            return self._result(task.task_id, False, 0.0, 0.0, 0, 0, audit)

        prediction_columns = {_normalize_column(column): column for column in prediction.columns}
        missing = tuple(column for column in required_norm if column not in prediction_columns)
        if missing and self._judge is not None:
            prediction_columns = await self._align_columns(
                task,
                prediction.columns,
                required,
                request_id,
                audit,
            )
        if not set(required_norm).issubset(prediction_columns) or not set(unique_norm).issubset(
            prediction_columns
        ):
            return self._result(task.task_id, False, 0.0, 0.0, 0, 0, audit)

        reference_rows = _dedupe_rows(reference, reference_columns, unique_norm)
        prediction_rows = _dedupe_rows(prediction, prediction_columns, unique_norm)
        reference_by_key = {
            _row_key(row, reference_columns, unique_norm): row for row in reference_rows
        }
        prediction_keys = [
            _row_key(row, prediction_columns, unique_norm) for row in prediction_rows
        ]
        mapped_keys = await self._align_keys(
            task,
            tuple(prediction_keys),
            tuple(reference_by_key),
            request_id,
            audit,
        )

        true_positive = 0.0
        semantic_candidates: list[str] = []
        semantic_references: list[str] = []
        for row, prediction_key in zip(prediction_rows, prediction_keys, strict=True):
            reference_key = mapped_keys.get(prediction_key, prediction_key)
            reference_row = reference_by_key.get(reference_key)
            if reference_row is None:
                continue
            for column in required_norm:
                candidate = row[prediction.columns.index(prediction_columns[column])]
                target = reference_row[reference.columns.index(reference_columns[column])]
                if column in unique_norm or _normalize_value(candidate) == _normalize_value(target):
                    true_positive += 1.0
                else:
                    semantic_candidates.append(candidate)
                    semantic_references.append(target)

        if semantic_candidates and self._judge is not None:
            scores = await self._judge_pairs(
                task=task,
                operation="cell_equivalence",
                candidates=tuple(semantic_candidates),
                references=tuple(semantic_references),
                request_id=f"{request_id}:cells",
                audit=audit,
            )
            true_positive += sum(scores)

        predicted_items = len(prediction_rows) * len(required_norm)
        reference_items = len(reference_rows) * len(required_norm)
        precision = true_positive / predicted_items if predicted_items else 0.0
        recall = true_positive / reference_items if reference_items else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return self._result(
            task.task_id,
            True,
            f1,
            true_positive,
            predicted_items,
            reference_items,
            audit,
        )

    async def _align_columns(
        self,
        task: ResearchTask,
        candidates: tuple[str, ...],
        references: tuple[str, ...],
        request_id: str,
        audit: _JudgeAudit,
    ) -> dict[str, str]:
        mapping = {_normalize_column(column): column for column in candidates}
        unresolved = [
            column
            for column in candidates
            if _normalize_column(column)
            not in {_normalize_column(reference) for reference in references}
        ]
        for index, candidate in enumerate(unresolved):
            scores = await self._judge_pairs(
                task=task,
                operation="column_equivalence",
                candidates=tuple(candidate for _ in references),
                references=references,
                request_id=f"{request_id}:column:{index}",
                audit=audit,
            )
            matches = [
                reference for reference, score in zip(references, scores, strict=True) if score
            ]
            if len(matches) == 1:
                mapping[_normalize_column(matches[0])] = candidate
        return mapping

    async def _align_keys(
        self,
        task: ResearchTask,
        candidates: tuple[tuple[str, ...], ...],
        references: tuple[tuple[str, ...], ...],
        request_id: str,
        audit: _JudgeAudit,
    ) -> dict[tuple[str, ...], tuple[str, ...]]:
        reference_set = set(references)
        mapping: dict[tuple[str, ...], tuple[str, ...]] = {}
        if self._judge is None:
            return mapping
        for index, candidate in enumerate(candidates):
            if candidate in reference_set:
                continue
            candidate_text = json.dumps(candidate, ensure_ascii=False)
            reference_texts = tuple(
                json.dumps(reference, ensure_ascii=False) for reference in references
            )
            scores = await self._judge_pairs(
                task=task,
                operation="key_equivalence",
                candidates=tuple(candidate_text for _ in references),
                references=reference_texts,
                request_id=f"{request_id}:key:{index}",
                audit=audit,
            )
            matches = [
                reference for reference, score in zip(references, scores, strict=True) if score
            ]
            if len(matches) == 1:
                mapping[candidate] = matches[0]
        return mapping

    async def _judge_pairs(
        self,
        *,
        task: ResearchTask,
        operation: SemanticJudgeOperation,
        candidates: tuple[str, ...],
        references: tuple[str, ...],
        request_id: str,
        audit: _JudgeAudit,
    ) -> tuple[int, ...]:
        if self._judge is None:
            return tuple(0 for _ in candidates)
        result = await self._judge.judge(
            SemanticJudgeRequest(
                request_id=request_id,
                task_id=task.task_id,
                operation=operation,
                question=task.prompt,
                candidates=candidates,
                references=references,
            )
        )
        if result.cache_hit:
            audit.cache_hits += 1
        else:
            audit.calls += 1
        if audit.response_digests is None:
            raise AssertionError("Judge audit must initialize response digests")
        audit.response_digests.append(result.provider_response_digest)
        return result.scores

    def _result(
        self,
        task_id: TaskId,
        format_ok: bool,
        outcome_score: float,
        true_positive: float,
        predicted_items: int,
        reference_items: int,
        audit: _JudgeAudit,
    ) -> WideSeekEvaluation:
        return WideSeekEvaluation(
            task_id=task_id,
            evaluator_revision=self._revision,
            format_ok=format_ok,
            outcome_score=outcome_score,
            true_positive_items=true_positive,
            predicted_items=predicted_items,
            reference_items=reference_items,
            judge_calls=audit.calls,
            judge_cache_hits=audit.cache_hits,
            judge_response_digests=tuple(audit.response_digests or ()),
        )


def parse_boxed_answer(text: str) -> str | None:
    """Return the last balanced ``\\boxed{...}`` after the final think block."""

    value = text.split("</think>")[-1].strip()
    matches: list[str] = []
    index = 0
    while True:
        start = value.find(r"\boxed{", index)
        if start < 0:
            break
        cursor = start + len(r"\boxed{")
        depth = 1
        content_start = cursor
        while cursor < len(value) and depth:
            if value[cursor] == "{":
                depth += 1
            elif value[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            matches.append(value[content_start : cursor - 1].strip())
            index = cursor
        else:
            index = content_start
    return matches[-1] if matches and matches[-1] else None


def parse_markdown_table(text: str, *, strict: bool) -> MarkdownTable | None:
    """Parse the final fenced table, with a pipe-table fallback for references."""

    value = text.split("</think>")[-1].strip()
    fenced = re.findall(r"```markdown\s*(.*?)```", value, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced[-1]
    elif strict:
        return None
    else:
        pipe_lines = [line for line in value.splitlines() if "|" in line]
        candidate = "\n".join(pipe_lines)
    lines = [line.strip() for line in candidate.splitlines() if line.strip()]
    content_lines = [line for line in lines if "|" in line and not _is_separator_line(line)]
    if len(content_lines) < 2:
        return None
    rows = tuple(_split_markdown_row(line) for line in content_lines)
    width = len(rows[0])
    if width < 1 or any(len(row) != width for row in rows):
        return None
    columns = tuple(cell.strip() for cell in rows[0])
    if any(not column for column in columns) or len(set(columns)) != len(columns):
        return None
    try:
        return MarkdownTable(columns=columns, rows=rows[1:])
    except ValueError:
        return None


def _split_markdown_row(line: str) -> tuple[str, ...]:
    stripped = line.strip().strip("|")
    return tuple(cell.strip().replace("<br>", "\n") for cell in stripped.split("|"))


def _is_separator_line(line: str) -> bool:
    return set(line.strip()).issubset(set("|- :"))


def _normalize_column(value: str) -> str:
    return "".join(value.strip().casefold().split())


def _normalize_value(value: str) -> str:
    stripped = " ".join(value.strip().casefold().split())
    numeric = stripped.replace(",", "")
    try:
        return format(Decimal(numeric), "f").rstrip("0").rstrip(".") or "0"
    except InvalidOperation:
        return stripped


def _dedupe_rows(
    table: MarkdownTable,
    columns: dict[str, str],
    unique_columns: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    seen: set[tuple[str, ...]] = set()
    rows: list[tuple[str, ...]] = []
    for row in table.rows:
        key = _row_key(row, columns, unique_columns)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return tuple(rows)


def _row_key(
    row: tuple[str, ...],
    columns: dict[str, str],
    unique_columns: tuple[str, ...],
) -> tuple[str, ...]:
    if not unique_columns:
        return tuple(_normalize_value(value) for value in row)
    original_columns = tuple(columns.values())
    return tuple(
        _normalize_value(row[original_columns.index(columns[column])]) for column in unique_columns
    )
