from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from heterospawn.benchmarks.wideseek import WideSeekDataset, load_wideseek_dataset
from heterospawn.errors import JudgeRequestError
from heterospawn.evaluation.semantic_judge import (
    SemanticJudgeRequest,
    SemanticJudgeResult,
    SemanticJudgeRevision,
)
from heterospawn.evaluation.wideseek import (
    WideSeekEvaluator,
    parse_boxed_answer,
    parse_markdown_table,
)
from heterospawn.policies.base import TokenUsage


def _dataset(tmp_path: Path, record: dict[str, object], split: str) -> WideSeekDataset:
    content = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    path = tmp_path / f"{split}.jsonl"
    path.write_bytes(content.encode())
    return load_wideseek_dataset(
        path,
        split=split,  # type: ignore[arg-type]
        expected_sha256=hashlib.sha256(content.encode()).hexdigest(),
    )


class _FakeJudge:
    revision = SemanticJudgeRevision(
        mode="fake",
        provider="fake",
        model="semantic-fixture",
        prompt_revision="fixture-v1",
        sampling_params=(("temperature", 0.0),),
        max_format_attempts=1,
    )

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list[SemanticJudgeRequest] = []

    async def judge(self, request: SemanticJudgeRequest) -> SemanticJudgeResult:
        if self.fail:
            raise JudgeRequestError("synthetic terminal failure")
        self.requests.append(request)
        equivalent = {
            ("entity", "name"),
            ('["a"]', '["alpha"]'),
            ("nyc", "new york city"),
            ("one", "1"),
        }
        scores = tuple(
            1
            if candidate.strip().casefold() == reference.strip().casefold()
            or (candidate.strip().casefold(), reference.strip().casefold()) in equivalent
            else 0
            for candidate, reference in zip(request.candidates, request.references, strict=True)
        )
        return SemanticJudgeResult(
            request_id=request.request_id,
            scores=scores,  # type: ignore[arg-type]
            cache_key=hashlib.sha256(request.request_id.encode()).hexdigest(),
            cache_hit=False,
            provider_response_digest=f"digest-{len(self.requests)}",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


def test_boxed_and_markdown_parsers_cover_balanced_and_strict_formats() -> None:
    assert parse_boxed_answer(r"reasoning \boxed{alpha_{nested}}") == "alpha_{nested}"
    assert parse_boxed_answer(r"\boxed{broken") is None
    assert parse_markdown_table("| A |\n|---|\n| 1 |", strict=True) is None
    table = parse_markdown_table(
        "```markdown\n| A | B |\n|---|---|\n| 1 | x |\n```",
        strict=True,
    )
    assert table is not None
    assert table.columns == ("A", "B")
    assert table.rows == (("1", "x"),)


@pytest.mark.asyncio
async def test_markdown_required_columns_key_alignment_and_item_f1(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        {
            "question": "table",
            "answer": (
                "```markdown\n| Name | City |\n|---|---|\n"
                "| Alpha | New York City |\n| Beta | Paris |\n```"
            ),
            "unique_columns": ["Name"],
        },
        "width_20k",
    )
    judge = _FakeJudge()
    evaluator = WideSeekEvaluator(dataset, judge)
    result = await evaluator.evaluate(
        dataset.tasks[0],
        ("```markdown\n| Entity | City |\n|---|---|\n| A | NYC |\n| Beta | Paris |\n```"),
        request_id="markdown",
    )

    assert result.format_ok is True
    assert result.outcome_score == pytest.approx(1.0)
    assert result.true_positive_items == 4
    assert result.predicted_items == result.reference_items == 4
    assert result.judge_calls == 3
    assert {request.operation for request in judge.requests} == {
        "column_equivalence",
        "key_equivalence",
        "cell_equivalence",
    }


@pytest.mark.asyncio
async def test_missing_required_column_is_format_failure_without_judge(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        {
            "question": "table",
            "answer": "```markdown\n| Name | City |\n|---|---|\n| A | Paris |\n```",
            "unique_columns": ["Name"],
        },
        "width_20k",
    )
    result = await WideSeekEvaluator(dataset).evaluate(
        dataset.tasks[0],
        "```markdown\n| Name |\n|---|\n| A |\n```",
        request_id="missing",
    )
    assert result.format_ok is False
    assert result.outcome_score == 0.0


@pytest.mark.asyncio
async def test_boxed_semantic_judge_and_failure_propagation(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        {
            "question": "fact",
            "answer": "1",
            "source": "fixture",
            "aug_answer": [],
            "qid": 1,
        },
        "depth_20k",
    )
    result = await WideSeekEvaluator(dataset, _FakeJudge()).evaluate(
        dataset.tasks[0],
        r"\boxed{one}",
        request_id="boxed",
    )
    assert result.format_ok is True
    assert result.outcome_score == 1.0

    with pytest.raises(JudgeRequestError, match="terminal"):
        await WideSeekEvaluator(dataset, _FakeJudge(fail=True)).evaluate(
            dataset.tasks[0],
            r"\boxed{different}",
            request_id="failure",
        )
