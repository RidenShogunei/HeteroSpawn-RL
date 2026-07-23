from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from heterospawn.benchmarks.wideseek import load_wideseek_dataset
from heterospawn.errors import BenchmarkDataError


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> str:
    content = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    )
    path.write_bytes(content.encode())
    return hashlib.sha256(content.encode()).hexdigest()


@pytest.mark.parametrize(
    ("split", "record", "answer_format"),
    [
        (
            "width_20k",
            {
                "question": "make a table",
                "answer": "```markdown\n| Name | Value |\n|---|---|\n| A | 1 |\n```",
                "unique_columns": ["Name"],
            },
            "markdown_table",
        ),
        (
            "depth_20k",
            {
                "question": "answer a fact",
                "answer": "alpha",
                "source": "fixture",
                "aug_answer": ["alpha"],
                "qid": 7,
            },
            "boxed",
        ),
        (
            "hybrid_20k",
            {
                "question": "hybrid table",
                "answer": "```markdown\n| Name |\n|---|\n| A |\n```",
                "unique_columns": ["Name"],
                "is_markdown": True,
                "instance_id": 11,
            },
            "markdown_table",
        ),
    ],
)
def test_loader_validates_each_split_and_keeps_answer_private(
    tmp_path: Path,
    split: str,
    record: dict[str, object],
    answer_format: str,
) -> None:
    path = tmp_path / f"{split}.jsonl"
    digest = _write_jsonl(path, [record])
    dataset = load_wideseek_dataset(
        path,
        split=split,  # type: ignore[arg-type]
        expected_sha256=digest,
    )

    assert len(dataset.tasks) == 1
    task = dataset.tasks[0]
    assert task.prompt == record["question"]
    assert task.answer_format == answer_format
    assert "answer" not in task.model_dump(mode="json")
    assert record["answer"] not in task.model_dump_json()
    assert dataset.evaluator_record(task).answers


def test_hybrid_plain_record_uses_boxed_semantics(tmp_path: Path) -> None:
    path = tmp_path / "hybrid_20k.jsonl"
    digest = _write_jsonl(
        path,
        [
            {
                "question": "fact",
                "answer": "alpha",
                "unique_columns": None,
                "is_markdown": False,
                "instance_id": 3,
            }
        ],
    )
    dataset = load_wideseek_dataset(
        path,
        split="hybrid_20k",
        expected_sha256=digest,
    )
    assert dataset.tasks[0].answer_format == "boxed"


def test_digest_duplicate_id_and_schema_errors_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "depth_20k.jsonl"
    records = [
        {
            "question": "q1",
            "answer": "a1",
            "source": "fixture",
            "aug_answer": [],
            "qid": 1,
        },
        {
            "question": "q2",
            "answer": "a2",
            "source": "fixture",
            "aug_answer": [],
            "qid": 1,
        },
    ]
    digest = _write_jsonl(path, records)
    with pytest.raises(BenchmarkDataError, match="digest"):
        load_wideseek_dataset(path, split="depth_20k", expected_sha256="0" * 64)
    with pytest.raises(BenchmarkDataError, match="duplicate"):
        load_wideseek_dataset(path, split="depth_20k", expected_sha256=digest)

    bad_path = tmp_path / "width_20k.jsonl"
    bad_digest = _write_jsonl(
        bad_path,
        [{"question": "q", "answer": "a", "unique_columns": [], "secret": "no"}],
    )
    with pytest.raises(BenchmarkDataError, match="unsupported fields"):
        load_wideseek_dataset(
            bad_path,
            split="width_20k",
            expected_sha256=bad_digest,
        )


def test_summary_is_reference_free(tmp_path: Path) -> None:
    path = tmp_path / "width_20k.jsonl"
    reference = "PRIVATE_REFERENCE_SENTINEL"
    digest = _write_jsonl(
        path,
        [
            {
                "question": "q",
                "answer": f"```markdown\n| Key |\n|---|\n| {reference} |\n```",
                "unique_columns": ["Key"],
            }
        ],
    )
    summary = load_wideseek_dataset(
        path,
        split="width_20k",
        expected_sha256=digest,
    ).summary()
    assert reference not in summary.model_dump_json()
