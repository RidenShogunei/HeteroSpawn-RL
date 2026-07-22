from __future__ import annotations

import base64
import csv
from pathlib import Path

import pytest

from heterospawn.benchmarks.xbench import BenchmarkTask, load_xbench, parse_final_answer
from heterospawn.domain.ids import TaskId
from heterospawn.errors import BenchmarkDataError


def _encrypt(value: str, key: str) -> str:
    key_bytes = key.encode()
    encrypted = bytes(
        byte ^ key_bytes[index % len(key_bytes)] for index, byte in enumerate(value.encode())
    )
    return base64.b64encode(encrypted).decode()


def _write_synthetic_fixture(path: Path) -> None:
    key = "synthetic-canary"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "prompt", "answer", "canary"])
        writer.writeheader()
        writer.writerow(
            {
                "id": "synthetic-1",
                "prompt": _encrypt("policy-visible question", key),
                "answer": _encrypt("42", key),
                "canary": key,
            }
        )


def test_encrypted_fixture_is_decrypted_in_memory_and_answer_stays_evaluator_only(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "synthetic.csv"
    _write_synthetic_fixture(fixture)

    dataset = load_xbench(fixture, verify_official_digest=False)

    assert dataset.tasks[0].prompt == "policy-visible question"
    assert "answer" not in BenchmarkTask.model_fields
    score = dataset.evaluate_exact({TaskId("synthetic-1"): "reasoning\n最终答案:42"})
    assert score.correct == 1
    assert score.accuracy == 1.0
    assert score.comparable_to_official is False


def test_official_digest_is_required_by_default(tmp_path: Path) -> None:
    fixture = tmp_path / "synthetic.csv"
    _write_synthetic_fixture(fixture)

    with pytest.raises(BenchmarkDataError, match="digest"):
        load_xbench(fixture)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("最终答案:alpha", "alpha"),
        ("text\n最终答案::: beta", ""),
        ("no marker", None),
    ],
)
def test_final_answer_parser_matches_pinned_deterministic_shortcut(
    response: str, expected: str | None
) -> None:
    assert parse_final_answer(response) == expected
