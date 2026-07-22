from __future__ import annotations

import base64
import csv
from collections.abc import Callable
from pathlib import Path

import pytest

XBenchFixtureFactory = Callable[[tuple[tuple[str, str, str], ...]], Path]


def _encrypt(value: str, key: str) -> str:
    key_bytes = key.encode()
    encrypted = bytes(
        byte ^ key_bytes[index % len(key_bytes)] for index, byte in enumerate(value.encode())
    )
    return base64.b64encode(encrypted).decode()


@pytest.fixture
def xbench_fixture_factory(tmp_path: Path) -> XBenchFixtureFactory:
    """Create encrypted synthetic xbench files without duplicating crypto helpers."""

    def write(records: tuple[tuple[str, str, str], ...]) -> Path:
        path = tmp_path / "synthetic.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "prompt", "answer", "canary"])
            writer.writeheader()
            for task_id, prompt, answer in records:
                key = f"synthetic-canary-{task_id}"
                writer.writerow(
                    {
                        "id": task_id,
                        "prompt": _encrypt(prompt, key),
                        "answer": _encrypt(answer, key),
                        "canary": key,
                    }
                )
        return path

    return write
