from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_remote_preflight_writes_redacted_report(tmp_path: Path) -> None:
    repository_root = Path(__file__).parents[2]
    report_path = tmp_path / "preflight.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "remote_preflight.py"),
            "--output",
            str(report_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = json.loads(completed.stdout.strip())
    assert report["schema_version"] == 1
    assert report["report_digest"] == summary["report_digest"]
    assert report["nvidia"]["gpu_count"] == len(report["nvidia"]["gpus"])
    serialized = json.dumps(report).lower()
    assert "hostname" not in serialized
    assert "username" not in serialized
    assert "api_key" not in serialized
    assert "identityfile" not in serialized
