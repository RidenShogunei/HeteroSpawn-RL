#!/usr/bin/env python3
"""Collect a credential-safe, read-only inventory for remote backend spikes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(command: list[str], *, timeout: int = 15) -> tuple[str, str]:
    if shutil.which(command[0]) is None:
        return "unavailable", ""
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "error", ""
    if completed.returncode != 0:
        return "error", ""
    return "ok", completed.stdout.strip()


def _version(command: list[str]) -> dict[str, str]:
    status, output = _run(command)
    return {"status": status, "version": output.splitlines()[0] if output else ""}


def _gpu_inventory() -> tuple[list[dict[str, Any]], str]:
    query = (
        "--query-gpu=index,name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    )
    status, output = _run(["nvidia-smi", *query])
    include_compute_capability = status == "ok"
    if status != "ok":
        status, output = _run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
    if status != "ok":
        return [], status

    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        minimum_fields = 5 if include_compute_capability else 4
        if len(fields) != minimum_fields:
            continue
        item: dict[str, Any] = {
            "index": int(fields[0]),
            "name": fields[1],
            "memory_total_mib": int(fields[2]),
            "driver_version": fields[3],
        }
        if include_compute_capability:
            item["compute_capability"] = fields[4]
        gpus.append(item)
    return gpus, "ok"


def _driver_cuda_version() -> str:
    status, output = _run(["nvidia-smi"])
    if status != "ok":
        return ""
    match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
    return match.group(1) if match else ""


def _nvcc_version() -> dict[str, str]:
    status, output = _run(["nvcc", "--version"])
    match = re.search(r"release\s+([0-9.]+)", output)
    return {"status": status, "version": match.group(1) if match else ""}


def _torch_inventory() -> dict[str, Any]:
    probe = (
        "import json, torch; "
        "print(json.dumps({'version': torch.__version__, "
        "'cuda_build': torch.version.cuda or '', "
        "'cuda_available': torch.cuda.is_available(), "
        "'device_count': torch.cuda.device_count()}))"
    )
    status, output = _run([sys.executable, "-c", probe], timeout=30)
    if status != "ok":
        return {"status": status}
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return {"status": "error"}
    return {"status": "ok", **parsed}


def build_report() -> dict[str, Any]:
    gpus, gpu_status = _gpu_inventory()
    disk = shutil.disk_usage(Path.home())
    report: dict[str, Any] = {
        "schema_version": 1,
        # The standalone bootstrap must run on the remote host's Python 3.10.
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "tools": {
            "git": _version(["git", "--version"]),
            "docker": _version(["docker", "--version"]),
            "uv": _version(["uv", "--version"]),
            "nvcc": _nvcc_version(),
        },
        "nvidia": {
            "status": gpu_status,
            "driver_cuda_version": _driver_cuda_version(),
            "gpu_count": len(gpus),
            "gpus": gpus,
        },
        "torch": _torch_inventory(),
        "home_filesystem": {
            "total_bytes": disk.total,
            "free_bytes": disk.free,
        },
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_digest"] = hashlib.sha256(canonical).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-gpu-count", type=int)
    args = parser.parse_args()

    report = build_report()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        print(json.dumps(report, indent=2, sort_keys=True))

    print(
        json.dumps(
            {
                "gpu_count": report["nvidia"]["gpu_count"],
                "report_digest": report["report_digest"],
            },
            sort_keys=True,
        )
    )
    if (
        args.require_gpu_count is not None
        and report["nvidia"]["gpu_count"] != args.require_gpu_count
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
