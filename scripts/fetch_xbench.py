"""Fetch the pinned encrypted xbench dataset and verify its digest."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path

REVISION = "17c562192cc7e62215bfb98b65e9f8806fb95504"
EXPECTED_SHA256 = "a9378e56b05ec8f007b8ecc8f6ac74900abafd558267acd5839d0d05fbc6977a"
URL = (
    f"https://raw.githubusercontent.com/xbench-ai/xbench-evals/{REVISION}/data/DeepSearch-2510.csv"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/private/xbench/DeepSearch-2510.csv"),
    )
    args = parser.parse_args()

    with urllib.request.urlopen(URL, timeout=30) as response:
        encrypted = response.read()
    digest = hashlib.sha256(encrypted).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError("downloaded encrypted dataset failed digest verification")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as temporary:
        temporary.write(encrypted)
        temporary_path = Path(temporary.name)
    temporary_path.replace(args.output)
    print(f"stored verified encrypted dataset at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
