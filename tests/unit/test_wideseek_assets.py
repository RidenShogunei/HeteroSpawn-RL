from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from heterospawn.assets.huggingface import (
    HF_MIRROR_ENDPOINT,
    OFFICIAL_HF_ENDPOINT,
    AssetFile,
    AssetPreparer,
    HubDownloadError,
    HuggingFaceAssetManifest,
    create_manifest,
)
from heterospawn.errors import AssetPreparationError


def _manifest(content: bytes = b"trusted") -> HuggingFaceAssetManifest:
    return create_manifest(
        asset_name="fixture",
        repo_id="owner/repo",
        repo_type="dataset",
        revision="a" * 40,
        files=(
            AssetFile(
                path="data.jsonl",
                size=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            ),
        ),
    )


class _Downloader:
    def __init__(
        self,
        content: bytes,
        *,
        official_error: HubDownloadError | None = None,
    ) -> None:
        self.content = content
        self.official_error = official_error
        self.calls: list[str] = []

    def download_file(
        self,
        manifest: HuggingFaceAssetManifest,
        file: AssetFile,
        *,
        endpoint: str,
        destination: Path,
    ) -> Path:
        del manifest
        self.calls.append(endpoint)
        if endpoint == OFFICIAL_HF_ENDPOINT and self.official_error is not None:
            raise self.official_error
        path = destination / file.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.content)
        return path


def test_auto_retries_official_three_times_then_uses_mirror(tmp_path: Path) -> None:
    downloader = _Downloader(
        b"trusted",
        official_error=HubDownloadError("timeout", retryable=True),
    )
    report = AssetPreparer(downloader).prepare(_manifest(), tmp_path)

    assert downloader.calls == [
        OFFICIAL_HF_ENDPOINT,
        OFFICIAL_HF_ENDPOINT,
        OFFICIAL_HF_ENDPOINT,
        HF_MIRROR_ENDPOINT,
    ]
    assert report.endpoint_used == "mirror"
    assert [attempt.status for attempt in report.attempts] == [
        "retryable_failure",
        "retryable_failure",
        "retryable_failure",
        "success",
    ]
    assert report.verified_files == 1


def test_authorization_error_is_terminal_and_never_switches_source(tmp_path: Path) -> None:
    downloader = _Downloader(
        b"trusted",
        official_error=HubDownloadError("http_403", retryable=False),
    )
    with pytest.raises(AssetPreparationError, match="terminal Hub error"):
        AssetPreparer(downloader).prepare(_manifest(), tmp_path)
    assert downloader.calls == [OFFICIAL_HF_ENDPOINT]


def test_digest_mismatch_is_quarantined_without_source_retry(tmp_path: Path) -> None:
    downloader = _Downloader(b"corrupt")
    with pytest.raises(AssetPreparationError, match="integrity mismatch"):
        AssetPreparer(downloader).prepare(_manifest(), tmp_path, endpoint="official")

    assert downloader.calls == [OFFICIAL_HF_ENDPOINT]
    assert not (tmp_path / "data.jsonl").exists()
    quarantined = tuple((tmp_path / ".quarantine").iterdir())
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"corrupt"


def test_existing_machine_copy_uses_same_manifest(tmp_path: Path) -> None:
    (tmp_path / "data.jsonl").write_bytes(b"trusted")
    report = AssetPreparer().verify_copy(_manifest(), tmp_path)
    assert report.endpoint_used == "existing"
    assert report.attempts == ()
    assert report.verified_bytes == len(b"trusted")


def test_untrusted_hf_endpoint_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_ENDPOINT", "https://untrusted.example")
    with pytest.raises(AssetPreparationError, match="HF_ENDPOINT"):
        AssetPreparer._endpoint_plan("auto")


@pytest.mark.parametrize("path", ("../escape", "/absolute", r"windows\separator"))
def test_manifest_rejects_unsafe_asset_paths(path: str) -> None:
    with pytest.raises(ValueError, match="normalized relative"):
        AssetFile(
            path=path,
            size=0,
            sha256=hashlib.sha256(b"").hexdigest(),
        )
