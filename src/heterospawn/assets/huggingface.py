"""Hugging Face asset download with bounded official-to-mirror fallback."""

from __future__ import annotations

import hashlib
import importlib
import os
import uuid
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from heterospawn.domain.training import canonical_digest
from heterospawn.errors import AssetPreparationError

OFFICIAL_HF_ENDPOINT = "https://huggingface.co"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
ASSET_MANIFEST_SCHEMA: Literal["heterospawn-hf-assets-v1"] = "heterospawn-hf-assets-v1"
EndpointMode = Literal["auto", "official", "mirror"]


class AssetFile(BaseModel):
    """One trusted file identity at a pinned Hub revision."""

    model_config = ConfigDict(frozen=True, strict=True)

    path: str = Field(min_length=1)
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def path_must_be_relative_and_normalized(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or "\\" in value or ".." in path.parts or value != path.as_posix():
            raise ValueError("asset path must be a normalized relative POSIX path")
        return value


class HuggingFaceAssetManifest(BaseModel):
    """Committed trust root shared by official, mirror, and copied assets."""

    model_config = ConfigDict(frozen=True, strict=True)

    schema_revision: Literal["heterospawn-hf-assets-v1"] = ASSET_MANIFEST_SCHEMA
    asset_name: str = Field(min_length=1)
    repo_id: str = Field(min_length=1)
    repo_type: Literal["model", "dataset"]
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    files: tuple[AssetFile, ...] = Field(min_length=1)
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def identity_must_be_unique_and_digest_protected(self) -> HuggingFaceAssetManifest:
        if len({file.path for file in self.files}) != len(self.files):
            raise ValueError("asset manifest contains duplicate paths")
        if tuple(sorted(file.path for file in self.files)) != tuple(
            file.path for file in self.files
        ):
            raise ValueError("asset manifest files must use sorted path order")
        if self.manifest_digest != canonical_digest(self._digest_payload()):
            raise ValueError("asset manifest digest does not match contents")
        return self

    def _digest_payload(self) -> dict[str, object]:
        return {
            "schema_revision": self.schema_revision,
            "asset_name": self.asset_name,
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "revision": self.revision,
            "files": [file.model_dump(mode="json") for file in self.files],
        }


class DownloadAttempt(BaseModel):
    """Credential-safe audit record for one endpoint attempt."""

    model_config = ConfigDict(frozen=True, strict=True)

    endpoint: Literal["official", "mirror"]
    attempt: int = Field(ge=1)
    status: Literal["success", "retryable_failure", "terminal_failure"]
    error_code: str | None = None


class AssetPreparationReport(BaseModel):
    """Safe result that omits cache paths, tokens, and signed URLs."""

    model_config = ConfigDict(frozen=True, strict=True)

    asset_name: str
    repo_id: str
    revision: str
    manifest_digest: str
    endpoint_used: Literal["official", "mirror", "existing"]
    attempts: tuple[DownloadAttempt, ...]
    verified_files: int = Field(ge=0)
    verified_bytes: int = Field(ge=0)


class HubDownloadError(Exception):
    """Normalized transport failure used to make fallback behavior testable."""

    def __init__(self, error_code: str, *, retryable: bool) -> None:
        super().__init__(error_code)
        self.error_code = error_code
        self.retryable = retryable


class HubDownloader(Protocol):
    def download_file(
        self,
        manifest: HuggingFaceAssetManifest,
        file: AssetFile,
        *,
        endpoint: str,
        destination: Path,
    ) -> Path: ...


class HuggingFaceHubDownloader:
    """Hub client with a resumable standard-resolve fallback for mirror metadata gaps."""

    def download_file(
        self,
        manifest: HuggingFaceAssetManifest,
        file: AssetFile,
        *,
        endpoint: str,
        destination: Path,
    ) -> Path:
        try:
            huggingface_hub = importlib.import_module("huggingface_hub")
            hf_hub_download = huggingface_hub.hf_hub_download

            try:
                path = hf_hub_download(
                    repo_id=manifest.repo_id,
                    repo_type=manifest.repo_type,
                    revision=manifest.revision,
                    filename=file.path,
                    local_dir=destination,
                    endpoint=endpoint,
                )
                return Path(path)
            except Exception as exc:
                if endpoint != HF_MIRROR_ENDPOINT or type(exc).__name__ not in {
                    "FileMetadataError",
                    "LocalEntryNotFoundError",
                }:
                    raise
                return self._download_resolve_url(
                    manifest,
                    file,
                    endpoint=endpoint,
                    destination=destination,
                )
        except Exception as exc:
            if isinstance(exc, HubDownloadError):
                raise
            raise self._normalize(exc) from None

    @staticmethod
    def _download_resolve_url(
        manifest: HuggingFaceAssetManifest,
        file: AssetFile,
        *,
        endpoint: str,
        destination: Path,
    ) -> Path:
        """Download through the public Hub resolve route while retaining partial bytes."""
        repo_prefix = "datasets/" if manifest.repo_type == "dataset" else ""
        encoded_path = "/".join(quote(part, safe="") for part in file.path.split("/"))
        url = (
            f"{endpoint}/{repo_prefix}{manifest.repo_id}/resolve/{manifest.revision}/{encoded_path}"
        )
        final_path = destination / file.path
        partial_path = destination / ".partials" / f"{file.path}.part"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_size = partial_path.stat().st_size if partial_path.is_file() else 0
        request_headers = {"Range": f"bytes={partial_size}-"} if partial_size else {}

        try:
            with (
                httpx.Client(
                    follow_redirects=True,
                    timeout=httpx.Timeout(60.0, connect=15.0),
                ) as client,
                client.stream("GET", url, headers=request_headers) as response,
            ):
                if response.status_code in (401, 403):
                    raise HubDownloadError(
                        f"http_{response.status_code}",
                        retryable=False,
                    )
                if response.status_code == 404:
                    raise HubDownloadError("revision_or_file_not_found", retryable=False)
                if 500 <= response.status_code <= 599:
                    raise HubDownloadError(
                        f"http_{response.status_code}",
                        retryable=True,
                    )
                if response.status_code == 416 and partial_size == file.size:
                    partial_path.replace(final_path)
                    return final_path
                if response.status_code not in (200, 206):
                    raise HubDownloadError(
                        f"http_{response.status_code}",
                        retryable=False,
                    )

                revisions = {
                    candidate.headers.get("x-repo-commit")
                    for candidate in (*response.history, response)
                    if candidate.headers.get("x-repo-commit") is not None
                }
                if revisions and revisions != {manifest.revision}:
                    raise HubDownloadError("revision_mismatch", retryable=False)

                append = response.status_code == 206 and partial_size > 0
                if append:
                    content_range = response.headers.get("content-range", "")
                    if not content_range.startswith(f"bytes {partial_size}-"):
                        raise HubDownloadError("invalid_content_range", retryable=False)
                with partial_path.open("ab" if append else "wb") as handle:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        handle.write(chunk)
        except HubDownloadError:
            raise
        except (httpx.TimeoutException, httpx.NetworkError):
            raise HubDownloadError("mirror_transport_error", retryable=True) from None

        actual_size = partial_path.stat().st_size
        if actual_size != file.size:
            raise HubDownloadError(
                "incomplete_download" if actual_size < file.size else "oversized_download",
                retryable=actual_size < file.size,
            )
        partial_path.replace(final_path)
        return final_path

    @staticmethod
    def _normalize(exc: Exception) -> HubDownloadError:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        name = type(exc).__name__
        if status_code in (401, 403):
            return HubDownloadError(f"http_{status_code}", retryable=False)
        if status_code is not None and 500 <= status_code <= 599:
            return HubDownloadError(f"http_{status_code}", retryable=True)
        if name in {
            "RevisionNotFoundError",
            "EntryNotFoundError",
            "RepositoryNotFoundError",
        }:
            return HubDownloadError(name, retryable=False)
        retryable = isinstance(exc, (ConnectionError, TimeoutError)) or name in {
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
            "ProxyError",
            "LocalEntryNotFoundError",
            "FileMetadataError",
        }
        return HubDownloadError(name, retryable=retryable)


class AssetPreparer:
    """Prepare every manifest file, then verify all bytes before publication."""

    def __init__(
        self,
        downloader: HubDownloader | None = None,
        *,
        attempts_per_endpoint: int = 3,
    ) -> None:
        if attempts_per_endpoint < 1:
            raise ValueError("attempts_per_endpoint must be positive")
        self._downloader = downloader or HuggingFaceHubDownloader()
        self._attempts_per_endpoint = attempts_per_endpoint

    def prepare(
        self,
        manifest: HuggingFaceAssetManifest,
        destination: Path,
        *,
        endpoint: EndpointMode = "auto",
    ) -> AssetPreparationReport:
        destination = destination.resolve()
        destination.mkdir(parents=True, exist_ok=True)
        existing = self._verify_existing(manifest, destination)
        if existing is not None:
            return existing

        endpoint_plan = self._endpoint_plan(endpoint)
        attempts: list[DownloadAttempt] = []
        used_endpoint: Literal["official", "mirror"] | None = None
        for endpoint_name, endpoint_url in endpoint_plan:
            for attempt in range(1, self._attempts_per_endpoint + 1):
                try:
                    for file in manifest.files:
                        self._downloader.download_file(
                            manifest,
                            file,
                            endpoint=endpoint_url,
                            destination=destination,
                        )
                    attempts.append(
                        DownloadAttempt(
                            endpoint=endpoint_name,
                            attempt=attempt,
                            status="success",
                        )
                    )
                    used_endpoint = endpoint_name
                    break
                except HubDownloadError as exc:
                    attempts.append(
                        DownloadAttempt(
                            endpoint=endpoint_name,
                            attempt=attempt,
                            status=("retryable_failure" if exc.retryable else "terminal_failure"),
                            error_code=exc.error_code,
                        )
                    )
                    if not exc.retryable:
                        raise AssetPreparationError(
                            f"terminal Hub error for {manifest.asset_name}: {exc.error_code}"
                        ) from None
            if used_endpoint is not None:
                break
            if endpoint != "auto":
                break

        if used_endpoint is None:
            raise AssetPreparationError(
                f"asset download exhausted configured endpoints: {manifest.asset_name}"
            )
        verified_files, verified_bytes = self._verify_or_quarantine(manifest, destination)
        return AssetPreparationReport(
            asset_name=manifest.asset_name,
            repo_id=manifest.repo_id,
            revision=manifest.revision,
            manifest_digest=manifest.manifest_digest,
            endpoint_used=used_endpoint,
            attempts=tuple(attempts),
            verified_files=verified_files,
            verified_bytes=verified_bytes,
        )

    def verify_copy(
        self,
        manifest: HuggingFaceAssetManifest,
        destination: Path,
    ) -> AssetPreparationReport:
        destination = destination.resolve()
        verified_files, verified_bytes = self._verify_or_quarantine(manifest, destination)
        return AssetPreparationReport(
            asset_name=manifest.asset_name,
            repo_id=manifest.repo_id,
            revision=manifest.revision,
            manifest_digest=manifest.manifest_digest,
            endpoint_used="existing",
            attempts=(),
            verified_files=verified_files,
            verified_bytes=verified_bytes,
        )

    def _verify_existing(
        self,
        manifest: HuggingFaceAssetManifest,
        destination: Path,
    ) -> AssetPreparationReport | None:
        if not all((destination / file.path).is_file() for file in manifest.files):
            return None
        return self.verify_copy(manifest, destination)

    @staticmethod
    def _endpoint_plan(
        endpoint: EndpointMode,
    ) -> tuple[tuple[Literal["official", "mirror"], str], ...]:
        if endpoint == "official":
            return (("official", OFFICIAL_HF_ENDPOINT),)
        if endpoint == "mirror":
            return (("mirror", HF_MIRROR_ENDPOINT),)
        environment_endpoint = os.environ.get("HF_ENDPOINT", "").rstrip("/")
        if environment_endpoint == HF_MIRROR_ENDPOINT:
            return (("mirror", HF_MIRROR_ENDPOINT),)
        if environment_endpoint and environment_endpoint != OFFICIAL_HF_ENDPOINT:
            raise AssetPreparationError(
                "HF_ENDPOINT must be the official endpoint or the configured mirror"
            )
        return (
            ("official", OFFICIAL_HF_ENDPOINT),
            ("mirror", HF_MIRROR_ENDPOINT),
        )

    @staticmethod
    def _verify_or_quarantine(
        manifest: HuggingFaceAssetManifest,
        destination: Path,
    ) -> tuple[int, int]:
        total_bytes = 0
        for expected in manifest.files:
            path = (destination / expected.path).resolve()
            if destination not in path.parents:
                raise AssetPreparationError("asset path escapes destination")
            if not path.is_file():
                raise AssetPreparationError(f"asset file is missing: {expected.path}")
            size, digest = _file_identity(path)
            if size != expected.size or digest != expected.sha256:
                quarantine = destination / ".quarantine"
                quarantine.mkdir(parents=True, exist_ok=True)
                quarantined = quarantine / (f"{path.name}.{digest[:12]}.{uuid.uuid4().hex}.corrupt")
                path.replace(quarantined)
                raise AssetPreparationError(
                    f"asset integrity mismatch quarantined: {expected.path}"
                )
            total_bytes += size
        return len(manifest.files), total_bytes


def load_asset_manifest(path: Path) -> HuggingFaceAssetManifest:
    try:
        return HuggingFaceAssetManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AssetPreparationError(f"invalid asset manifest: {type(exc).__name__}") from None


def _file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def create_manifest(
    *,
    asset_name: str,
    repo_id: str,
    repo_type: Literal["model", "dataset"],
    revision: str,
    files: tuple[AssetFile, ...],
) -> HuggingFaceAssetManifest:
    ordered = tuple(sorted(files, key=lambda file: file.path))
    payload = {
        "schema_revision": ASSET_MANIFEST_SCHEMA,
        "asset_name": asset_name,
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision,
        "files": [file.model_dump(mode="json") for file in ordered],
    }
    return HuggingFaceAssetManifest(
        schema_revision=ASSET_MANIFEST_SCHEMA,
        asset_name=asset_name,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        files=ordered,
        manifest_digest=canonical_digest(payload),
    )
