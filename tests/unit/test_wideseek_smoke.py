from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from heterospawn.backends.local_hf.config import LocalLoraConfig
from heterospawn.cli import build_parser
from heterospawn.domain.ids import PolicyId
from heterospawn.domain.training import canonical_digest
from heterospawn.domain.versions import RolloutRevision
from heterospawn.search.wideseek_local import (
    WideSeekLocalConfig,
    WideSeekLocalToolService,
)
from heterospawn.training.mock import MockTrainingBackend
from heterospawn.training.wideseek_smoke import (
    _ScriptedPolicyService,
    _Utf8ToolCodec,
    run_wideseek_compliance_baseline,
    run_wideseek_rollout_smoke,
)


class _ComplianceBackend:
    def __init__(self, policy_id: PolicyId, url: str) -> None:
        self._mock = MockTrainingBackend((policy_id,))
        self._policy_id = policy_id
        self.prompt_encoder = _Utf8ToolCodec()
        self._service = _ScriptedPolicyService(
            self._mock,
            policy_id,
            discovered_url=url,
        )

    def endpoint(self, policy_id: PolicyId) -> _ScriptedPolicyService:
        assert policy_id == self._policy_id
        return self._service

    def rollout_revision(self, policy_id: PolicyId) -> RolloutRevision:
        return self._mock.rollout_revision(policy_id)

    def adapter_hash(self, policy_id: PolicyId, *, rollout: bool = False) -> str:
        assert policy_id == self._policy_id
        return "fixed-rollout" if rollout else "fixed-train"


@pytest.mark.asyncio
async def test_rollout_smoke_forces_real_shape_search_then_access_without_plaintext_report(
    tmp_path: Path,
) -> None:
    url = "https://en.wikipedia.org/wiki/Red_Bull"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/retrieve":
            return httpx.Response(
                200,
                json={
                    "result": [
                        [
                            {
                                "document": {
                                    "url": url,
                                    "contents": "retrieved private snippet",
                                },
                                "score": 1.0,
                            }
                        ]
                    ]
                },
            )
        if request.url.path == "/access":
            assert json.loads(request.content) == {"urls": [url]}
            return httpx.Response(
                200,
                json={
                    "result": [
                        {
                            "url": url,
                            "contents": "accessed private full page",
                        }
                    ]
                },
            )
        return httpx.Response(404)

    report_path = tmp_path / "report.json"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tools = WideSeekLocalToolService(WideSeekLocalConfig(), client=client)
        report = await run_wideseek_rollout_smoke(
            service_url="http://unused.invalid",
            qdrant_url="http://unused.invalid",
            report_path=report_path,
            tool_service=tools,
        )

    assert report["status"] == "passed"
    assert report["environment_mode"] == "controlled-fixture"
    assert report["tool_sequence"] == ["search", "access"]
    assert report["access_has_search_provenance"] is True
    persisted = report_path.read_text(encoding="utf-8")
    assert "private" not in persisted
    assert "wikipedia" not in persisted
    assert "Red Bull" not in persisted


@pytest.mark.asyncio
async def test_compliance_baseline_is_rollout_only_and_reference_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://en.wikipedia.org/wiki/Red_Bull"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/retrieve":
            return httpx.Response(
                200,
                json={
                    "result": [
                        [
                            {
                                "document": {
                                    "url": url,
                                    "contents": "PRIVATE_RETRIEVAL_SENTINEL",
                                },
                                "score": 1.0,
                            }
                        ]
                    ]
                },
            )
        if request.url.path == "/access":
            return httpx.Response(
                200,
                json={
                    "result": [
                        {
                            "url": url,
                            "contents": "PRIVATE_ACCESS_SENTINEL",
                        }
                    ]
                },
            )
        return httpx.Response(404)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    record = {
        "question": "PRIVATE_QUESTION_SENTINEL",
        "answer": (
            "```markdown\n| Name | Value |\n|---|---|\n| PRIVATE_REFERENCE_SENTINEL | 1 |\n```"
        ),
        "unique_columns": ["Name"],
    }
    content = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    data_path = data_dir / "width_20k.jsonl"
    data_path.write_bytes(content.encode())
    file_digest = hashlib.sha256(content.encode()).hexdigest()
    manifest_payload = {
        "schema_revision": "heterospawn-hf-assets-v1",
        "asset_name": "wideseek-train-data",
        "repo_id": "RLinf/WideSeek-R1-train-data",
        "repo_type": "dataset",
        "revision": "47832ea20581f78d32cd6b32b4b37b985cbbc9df",
        "files": [
            {
                "path": "width_20k.jsonl",
                "size": len(content.encode()),
                "sha256": file_digest,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                **manifest_payload,
                "manifest_digest": canonical_digest(manifest_payload),
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    policy_id = PolicyId("shared")
    monkeypatch.setattr(
        "heterospawn.training.wideseek_smoke.importlib.import_module",
        lambda name: pytest.fail(f"CPU compliance baseline imported optional module: {name}"),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tools = WideSeekLocalToolService(WideSeekLocalConfig(), client=client)
        report = await run_wideseek_compliance_baseline(
            topology="shared",
            task_selection=(("width_20k", 0),),
            rollouts_per_task=1,
            data_manifest_path=manifest_path,
            data_dir=data_dir,
            service_url="http://unused.invalid",
            qdrant_url="http://unused.invalid",
            local_config=LocalLoraConfig(
                device="cpu",
                dtype="float32",
                artifact_dir=tmp_path / "checkpoints",
            ),
            report_path=report_path,
            tool_service=tools,
            backend=_ComplianceBackend(policy_id, url),
        )

    assert report["optimizer_updates"] == 0
    assert report["summary"]["episodes"] == 1
    assert report["summary"]["success_rate"] == 1.0
    assert report["summary"]["spawn_rate"] == 1.0
    assert report["summary"]["format_ok_rate"] == 0.0
    assert report["summary"]["nonzero_outcome_rate"] == 0.0
    assert report["summary"]["tool_calls"] == 2
    assert report["summary"]["prompt_token_counts"]["count"] == report["summary"]["model_steps"]
    assert report["summary"]["response_token_counts"]["count"] == report["summary"]["model_steps"]
    assert report["summary"]["sequence_token_counts"]["max"] > 0
    assert report["max_sequence_length"] == 1024
    assert all(report["checks"].values())
    persisted = report_path.read_text(encoding="utf-8")
    assert "PRIVATE_" not in persisted
    assert url not in persisted


def test_compliance_cli_parses_explicit_split_indices() -> None:
    args = build_parser().parse_args(
        [
            "wideseek-compliance-baseline",
            "--task",
            "width_20k:7",
            "--task",
            "depth_20k:9",
            "--model-path",
            "model",
        ]
    )
    assert args.tasks == [("width_20k", 7), ("depth_20k", 9)]

    with pytest.raises(SystemExit):
        build_parser().parse_args(["wideseek-compliance-baseline", "--task", "unknown:0"])


@pytest.mark.asyncio
async def test_compliance_rejects_incomplete_independent_checkpoint_pair(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="both Main and Sub"):
        await run_wideseek_compliance_baseline(
            topology="independent",
            task_selection=(("width_20k", 0),),
            rollouts_per_task=1,
            data_manifest_path=tmp_path / "missing.json",
            data_dir=tmp_path,
            service_url="http://unused.invalid",
            qdrant_url="http://unused.invalid",
            local_config=LocalLoraConfig(
                device="cpu",
                dtype="float32",
                artifact_dir=tmp_path / "checkpoints",
            ),
            report_path=tmp_path / "report.json",
            main_checkpoint_dir=tmp_path / "main",
        )


def test_independent_train_accepts_explicit_shared_checkpoint_fork() -> None:
    args = build_parser().parse_args(
        [
            "wideseek-train-smoke",
            "--topology",
            "independent",
            "--checkpoint-dir",
            "shared-checkpoint",
            "--model-path",
            "model",
        ]
    )

    assert args.topology == "independent"
    assert args.checkpoint_dir == Path("shared-checkpoint")
