from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from heterospawn.cli import build_parser
from heterospawn.experiments.wideseek import (
    CheckpointState,
    StageRecord,
    StageRequest,
    WideSeekExperimentRunner,
    WideSeekExperimentRuntime,
    _stage_input_digest,
    _training_checkpoint_states,
    load_wideseek_experiment_config,
)


class _RecordingExecutor:
    def __init__(self, config_digest: str, *, fail_once_at: str | None = None) -> None:
        self.config_digest = config_digest
        self.fail_once_at = fail_once_at
        self.calls: list[str] = []

    async def execute(self, request: StageRequest) -> StageRecord:
        self.calls.append(request.name)
        if self.fail_once_at == request.name:
            self.fail_once_at = None
            raise RuntimeError("injected stage failure")
        report_path = (
            request.dependencies[0].report_path.parent / f"{request.name}.json"
            if (request.dependencies)
            else Path.cwd() / f"{request.name}.json"
        )
        if request.name == "sft":
            report_path = Path.cwd() / "sft.json"
        report_path = report_path.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"schema_revision": "test", "status": "passed"}),
            encoding="utf-8",
        )
        report_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
        return StageRecord(
            name=request.name,
            kind=request.kind,
            input_digest=_stage_input_digest(self.config_digest, request),
            report_path=report_path,
            report_digest=report_digest,
        )


def _runtime(tmp_path: Path) -> WideSeekExperimentRuntime:
    return WideSeekExperimentRuntime(
        run_dir=tmp_path / "run",
        model_path=tmp_path / "model",
        model_manifest_path=tmp_path / "model-manifest.json",
        data_manifest_path=tmp_path / "data-manifest.json",
        data_dir=tmp_path / "data",
    )


@pytest.mark.asyncio
async def test_unified_runner_resumes_at_first_uncommitted_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = Path(__file__).parents[2] / "configs" / "wideseek-qwen3-4b-2080ti.json"
    config = load_wideseek_experiment_config(config_path)
    runtime = _runtime(tmp_path)
    first_runner = WideSeekExperimentRunner(config, runtime)
    first_executor = _RecordingExecutor(
        first_runner.config_digest,
        fail_once_at="independent-cycle-000",
    )
    first_runner.executor = first_executor

    with pytest.raises(RuntimeError, match="injected"):
        await first_runner.run(resume=False)
    assert first_executor.calls == [
        "sft",
        "shared-cycle-000",
        "independent-cycle-000",
    ]

    resumed_runner = WideSeekExperimentRunner(config, runtime)
    resumed_executor = _RecordingExecutor(resumed_runner.config_digest)
    resumed_runner.executor = resumed_executor
    report = await resumed_runner.run(resume=True)

    assert resumed_executor.calls == [
        "independent-cycle-000",
        "eval-sft",
        "eval-shared",
        "eval-independent",
        "comparison",
    ]
    assert report["status"] == "passed"
    assert report["completed_stages"] == [
        "sft",
        "shared-cycle-000",
        "independent-cycle-000",
        "eval-sft",
        "eval-shared",
        "eval-independent",
        "comparison",
    ]


@pytest.mark.asyncio
async def test_resume_rejects_configuration_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = Path(__file__).parents[2] / "configs" / "wideseek-qwen3-4b-2080ti.json"
    config = load_wideseek_experiment_config(config_path)
    runtime = _runtime(tmp_path)
    runner = WideSeekExperimentRunner(config, runtime)
    executor = _RecordingExecutor(runner.config_digest)
    runner.executor = executor
    await runner.run(resume=False)

    changed = config.model_copy(update={"max_new_tokens": 512})
    changed_runner = WideSeekExperimentRunner(changed, runtime)
    changed_runner.executor = _RecordingExecutor(changed_runner.config_digest)
    with pytest.raises(ValueError, match="configuration differs"):
        await changed_runner.run(resume=True)


def test_canonical_config_and_cli_are_loadable() -> None:
    config_path = Path(__file__).parents[2] / "configs" / "wideseek-qwen3-4b-2080ti.json"
    config = load_wideseek_experiment_config(config_path)

    assert config.model_profile == "qwen3-4b"
    assert config.max_sequence_length == 4096
    assert config.max_new_tokens == 1024
    assert len(config.evaluation.tasks) == 16
    args = build_parser().parse_args(
        [
            "wideseek-run",
            "--run-dir",
            "runtime/run",
            "--model-path",
            "runtime/model",
            "--data-dir",
            "runtime/data",
            "--resume",
        ]
    )
    assert args.command == "wideseek-run"
    assert args.resume is True


def test_independent_checkpoint_pair_carries_empty_sub_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "a" * 64
    previous_main = CheckpointState(
        policy_id="main",
        checkpoint_id="main:1",
        optimizer_step=1,
        checkpoint_digest=digest,
        directory=tmp_path / "previous-main",
    )
    previous_sub = CheckpointState(
        policy_id="sub",
        checkpoint_id="sub:1",
        optimizer_step=1,
        checkpoint_digest=digest,
        directory=tmp_path / "previous-sub",
    )
    updated_main = previous_main.model_copy(
        update={
            "checkpoint_id": "main:2",
            "optimizer_step": 2,
            "directory": tmp_path / "updated-main",
        }
    )
    loaded_ids: list[str] = []

    def fake_load(_: Path, checkpoint_id: str) -> CheckpointState:
        loaded_ids.append(checkpoint_id)
        assert checkpoint_id == "main:2"
        return updated_main

    monkeypatch.setattr(
        "heterospawn.experiments.wideseek._load_checkpoint_state",
        fake_load,
    )
    report = {
        "topology": "independent",
        "loaded_checkpoint": {
            "initialization": "independent_checkpoint_restore",
            "targets": [
                {"policy_id": "main", "checkpoint_id": "main:1"},
                {"policy_id": "sub", "checkpoint_id": "sub:1"},
            ],
        },
        "updates": [{"policy_id": "main", "checkpoint_id": "main:2"}],
    }

    checkpoints = _training_checkpoint_states(
        report,
        tmp_path / "current",
        (previous_main, previous_sub),
    )

    assert loaded_ids == ["main:2"]
    assert {checkpoint.policy_id: checkpoint for checkpoint in checkpoints} == {
        "main": updated_main,
        "sub": previous_sub,
    }
