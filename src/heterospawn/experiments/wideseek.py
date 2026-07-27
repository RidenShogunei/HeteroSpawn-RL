"""One resumable WideSeek SFT, RL, and evaluation experiment."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from heterospawn.backends.local_hf import load_local_checkpoint_ref
from heterospawn.backends.local_hf.config import (
    QWEN3_4B_MANIFEST_DIGEST,
    QWEN3_4B_MODEL_ID,
    QWEN3_4B_MODEL_REVISION,
    LocalLoraConfig,
)
from heterospawn.benchmarks.wideseek import WideSeekSplit
from heterospawn.domain.training import canonical_digest
from heterospawn.evaluation.wideseek_comparison import compare_wideseek_reports
from heterospawn.training.wideseek_sft_smoke import run_wideseek_sft_smoke
from heterospawn.training.wideseek_smoke import (
    run_wideseek_compliance_baseline,
    run_wideseek_train_smoke,
)

EXPERIMENT_SCHEMA_REVISION: Literal["heterospawn-wideseek-experiment-v1"] = (
    "heterospawn-wideseek-experiment-v1"
)
STATE_SCHEMA_REVISION: Literal["heterospawn-wideseek-experiment-state-v1"] = (
    "heterospawn-wideseek-experiment-state-v1"
)


class TaskSelection(BaseModel):
    """A public task coordinate without reference-answer material."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    split: WideSeekSplit
    task_index: int = Field(ge=0)


class SftSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    split: WideSeekSplit = "hybrid_20k"
    task_limit: int = Field(default=192, ge=1)
    tasks_per_step: int = Field(default=4, ge=1)
    epochs: int = Field(default=1, ge=1)
    training_max_sequence_length: int = Field(default=2304, ge=16)


class RlSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    split: WideSeekSplit = "width_20k"
    task_indices: tuple[int, ...]
    rollouts_per_task: int = Field(default=8, ge=2)
    shared_cycles: int = Field(default=1, ge=1)
    independent_cycles: int = Field(default=1, ge=1)
    require_sub_update: bool = True
    require_learning_signal: bool = True

    @model_validator(mode="after")
    def task_indices_are_unique(self) -> RlSchedule:
        if not self.task_indices or len(set(self.task_indices)) != len(self.task_indices):
            raise ValueError("RL task_indices must be non-empty and unique")
        if min(self.task_indices) < 0:
            raise ValueError("RL task_indices cannot be negative")
        return self


class EvaluationSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tasks: tuple[TaskSelection, ...]
    rollouts_per_task: int = Field(default=2, ge=1)
    bootstrap_samples: int = Field(default=10_000, ge=100)
    bootstrap_seed: int = 20_260_727

    @model_validator(mode="after")
    def tasks_are_unique(self) -> EvaluationSchedule:
        identities = tuple((task.split, task.task_index) for task in self.tasks)
        if not identities or len(set(identities)) != len(identities):
            raise ValueError("evaluation tasks must be non-empty and unique")
        return self


class ToolBudgets(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    search_results: int = Field(default=3, ge=1)
    search_content_characters: int = Field(default=600, ge=1)
    access_characters: int = Field(default=800, ge=1)


class WideSeekExperimentConfig(BaseModel):
    """The sole committed configuration for the runnable Qwen3 experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_revision: Literal["heterospawn-wideseek-experiment-v1"] = EXPERIMENT_SCHEMA_REVISION
    experiment_id: str = Field(min_length=1)
    model_profile: Literal["qwen3-4b"] = "qwen3-4b"
    max_sequence_length: int = Field(default=4096, ge=16)
    max_new_tokens: int = Field(default=1024, ge=1)
    do_sample: bool = True
    judge_mode: Literal["none", "minimax-development"] = "none"
    sft: SftSchedule
    rl: RlSchedule
    evaluation: EvaluationSchedule
    tools: ToolBudgets = ToolBudgets()

    @model_validator(mode="after")
    def sequence_limits_are_consistent(self) -> WideSeekExperimentConfig:
        if self.max_new_tokens >= self.max_sequence_length:
            raise ValueError("max_new_tokens must be smaller than max_sequence_length")
        if self.sft.training_max_sequence_length > self.max_sequence_length:
            raise ValueError("SFT training sequence limit exceeds the model sequence limit")
        return self


class CheckpointState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    policy_id: str
    checkpoint_id: str
    optimizer_step: int = Field(ge=0)
    checkpoint_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    directory: Path


class StageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str
    kind: Literal["sft", "train", "evaluate", "compare"]
    input_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_path: Path
    report_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoints: tuple[CheckpointState, ...] = ()


class ExperimentState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_revision: Literal["heterospawn-wideseek-experiment-state-v1"] = STATE_SCHEMA_REVISION
    experiment_id: str
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    stages: tuple[StageRecord, ...] = ()


@dataclass(frozen=True)
class WideSeekExperimentRuntime:
    run_dir: Path
    model_path: Path
    model_manifest_path: Path
    data_manifest_path: Path
    data_dir: Path
    device: str = "cuda:0"
    service_url: str = "http://127.0.0.1:8000"
    qdrant_url: str = "http://127.0.0.1:6333"
    allow_network_judge: bool = False


@dataclass(frozen=True)
class StageRequest:
    name: str
    kind: Literal["sft", "train", "evaluate", "compare"]
    topology: Literal["shared", "independent"] | None = None
    cycle_index: int | None = None
    dependencies: tuple[StageRecord, ...] = ()


class ExperimentExecutor(Protocol):
    async def execute(self, request: StageRequest) -> StageRecord: ...


def load_wideseek_experiment_config(path: Path) -> WideSeekExperimentConfig:
    """Load a strict JSON configuration and reject unknown or coerced values."""

    try:
        encoded = path.read_text(encoding="utf-8")
        return WideSeekExperimentConfig.model_validate_json(encoded)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load WideSeek experiment config: {path}") from exc


class WideSeekExperimentRunner:
    """Run or resume the one canonical experiment at committed stage boundaries."""

    def __init__(
        self,
        config: WideSeekExperimentConfig,
        runtime: WideSeekExperimentRuntime,
        executor: ExperimentExecutor | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.config_digest = canonical_digest(
            {
                "experiment": config.model_dump(mode="json"),
                "trusted_manifests": {
                    "model": _manifest_file_identity(runtime.model_manifest_path),
                    "data": _manifest_file_identity(runtime.data_manifest_path),
                },
            }
        )
        self.state_path = runtime.run_dir / "state.json"
        if config.judge_mode == "minimax-development" and not runtime.allow_network_judge:
            raise ValueError("development Judge requires explicit network acknowledgement")
        self.executor = executor or LocalWideSeekExperimentExecutor(config, runtime)

    async def run(self, *, resume: bool) -> dict[str, Any]:
        state = self._load_or_create_state(resume=resume)
        records = {record.name: record for record in state.stages}
        while True:
            progressed = False
            for request in self._plan(records):
                input_digest = _stage_input_digest(self.config_digest, request)
                existing = records.get(request.name)
                if existing is not None:
                    if existing.input_digest != input_digest:
                        raise ValueError(f"stage input changed: {request.name}")
                    _verify_stage_record(existing)
                    continue
                produced = await self.executor.execute(request)
                if produced.name != request.name or produced.kind != request.kind:
                    raise RuntimeError("experiment executor returned a record for another stage")
                if produced.input_digest != input_digest:
                    raise RuntimeError("experiment executor returned the wrong stage input digest")
                _verify_stage_record(produced)
                records[produced.name] = produced
                state = state.model_copy(update={"stages": tuple([*state.stages, produced])})
                _write_json(self.state_path, state.model_dump(mode="json"))
                await _release_accelerator_memory()
                progressed = True
                break
            if not progressed:
                break
        final = self._final_report(state)
        _write_json(self.runtime.run_dir / "report.json", final)
        return final

    def _load_or_create_state(self, *, resume: bool) -> ExperimentState:
        if self.state_path.exists():
            if not resume:
                raise ValueError("run state already exists; pass --resume to continue it")
            try:
                state = ExperimentState.model_validate_json(
                    self.state_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as exc:
                raise ValueError("cannot load experiment state") from exc
            if state.experiment_id != self.config.experiment_id:
                raise ValueError("experiment identity differs from saved state")
            if state.config_digest != self.config_digest:
                raise ValueError("experiment configuration differs from saved state")
            names = tuple(record.name for record in state.stages)
            if len(set(names)) != len(names):
                raise ValueError("experiment state contains duplicate stages")
            return state
        if resume:
            raise ValueError("cannot resume because the experiment state does not exist")
        self.runtime.run_dir.mkdir(parents=True, exist_ok=False)
        state = ExperimentState(
            experiment_id=self.config.experiment_id,
            config_digest=self.config_digest,
        )
        _write_json(self.state_path, state.model_dump(mode="json"))
        return state

    def _plan(self, records: dict[str, StageRecord]) -> tuple[StageRequest, ...]:
        requests: list[StageRequest] = [StageRequest(name="sft", kind="sft")]

        shared_dependency = records.get("sft")
        for cycle_index in range(self.config.rl.shared_cycles):
            name = f"shared-cycle-{cycle_index:03d}"
            dependency_name = "sft" if cycle_index == 0 else f"shared-cycle-{cycle_index - 1:03d}"
            dependency = records.get(dependency_name)
            requests.append(
                StageRequest(
                    name=name,
                    kind="train",
                    topology="shared",
                    cycle_index=cycle_index,
                    dependencies=(() if dependency is None else (dependency,)),
                )
            )
            shared_dependency = records.get(name) or shared_dependency

        independent_dependency = records.get("sft")
        for cycle_index in range(self.config.rl.independent_cycles):
            name = f"independent-cycle-{cycle_index:03d}"
            dependency_name = (
                "sft" if cycle_index == 0 else f"independent-cycle-{cycle_index - 1:03d}"
            )
            dependency = records.get(dependency_name)
            requests.append(
                StageRequest(
                    name=name,
                    kind="train",
                    topology="independent",
                    cycle_index=cycle_index,
                    dependencies=(() if dependency is None else (dependency,)),
                )
            )
            independent_dependency = records.get(name) or independent_dependency

        evaluation_dependencies: tuple[
            tuple[
                str,
                StageRecord | None,
                Literal["shared", "independent"],
            ],
            ...,
        ] = (
            ("eval-sft", records.get("sft"), "shared"),
            ("eval-shared", shared_dependency, "shared"),
            ("eval-independent", independent_dependency, "independent"),
        )
        for name, dependency, topology in evaluation_dependencies:
            requests.append(
                StageRequest(
                    name=name,
                    kind="evaluate",
                    topology=topology,
                    dependencies=(() if dependency is None else (dependency,)),
                )
            )
        comparison_dependencies = tuple(
            record
            for name in ("eval-sft", "eval-shared", "eval-independent")
            if (record := records.get(name)) is not None
        )
        requests.append(
            StageRequest(
                name="comparison",
                kind="compare",
                dependencies=comparison_dependencies,
            )
        )
        return tuple(requests)

    def _final_report(self, state: ExperimentState) -> dict[str, Any]:
        comparison = next(record for record in state.stages if record.name == "comparison")
        return {
            "schema_revision": EXPERIMENT_SCHEMA_REVISION,
            "status": "passed",
            "experiment_id": state.experiment_id,
            "config_digest": state.config_digest,
            "completed_stages": [record.name for record in state.stages],
            "comparison_report": str(comparison.report_path),
            "comparison_report_digest": comparison.report_digest,
            "state_digest": canonical_digest(state.model_dump(mode="json")),
            "recovery_semantics": {
                "between_stages": "automatic-with---resume",
                "within_training_phase": "fail-closed-wideseek-recover-phase",
            },
        }


class LocalWideSeekExperimentExecutor:
    """Adapt the existing real-model stages to the unified experiment state."""

    def __init__(
        self,
        config: WideSeekExperimentConfig,
        runtime: WideSeekExperimentRuntime,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.config_digest = canonical_digest(
            {
                "experiment": config.model_dump(mode="json"),
                "trusted_manifests": {
                    "model": _manifest_file_identity(runtime.model_manifest_path),
                    "data": _manifest_file_identity(runtime.data_manifest_path),
                },
            }
        )

    async def execute(self, request: StageRequest) -> StageRecord:
        if request.kind != "sft" and not request.dependencies:
            raise RuntimeError(f"stage dependency has not committed: {request.name}")
        report_path = self.runtime.run_dir / "reports" / f"{request.name}.json"
        artifact_dir = self.runtime.run_dir / "checkpoints" / request.name
        if request.kind == "sft":
            report = await self._run_sft(report_path, artifact_dir)
            checkpoints = _checkpoint_states(
                report,
                artifact_dir,
                source="sft",
            )
        elif request.kind == "train":
            report = await self._run_training(request, report_path, artifact_dir)
            self._validate_runtime_contract(report)
            checkpoints = _training_checkpoint_states(
                report,
                artifact_dir,
                request.dependencies[0].checkpoints,
            )
        elif request.kind == "evaluate":
            report = await self._run_evaluation(request, report_path, artifact_dir)
            self._validate_runtime_contract(report)
            checkpoints = ()
        else:
            self._run_comparison(request, report_path)
            checkpoints = ()
        return StageRecord(
            name=request.name,
            kind=request.kind,
            input_digest=_stage_input_digest(self.config_digest, request),
            report_path=report_path.resolve(),
            report_digest=_file_sha256(report_path),
            checkpoints=checkpoints,
        )

    async def _run_sft(self, report_path: Path, artifact_dir: Path) -> dict[str, Any]:
        schedule = self.config.sft
        return await run_wideseek_sft_smoke(
            split=schedule.split,
            task_indices=(),
            task_limit=schedule.task_limit,
            tasks_per_step=schedule.tasks_per_step,
            epochs=schedule.epochs,
            training_max_sequence_length=schedule.training_max_sequence_length,
            exclude_compliance_selection=True,
            data_manifest_path=self.runtime.data_manifest_path,
            data_dir=self.runtime.data_dir,
            local_config=self._local_config(artifact_dir),
            report_path=report_path,
            max_workers=4,
        )

    async def _run_training(
        self,
        request: StageRequest,
        report_path: Path,
        artifact_dir: Path,
    ) -> dict[str, Any]:
        assert request.topology is not None
        assert request.cycle_index is not None
        dependency = request.dependencies[0]
        checkpoint_dir: Path | None
        main_checkpoint_dir: Path | None
        sub_checkpoint_dir: Path | None
        if request.topology == "shared":
            checkpoint_dir = _checkpoint_for_policy(dependency, "shared").directory
            main_checkpoint_dir = None
            sub_checkpoint_dir = None
        elif request.cycle_index == 0:
            checkpoint_dir = _checkpoint_for_policy(dependency, "shared").directory
            main_checkpoint_dir = None
            sub_checkpoint_dir = None
        else:
            checkpoint_dir = None
            main_checkpoint_dir = _checkpoint_for_policy(dependency, "main").directory
            sub_checkpoint_dir = _checkpoint_for_policy(dependency, "sub").directory
        schedule = self.config.rl
        return await run_wideseek_train_smoke(
            topology=request.topology,
            split=schedule.split,
            task_indices=schedule.task_indices,
            rollouts_per_task=schedule.rollouts_per_task,
            data_manifest_path=self.runtime.data_manifest_path,
            data_dir=self.runtime.data_dir,
            service_url=self.runtime.service_url,
            qdrant_url=self.runtime.qdrant_url,
            local_config=self._local_config(artifact_dir),
            judge_mode=self.config.judge_mode,
            transaction_dir=self.runtime.run_dir / "transactions",
            report_path=report_path,
            cycle_index=request.cycle_index,
            experiment_id=self.config.experiment_id,
            require_sub_update=schedule.require_sub_update,
            require_learning_signal=schedule.require_learning_signal,
            do_sample=self.config.do_sample,
            max_search_message_results=self.config.tools.search_results,
            max_search_content_characters=self.config.tools.search_content_characters,
            max_access_characters=self.config.tools.access_characters,
            checkpoint_dir=checkpoint_dir,
            main_checkpoint_dir=main_checkpoint_dir,
            sub_checkpoint_dir=sub_checkpoint_dir,
        )

    async def _run_evaluation(
        self,
        request: StageRequest,
        report_path: Path,
        artifact_dir: Path,
    ) -> dict[str, Any]:
        assert request.topology is not None
        dependency = request.dependencies[0]
        checkpoint_dir: Path | None
        main_checkpoint_dir: Path | None
        sub_checkpoint_dir: Path | None
        if request.topology == "shared":
            checkpoint_dir = _checkpoint_for_policy(dependency, "shared").directory
            main_checkpoint_dir = None
            sub_checkpoint_dir = None
        else:
            checkpoint_dir = None
            main_checkpoint_dir = _checkpoint_for_policy(dependency, "main").directory
            sub_checkpoint_dir = _checkpoint_for_policy(dependency, "sub").directory
        return await run_wideseek_compliance_baseline(
            topology=request.topology,
            task_selection=tuple(
                (task.split, task.task_index) for task in self.config.evaluation.tasks
            ),
            rollouts_per_task=self.config.evaluation.rollouts_per_task,
            data_manifest_path=self.runtime.data_manifest_path,
            data_dir=self.runtime.data_dir,
            service_url=self.runtime.service_url,
            qdrant_url=self.runtime.qdrant_url,
            local_config=self._local_config(artifact_dir),
            report_path=report_path,
            do_sample=self.config.do_sample,
            max_search_message_results=self.config.tools.search_results,
            max_search_content_characters=self.config.tools.search_content_characters,
            max_access_characters=self.config.tools.access_characters,
            checkpoint_dir=checkpoint_dir,
            main_checkpoint_dir=main_checkpoint_dir,
            sub_checkpoint_dir=sub_checkpoint_dir,
        )

    def _validate_runtime_contract(self, report: dict[str, Any]) -> None:
        keys = (
            "model_id",
            "model_revision",
            "model_identity_kind",
            "model_identity",
            "max_sequence_length",
            "max_new_tokens",
            "sampling_params",
            "tool_message_budgets",
            "environment_revision",
            "dataset_revision",
        )
        observed = {key: report.get(key) for key in keys}
        reference_path = self.runtime.run_dir / "reports" / "shared-cycle-000.json"
        if reference_path.exists():
            try:
                reference = json.loads(reference_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("cannot load experiment runtime-contract reference") from exc
            if not isinstance(reference, dict):
                raise ValueError("experiment runtime-contract reference is invalid")
            expected = {key: reference.get(key) for key in keys}
            if observed != expected:
                differing = sorted(key for key in keys if observed[key] != expected.get(key))
                raise ValueError(f"experiment runtime contract drifted: {', '.join(differing)}")

    def _run_comparison(self, request: StageRequest, report_path: Path) -> None:
        reports = {dependency.name: dependency.report_path for dependency in request.dependencies}
        compare_wideseek_reports(
            baseline_name="sft",
            baseline_report=reports["eval-sft"],
            candidate_reports=(
                ("shared-rl", reports["eval-shared"]),
                ("heterospawn-rl", reports["eval-independent"]),
            ),
            output_path=report_path,
            bootstrap_samples=self.config.evaluation.bootstrap_samples,
            seed=self.config.evaluation.bootstrap_seed,
        )

    def _local_config(self, artifact_dir: Path) -> LocalLoraConfig:
        return LocalLoraConfig(
            model_id=QWEN3_4B_MODEL_ID,
            model_revision=QWEN3_4B_MODEL_REVISION,
            model_path=self.runtime.model_path,
            model_manifest_path=self.runtime.model_manifest_path,
            expected_model_manifest_digest=QWEN3_4B_MANIFEST_DIGEST,
            device=self.runtime.device,
            dtype="float16",
            quantization="bnb-4bit",
            attention_implementation="sdpa",
            response_only_logits=True,
            gradient_checkpointing=True,
            enable_thinking=False,
            max_sequence_length=self.config.max_sequence_length,
            max_new_tokens=self.config.max_new_tokens,
            artifact_dir=artifact_dir,
        )


def _stage_input_digest(config_digest: str, request: StageRequest) -> str:
    return canonical_digest(
        {
            "config_digest": config_digest,
            "name": request.name,
            "kind": request.kind,
            "topology": request.topology,
            "cycle_index": request.cycle_index,
            "dependencies": [
                {
                    "name": dependency.name,
                    "report_digest": dependency.report_digest,
                    "checkpoints": [
                        checkpoint.model_dump(mode="json") for checkpoint in dependency.checkpoints
                    ],
                }
                for dependency in request.dependencies
            ],
        }
    )


def _checkpoint_states(
    report: dict[str, Any],
    artifact_dir: Path,
    *,
    source: Literal["sft"],
) -> tuple[CheckpointState, ...]:
    if source != "sft":
        raise RuntimeError("unsupported checkpoint report source")
    checkpoint_id = _required_string(report["checkpoint"], "checkpoint_id")
    return (_load_checkpoint_state(artifact_dir, checkpoint_id),)


def _training_checkpoint_states(
    report: dict[str, Any],
    artifact_dir: Path,
    input_checkpoints: tuple[CheckpointState, ...],
) -> tuple[CheckpointState, ...]:
    expected = {"shared"} if report["topology"] == "shared" else {"main", "sub"}
    by_policy = {
        checkpoint.policy_id: checkpoint
        for checkpoint in input_checkpoints
        if checkpoint.policy_id in expected
    }
    loaded = report.get("loaded_checkpoint")
    if isinstance(loaded, dict):
        for target in loaded.get("targets", ()):
            if isinstance(target, dict):
                policy_id = _required_string(target, "policy_id")
                if policy_id in expected and policy_id not in by_policy:
                    by_policy[policy_id] = _load_checkpoint_state(
                        artifact_dir,
                        _required_string(target, "checkpoint_id"),
                    )
    for update in report["updates"]:
        policy_id = _required_string(update, "policy_id")
        by_policy[policy_id] = _load_checkpoint_state(
            artifact_dir,
            _required_string(update, "checkpoint_id"),
        )
    if set(by_policy) != expected:
        raise RuntimeError("training stage did not produce a complete policy checkpoint set")
    return tuple(by_policy[policy_id] for policy_id in sorted(by_policy))


def _load_checkpoint_state(artifact_dir: Path, checkpoint_id: str) -> CheckpointState:
    directory = (artifact_dir / checkpoint_id.replace(":", "_")).resolve()
    checkpoint = load_local_checkpoint_ref(directory)
    if str(checkpoint.checkpoint_id) != checkpoint_id:
        raise RuntimeError("checkpoint identity differs from its directory")
    return CheckpointState(
        policy_id=str(checkpoint.policy_id),
        checkpoint_id=checkpoint_id,
        optimizer_step=checkpoint.weight_version.optimizer_step,
        checkpoint_digest=checkpoint.weight_version.checkpoint_digest,
        directory=directory,
    )


def _checkpoint_for_policy(record: StageRecord, policy_id: str) -> CheckpointState:
    try:
        return next(
            checkpoint for checkpoint in record.checkpoints if checkpoint.policy_id == policy_id
        )
    except StopIteration:
        raise RuntimeError(f"{record.name} has no {policy_id} checkpoint") from None


def _verify_stage_record(record: StageRecord) -> None:
    if not record.report_path.is_file():
        raise ValueError(f"stage report is missing: {record.name}")
    if _file_sha256(record.report_path) != record.report_digest:
        raise ValueError(f"stage report digest differs: {record.name}")
    try:
        report = json.loads(record.report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"stage report is invalid: {record.name}") from exc
    if not isinstance(report, dict) or report.get("status") != "passed":
        raise ValueError(f"stage did not pass: {record.name}")
    for expected in record.checkpoints:
        observed = load_local_checkpoint_ref(expected.directory)
        if (
            str(observed.policy_id) != expected.policy_id
            or str(observed.checkpoint_id) != expected.checkpoint_id
            or observed.weight_version.optimizer_step != expected.optimizer_step
            or observed.weight_version.checkpoint_digest != expected.checkpoint_digest
        ):
            raise ValueError(f"stage checkpoint identity differs: {record.name}")


def _required_string(payload: object, key: str) -> str:
    if not isinstance(payload, dict):
        raise RuntimeError("stage report section must be an object")
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"stage report is missing {key}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_file_identity(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path_kind": "verified-content" if resolved.is_file() else "missing",
        "sha256": _file_sha256(resolved) if resolved.is_file() else "",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


async def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        await asyncio.sleep(0)
