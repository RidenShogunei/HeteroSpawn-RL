"""Credential-safe WideSeek rollout and LocalHF short-training validation."""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import math
import random
import statistics
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Protocol

from heterospawn.assets import load_asset_manifest
from heterospawn.backends.local_hf.backend import (
    LocalHfLoraBackend,
    load_local_checkpoint_ref,
)
from heterospawn.backends.local_hf.config import LocalLoraConfig
from heterospawn.benchmarks.wideseek import (
    WideSeekDataset,
    WideSeekSplit,
    load_wideseek_dataset,
)
from heterospawn.domain.ids import EpisodeId, PolicyId, RolloutId, TaskId
from heterospawn.domain.tasks import ResearchTask
from heterospawn.domain.training import (
    GenerationRequest,
    GenerationResult,
    JsonScalar,
    PromptEncoding,
    canonical_digest,
)
from heterospawn.domain.versions import RoleBinding, RolloutRevision
from heterospawn.evaluation.semantic_judge import (
    MiniMaxSemanticJudge,
    SemanticJudgeCache,
)
from heterospawn.evaluation.wideseek import (
    WideSeekEvaluation,
    WideSeekEvaluator,
)
from heterospawn.orchestration.trainable_models import TrainableEpisodeTrace
from heterospawn.orchestration.wideseek_actions import WIDESEEK_TOOL_SCHEMA_REVISION
from heterospawn.orchestration.wideseek_episode import (
    WideSeekEpisodeOrchestrator,
)
from heterospawn.policies.base import Message
from heterospawn.policies.minimax import MiniMaxChatClient, MiniMaxConfig
from heterospawn.policies.trainable import ToolDefinition
from heterospawn.search.base import SearchRequest
from heterospawn.search.wideseek_local import (
    WideSeekLocalConfig,
    WideSeekLocalToolService,
)
from heterospawn.training.episode_cycle import (
    RewardComposer,
    RewardConfig,
    TrainableAlternatingCycleRunner,
)
from heterospawn.training.mock import MockTrainingBackend
from heterospawn.training.registry import PolicyRegistry
from heterospawn.training.transactions import (
    FilePhaseTransactionStore,
    PhaseTransactionContext,
)
from heterospawn.training.wideseek_reward import (
    WideSeekRewardConfig,
    WideSeekRewardService,
)

Topology = Literal["shared", "independent"]
JudgeMode = Literal["none", "minimax-development"]
ComplianceSelection = tuple[tuple[WideSeekSplit, int], ...]
WIDESEEK_COMPLIANCE_PROFILE_V1: ComplianceSelection = (
    ("width_20k", 0),
    ("width_20k", 3999),
    ("width_20k", 7999),
    ("width_20k", 11999),
    ("width_20k", 15999),
    ("width_20k", 19999),
    ("depth_20k", 0),
    ("depth_20k", 4999),
    ("depth_20k", 9999),
    ("depth_20k", 14999),
    ("depth_20k", 19999),
    ("hybrid_20k", 0),
    ("hybrid_20k", 4999),
    ("hybrid_20k", 9999),
    ("hybrid_20k", 14999),
    ("hybrid_20k", 19999),
)


class _ComplianceBackend(Protocol):
    prompt_encoder: Any

    def endpoint(self, policy_id: PolicyId) -> Any: ...

    def rollout_revision(self, policy_id: PolicyId) -> RolloutRevision: ...

    def adapter_hash(self, policy_id: PolicyId, *, rollout: bool = False) -> str: ...


class _Utf8ToolCodec:
    def encode(
        self,
        messages: tuple[Message, ...],
        tools: tuple[ToolDefinition, ...] = (),
    ) -> PromptEncoding:
        payload = {
            "messages": [message.model_dump(mode="json") for message in messages],
            "tools": [tool.model_dump(mode="json") for tool in tools],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return PromptEncoding(
            prompt_ids=tuple(encoded),
            tokenizer_revision="utf8-validation-v1",
            prompt_template_revision=canonical_digest(
                {"codec": "utf8-validation-v1", "tools": payload["tools"]}
            ),
        )

    def decode(self, response_ids: tuple[int, ...]) -> str:
        return bytes(response_ids).decode()


class _ScriptedPolicyService:
    """Forces one real-shaped Search-to-Access loop without using API text as training data."""

    def __init__(
        self,
        backend: MockTrainingBackend,
        policy_id: PolicyId,
        *,
        discovered_url: str,
    ) -> None:
        self._backend = backend
        self._policy_id = policy_id
        self._discovered_url = discovered_url

    @property
    def policy_id(self) -> PolicyId:
        return self._policy_id

    async def current_rollout_revision(self) -> RolloutRevision:
        return self._backend.rollout_revision(self._policy_id)

    async def generate(
        self,
        request: GenerationRequest,
        expected_revision: RolloutRevision,
    ) -> GenerationResult:
        if expected_revision != self._backend.rollout_revision(self._policy_id):
            raise RuntimeError("scripted rollout revision is stale")
        if request.agent_role == "main":
            content = (
                '<tool_call>{"name":"subtask","arguments":{"subtask":"Red Bull"}}</tool_call>'
                if ":main:round-0:" in request.request_id
                else "Offline environment contract completed."
            )
        elif ":turn-0:" in request.request_id:
            content = (
                '<tool_call>{"name":"search","arguments":{"query":"Red Bull","topk":1}}</tool_call>'
            )
        elif ":turn-1:" in request.request_id:
            content = (
                '<tool_call>{"name":"access","arguments":{"url":'
                f"{json.dumps(self._discovered_url)},"
                '"info_to_extract":"basic facts"}}</tool_call>'
            )
        else:
            content = "Evidence summary completed."
        response_ids = tuple(content.encode())
        return GenerationResult(
            request_id=request.request_id,
            policy_id=self._policy_id,
            rollout_revision=expected_revision,
            response_ids=response_ids,
            response_log_probs=tuple(-0.01 for _ in response_ids),
            stop_reason="eos",
        )


async def run_wideseek_rollout_smoke(
    *,
    service_url: str,
    qdrant_url: str,
    report_path: Path,
    tool_service: WideSeekLocalToolService | None = None,
) -> dict[str, Any]:
    """Exercise one offline Search-to-Access episode with deterministic policy actions."""

    environment_mode = "controlled-fixture" if tool_service is not None else "offline-pinned"
    tools = tool_service or WideSeekLocalToolService(
        WideSeekLocalConfig(service_url=service_url, qdrant_url=qdrant_url)
    )
    probe = await tools.search(
        SearchRequest(request_id="rollout-smoke-probe", query="Red Bull", max_results=1)
    )
    if not probe.results:
        raise RuntimeError("WideSeek rollout smoke could not discover a URL")

    main_id = PolicyId("main-validation")
    sub_id = PolicyId("sub-validation")
    backend = MockTrainingBackend((main_id, sub_id))
    registry = PolicyRegistry(
        (
            RoleBinding(role="main", policy_id=main_id, trainable=True),
            RoleBinding(role="sub", policy_id=sub_id, trainable=True),
        ),
        (
            (main_id, backend.rollout_revision(main_id)),
            (sub_id, backend.rollout_revision(sub_id)),
        ),
    )
    codec = _Utf8ToolCodec()
    orchestrator = WideSeekEpisodeOrchestrator(
        registry,
        {
            "main": _ScriptedPolicyService(
                backend,
                main_id,
                discovered_url=probe.results[0].url,
            ),
            "sub": _ScriptedPolicyService(
                backend,
                sub_id,
                discovered_url=probe.results[0].url,
            ),
        },
        {"main": codec, "sub": codec},
        tools,
        max_concurrency=3,
        sampling_params=(("max_new_tokens", 128), ("do_sample", False)),
    )
    trace = await orchestrator.run(
        ResearchTask(
            task_id=TaskId("wideseek-offline-rollout-smoke"),
            prompt="Delegate one worker to research Red Bull.",
            dataset_revision="controlled-wideseek-shape-v1",
        ),
        EpisodeId("wideseek-offline-rollout-smoke"),
        RolloutId("wideseek-offline-rollout-smoke"),
        registry.snapshot(),
    )
    tool_sequence = tuple(outcome.tool_name for outcome in trace.tool_outcomes)
    exact_tokens = all(
        len(step.response_ids) == len(step.response_log_probs) for step in trace.model_steps
    )
    if trace.status != "success" or tool_sequence != ("search", "access") or not exact_tokens:
        raise RuntimeError("WideSeek rollout smoke failed its Search-to-Access contract")
    report: dict[str, Any] = {
        "schema_revision": "heterospawn-wideseek-rollout-smoke-v1",
        "status": "passed",
        "environment_mode": environment_mode,
        "environment_revision": tools.provider_revision,
        "episode_status": trace.status,
        "spawn_count": trace.spawn_count,
        "spawn_rounds": len(trace.spawn_rounds),
        "model_steps": len(trace.model_steps),
        "tool_sequence": list(tool_sequence),
        "exact_token_logprob_alignment": exact_tokens,
        "stable_event_order": tuple(event.event_index for event in trace.events)
        == tuple(range(len(trace.events))),
        "access_has_search_provenance": trace.tool_outcomes[1].source_search_step_id
        == trace.tool_outcomes[0].step_id,
    }
    await asyncio.to_thread(_write_report, report_path, report)
    return report


async def run_wideseek_compliance_baseline(
    *,
    topology: Topology,
    task_selection: ComplianceSelection,
    rollouts_per_task: int,
    data_manifest_path: Path,
    data_dir: Path,
    service_url: str,
    qdrant_url: str,
    local_config: LocalLoraConfig,
    report_path: Path,
    checkpoint_dir: Path | None = None,
    tool_service: WideSeekLocalToolService | None = None,
    backend: _ComplianceBackend | None = None,
    do_sample: bool = True,
    max_search_message_results: int = 3,
    max_search_content_characters: int = 600,
    max_access_characters: int = 800,
) -> dict[str, Any]:
    """Measure policy compliance and exact outcome without an optimizer update."""

    if not task_selection or len(set(task_selection)) != len(task_selection):
        raise ValueError("compliance task selection must be non-empty and unique")
    if rollouts_per_task < 1:
        raise ValueError("compliance baseline requires at least one rollout per task")
    if checkpoint_dir is not None and backend is not None:
        raise ValueError("checkpoint_dir cannot be combined with an injected backend")
    if checkpoint_dir is not None and topology != "shared":
        raise ValueError("one checkpoint directory can restore only the shared topology")

    manifest = load_asset_manifest(data_manifest_path)
    expected_by_split = {
        split: next(
            (
                file
                for file in manifest.files
                if file.path == f"{split}.jsonl" and file.sha256 is not None
            ),
            None,
        )
        for split in ("width_20k", "depth_20k", "hybrid_20k")
    }
    requested_splits = tuple(dict.fromkeys(split for split, _ in task_selection))
    datasets: dict[WideSeekSplit, WideSeekDataset] = {}
    for split in requested_splits:
        expected = expected_by_split[split]
        if expected is None or expected.sha256 is None:
            raise ValueError(f"{split} is absent from the trusted manifest")
        datasets[split] = load_wideseek_dataset(
            data_dir / f"{split}.jsonl",
            split=split,
            expected_sha256=expected.sha256,
            revision=manifest.revision,
        )
    selected_tasks: list[tuple[WideSeekSplit, int, ResearchTask]] = []
    for split, task_index in task_selection:
        try:
            task = datasets[split].tasks[task_index]
        except IndexError:
            raise ValueError(
                f"WideSeek compliance task index is out of range: {split}:{task_index}"
            ) from None
        selected_tasks.append((split, task_index, task))

    policy_ids = (
        (PolicyId("shared"),) if topology == "shared" else (PolicyId("main"), PolicyId("sub"))
    )
    loaded_checkpoint = None
    if backend is None:
        local_backend = LocalHfLoraBackend.from_pretrained(
            config=local_config,
            policy_ids=policy_ids,
        )
        if checkpoint_dir is not None:
            checkpoint = load_local_checkpoint_ref(checkpoint_dir)
            if checkpoint.policy_id != policy_ids[0]:
                raise ValueError("checkpoint policy does not match the shared compliance policy")
            restored_version = await local_backend.restore_checkpoint(checkpoint)
            await local_backend.sync_rollout_weights(checkpoint.policy_id, restored_version)
            loaded_checkpoint = {
                "checkpoint_id": str(checkpoint.checkpoint_id),
                "policy_id": str(checkpoint.policy_id),
                "optimizer_step": restored_version.optimizer_step,
                "checkpoint_digest": restored_version.checkpoint_digest,
            }
        active_backend: _ComplianceBackend = local_backend
    else:
        active_backend = backend
    main_id = policy_ids[0]
    sub_id = main_id if topology == "shared" else PolicyId("sub")
    registry = PolicyRegistry(
        (
            RoleBinding(role="main", policy_id=main_id, trainable=False),
            RoleBinding(role="sub", policy_id=sub_id, trainable=False),
        ),
        tuple((policy_id, active_backend.rollout_revision(policy_id)) for policy_id in policy_ids),
    )
    tools = tool_service or WideSeekLocalToolService(
        WideSeekLocalConfig(service_url=service_url, qdrant_url=qdrant_url)
    )
    sampling_params: tuple[tuple[str, JsonScalar], ...] = (
        (
            ("max_new_tokens", local_config.max_new_tokens),
            ("do_sample", True),
            ("temperature", 1.0),
            ("top_p", 1.0),
            ("top_k", 0),
        )
        if do_sample
        else (
            ("max_new_tokens", local_config.max_new_tokens),
            ("do_sample", False),
        )
    )
    orchestrator = WideSeekEpisodeOrchestrator(
        registry,
        {
            "main": active_backend.endpoint(main_id),
            "sub": active_backend.endpoint(sub_id),
        },
        {
            "main": active_backend.prompt_encoder,
            "sub": active_backend.prompt_encoder,
        },
        tools,
        max_concurrency=4,
        max_search_message_results=max_search_message_results,
        max_search_content_characters=max_search_content_characters,
        max_access_characters=max_access_characters,
        sampling_params=sampling_params,
    )
    evaluators = {split: WideSeekEvaluator(datasets[split]) for split in requested_splits}
    initial_registry_revisions = registry.snapshot()
    initial_backend_revisions = tuple(
        (policy_id, active_backend.rollout_revision(policy_id)) for policy_id in policy_ids
    )
    initial_hashes = {
        f"{policy_id}:{adapter_kind}": active_backend.adapter_hash(
            policy_id,
            rollout=adapter_kind == "rollout",
        )
        for policy_id in policy_ids
        for adapter_kind in ("train", "rollout")
    }
    cuda_enabled = str(local_config.device).startswith("cuda")
    torch = importlib.import_module("torch") if cuda_enabled else None
    if torch is not None:
        with torch.cuda.device(local_config.device):
            torch.cuda.reset_peak_memory_stats()

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for split, task_index, task in selected_tasks:
        evaluator = evaluators[split]
        for rollout_index in range(rollouts_per_task):
            identity = f"compliance:{split}:{task_index}:r{rollout_index}"
            trace = await orchestrator.run(
                task,
                EpisodeId(identity),
                RolloutId(identity),
                registry.snapshot(),
            )
            evaluation = (
                await evaluator.evaluate(
                    task,
                    trace.answer or "",
                    request_id=identity,
                )
                if trace.status == "success"
                else None
            )
            records.append(
                _compliance_record(
                    split=split,
                    task_index=task_index,
                    rollout_index=rollout_index,
                    trace=trace,
                    evaluation=evaluation,
                )
            )
    elapsed = time.perf_counter() - started

    final_registry_revisions = registry.snapshot()
    final_backend_revisions = tuple(
        (policy_id, active_backend.rollout_revision(policy_id)) for policy_id in policy_ids
    )
    final_hashes = {
        f"{policy_id}:{adapter_kind}": active_backend.adapter_hash(
            policy_id,
            rollout=adapter_kind == "rollout",
        )
        for policy_id in policy_ids
        for adapter_kind in ("train", "rollout")
    }
    revisions_unchanged = (
        final_registry_revisions == initial_registry_revisions
        and final_backend_revisions == initial_backend_revisions
    )
    adapters_unchanged = final_hashes == initial_hashes
    peak_bytes = 0
    gpu_name = None
    if torch is not None:
        device_index = torch.device(local_config.device).index or 0
        gpu_name = torch.cuda.get_device_name(device_index)
        with torch.cuda.device(local_config.device):
            peak_bytes = int(torch.cuda.max_memory_allocated())

    summary = _summarize_compliance(records)
    summary["by_split"] = {
        split: _summarize_compliance([record for record in records if record["split"] == split])
        for split in requested_splits
    }
    report: dict[str, Any] = {
        "schema_revision": "heterospawn-wideseek-compliance-v2",
        "status": "passed",
        "comparable_to_official": False,
        "optimizer_updates": 0,
        "topology": topology,
        "selection_profile": canonical_digest(task_selection),
        "task_selection": [
            {"split": split, "task_index": task_index} for split, task_index in task_selection
        ],
        "rollouts_per_task": rollouts_per_task,
        "model_id": local_config.model_id,
        "model_revision": local_config.model_revision,
        "model_identity_kind": local_config.base_model_identity_kind,
        "model_identity": local_config.base_model_identity,
        "quantization": local_config.quantization,
        "gradient_checkpointing": local_config.gradient_checkpointing,
        "enable_thinking": local_config.enable_thinking,
        "max_sequence_length": local_config.max_sequence_length,
        "max_new_tokens": local_config.max_new_tokens,
        "seed": local_config.seed,
        "loaded_checkpoint": loaded_checkpoint,
        "policy_weight_versions": {
            str(policy_id): revision.weight_version.model_dump(mode="json")
            for policy_id, revision in initial_backend_revisions
        },
        "sampling_params": dict(sampling_params),
        "sampling_logprob_semantics": "raw-policy",
        "tool_message_budgets": {
            "search_results": max_search_message_results,
            "search_content_characters": max_search_content_characters,
            "access_characters": max_access_characters,
        },
        "environment_revision": tools.provider_revision,
        "dataset_revision": manifest.revision,
        "evaluator_revisions": {split: evaluators[split].revision for split in requested_splits},
        "judge_mode": "none",
        "judge_provider_requests": 0,
        "elapsed_seconds": elapsed,
        "device": local_config.device,
        "gpu_name": gpu_name,
        "peak_allocated_vram_bytes": peak_bytes,
        "checks": {
            "exact_token_logprob_alignment": all(
                record["exact_token_logprob_alignment"] for record in records
            ),
            "stable_event_order": all(record["stable_event_order"] for record in records),
            "weight_versions_unchanged": revisions_unchanged,
            "adapter_hashes_unchanged": adapters_unchanged,
        },
        "summary": summary,
        "episodes": records,
    }
    if not all(report["checks"].values()):
        raise RuntimeError("WideSeek compliance baseline violated a rollout-only contract")
    await asyncio.to_thread(_write_report, report_path, report)
    return report


def _compliance_record(
    *,
    split: WideSeekSplit,
    task_index: int,
    rollout_index: int,
    trace: TrainableEpisodeTrace,
    evaluation: WideSeekEvaluation | None,
) -> dict[str, Any]:
    stop_reasons = Counter(str(step.stop_reason) for step in trace.model_steps)
    tool_counts = Counter(
        f"{outcome.tool_name}:{outcome.status}" for outcome in trace.tool_outcomes
    )
    prompt_token_counts = [len(step.prompt_ids) for step in trace.model_steps]
    response_token_counts = [len(step.response_ids) for step in trace.model_steps]
    return {
        "split": split,
        "task_index": task_index,
        "rollout_index": rollout_index,
        "task_id": str(trace.task_id),
        "status": trace.status,
        "failure_code": trace.failure_code,
        "format_ok": evaluation.format_ok if evaluation is not None else False,
        "outcome_score": evaluation.outcome_score if evaluation is not None else 0.0,
        "spawn_count": trace.spawn_count,
        "spawn_rounds": len(trace.spawn_rounds),
        "invalid_main_attempts": trace.invalid_main_attempts,
        "failed_subs": trace.failed_subs,
        "model_steps": len(trace.model_steps),
        "main_model_steps": sum(step.agent_role == "main" for step in trace.model_steps),
        "sub_model_steps": sum(step.agent_role == "sub" for step in trace.model_steps),
        "tool_counts": dict(sorted(tool_counts.items())),
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "length_truncated": stop_reasons["length"] > 0,
        "prompt_token_counts": prompt_token_counts,
        "response_token_counts": response_token_counts,
        "sequence_token_counts": [
            prompt + response
            for prompt, response in zip(
                prompt_token_counts,
                response_token_counts,
                strict=True,
            )
        ],
        "exact_token_logprob_alignment": all(
            len(step.response_ids) == len(step.response_log_probs) for step in trace.model_steps
        ),
        "stable_event_order": tuple(event.event_index for event in trace.events)
        == tuple(range(len(trace.events))),
    }


def _summarize_compliance(records: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = len(records)
    scores = [float(record["outcome_score"]) for record in records]
    spawn_counts = Counter(str(record["spawn_count"]) for record in records)
    statuses = Counter(str(record["status"]) for record in records)
    failures = Counter(
        str(record["failure_code"]) for record in records if record["failure_code"] is not None
    )
    prompt_tokens = [int(count) for record in records for count in record["prompt_token_counts"]]
    response_tokens = [
        int(count) for record in records for count in record["response_token_counts"]
    ]
    sequence_tokens = [
        int(count) for record in records for count in record["sequence_token_counts"]
    ]
    return {
        "episodes": episodes,
        "success_rate": _rate(record["status"] == "success" for record in records),
        "format_ok_rate": _rate(record["format_ok"] for record in records),
        "nonzero_outcome_rate": _rate(score > 0 for score in scores),
        "outcome_mean": statistics.fmean(scores),
        "outcome_population_std": statistics.pstdev(scores),
        "spawn_rate": _rate(int(record["spawn_count"]) > 0 for record in records),
        "zero_spawn_rate": _rate(int(record["spawn_count"]) == 0 for record in records),
        "invalid_main_episode_rate": _rate(
            int(record["invalid_main_attempts"]) > 0 for record in records
        ),
        "failed_sub_episode_rate": _rate(int(record["failed_subs"]) > 0 for record in records),
        "length_truncation_rate": _rate(record["length_truncated"] for record in records),
        "status_counts": dict(sorted(statuses.items())),
        "failure_counts": dict(sorted(failures.items())),
        "spawn_count_histogram": dict(sorted(spawn_counts.items())),
        "model_steps": sum(int(record["model_steps"]) for record in records),
        "tool_calls": sum(
            sum(int(count) for count in record["tool_counts"].values()) for record in records
        ),
        "prompt_token_counts": _token_count_summary(prompt_tokens),
        "response_token_counts": _token_count_summary(response_tokens),
        "sequence_token_counts": _token_count_summary(sequence_tokens),
    }


def _token_count_summary(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "p50": 0, "p95": 0, "max": 0, "total": 0}
    return {
        "count": len(ordered),
        "p50": ordered[math.ceil(0.50 * len(ordered)) - 1],
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max": ordered[-1],
        "total": sum(ordered),
    }


def _rate(values: Iterable[object]) -> float:
    materialized = tuple(bool(value) for value in values)
    return sum(materialized) / len(materialized) if materialized else 0.0


async def run_wideseek_train_smoke(
    *,
    topology: Topology,
    split: WideSeekSplit,
    task_indices: tuple[int, ...],
    rollouts_per_task: int,
    data_manifest_path: Path,
    data_dir: Path,
    service_url: str,
    qdrant_url: str,
    local_config: LocalLoraConfig,
    judge_mode: JudgeMode,
    transaction_dir: Path,
    report_path: Path,
    require_sub_update: bool = False,
    tool_service: WideSeekLocalToolService | None = None,
    do_sample: bool = False,
    max_search_message_results: int = 3,
    max_search_content_characters: int = 3000,
    max_access_characters: int = 2000,
) -> dict[str, Any]:
    """Run one short real-model WideSeek cycle and emit only safe aggregates."""

    if rollouts_per_task < 2:
        raise ValueError("WideSeek train smoke requires at least two rollouts per task")
    manifest = load_asset_manifest(data_manifest_path)
    filename = f"{split}.jsonl"
    expected = next((file for file in manifest.files if file.path == filename), None)
    if expected is None or expected.sha256 is None:
        raise ValueError("selected WideSeek split is absent from the trusted manifest")
    dataset = load_wideseek_dataset(
        data_dir / filename,
        split=split,
        expected_sha256=expected.sha256,
        revision=manifest.revision,
    )
    if not task_indices or len(set(task_indices)) != len(task_indices):
        raise ValueError("task indices must be non-empty and unique")
    try:
        tasks = tuple(dataset.tasks[index] for index in task_indices)
    except IndexError:
        raise ValueError("WideSeek task index is out of range") from None

    policy_ids = (
        (PolicyId("shared"),) if topology == "shared" else (PolicyId("main"), PolicyId("sub"))
    )
    backend = LocalHfLoraBackend.from_pretrained(config=local_config, policy_ids=policy_ids)
    main_id = policy_ids[0]
    sub_id = main_id if topology == "shared" else PolicyId("sub")
    registry = PolicyRegistry(
        (
            RoleBinding(role="main", policy_id=main_id, trainable=True),
            RoleBinding(role="sub", policy_id=sub_id, trainable=True),
        ),
        tuple((policy_id, backend.rollout_revision(policy_id)) for policy_id in policy_ids),
    )
    environment_mode = "controlled-fixture" if tool_service is not None else "offline-pinned"
    tools = tool_service or WideSeekLocalToolService(
        WideSeekLocalConfig(service_url=service_url, qdrant_url=qdrant_url)
    )
    judge: MiniMaxSemanticJudge | None = None
    if judge_mode == "minimax-development":
        judge = MiniMaxSemanticJudge(
            MiniMaxChatClient(MiniMaxConfig.from_environment()),
            cache=SemanticJudgeCache(transaction_dir / "judge-cache.json"),
            max_provider_requests=128,
        )
    evaluator = WideSeekEvaluator(dataset, judge)
    outcome_reward = WideSeekRewardService(
        evaluator,
        WideSeekRewardConfig(
            spawn_cost=0.01,
            search_cost=0.001,
            token_cost=0.000001,
            invalid_action_cost=0.05,
        ),
    )
    reward = RewardComposer(outcome_reward, RewardConfig())
    sampling_params: tuple[tuple[str, JsonScalar], ...] = (
        (
            ("max_new_tokens", local_config.max_new_tokens),
            ("do_sample", True),
            ("temperature", 1.0),
            ("top_p", 1.0),
            ("top_k", 0),
        )
        if do_sample
        else (
            ("max_new_tokens", local_config.max_new_tokens),
            ("do_sample", False),
        )
    )
    orchestrator = WideSeekEpisodeOrchestrator(
        registry,
        {
            "main": backend.endpoint(main_id),
            "sub": backend.endpoint(sub_id),
        },
        {
            "main": backend.prompt_encoder,
            "sub": backend.prompt_encoder,
        },
        tools,
        max_concurrency=4,
        max_search_message_results=max_search_message_results,
        max_search_content_characters=max_search_content_characters,
        max_access_characters=max_access_characters,
        sampling_params=sampling_params,
    )
    config_digest = canonical_digest(
        {
            "topology": topology,
            "split": split,
            "task_ids": [task.task_id for task in tasks],
            "rollouts_per_task": rollouts_per_task,
            "model": local_config.model_dump(mode="json"),
            "sampling_params": sampling_params,
            "prompt_revision": orchestrator.prompt_revision,
            "environment_revision": tools.provider_revision,
            "reward_revision": reward.revision,
        }
    )
    context = PhaseTransactionContext(
        experiment_id=f"wideseek-train-smoke-{topology}",
        config_digest=config_digest,
        rng_state=_rng_state(),
        sampler_state=canonical_digest(
            {
                "task_ids": [task.task_id for task in tasks],
                "rollouts_per_task": rollouts_per_task,
            }
        ),
        dataset_revision=dataset.revision,
        corpus_revision=tools.identity.corpus_revision,
        tool_revision=WIDESEEK_TOOL_SCHEMA_REVISION,
        prompt_revision=orchestrator.prompt_revision,
        judge_revision=(
            canonical_digest(judge.revision) if judge is not None else "no-semantic-judge"
        ),
        environment_snapshot=tools.provider_revision,
        reward_revision=reward.revision,
    )
    initial_revisions = registry.snapshot()
    initial_hashes = {str(policy_id): backend.adapter_hash(policy_id) for policy_id in policy_ids}
    torch = importlib.import_module("torch")
    if str(local_config.device).startswith("cuda"):
        with torch.cuda.device(local_config.device):
            torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    runner = TrainableAlternatingCycleRunner(
        registry,
        backend,
        orchestrator,
        reward,
        rollouts_per_task=rollouts_per_task,
        transaction_store=FilePhaseTransactionStore(transaction_dir),
    )
    result = await runner.run_cycle(
        cycle_id=f"{split}-cycle-0",
        tasks=tasks,
        transaction_context=context,
    )
    elapsed = time.perf_counter() - started

    updates = tuple(
        update
        for update in (
            result.updates.joint_update,
            result.updates.main_update,
            result.updates.sub_update,
        )
        if update is not None
    )
    restored = {
        str(update.policy_id): (await backend.restore_checkpoint(update.checkpoint))
        == update.trained_version
        for update in updates
    }
    source_steps = {
        step.step_id: step
        for phase in result.phases
        for group in phase.groups
        for trace in group.traces
        for step in trace.model_steps
    }
    exact_round_trip = all(
        sample.response_ids == source_steps[sample.source_step_id].response_ids
        and sample.old_log_probs == source_steps[sample.source_step_id].response_log_probs
        for phase in result.phases
        for sample in phase.batch.samples
    )
    sub_updated = result.updates.joint_update is not None or result.updates.sub_update is not None
    if require_sub_update and not sub_updated:
        raise RuntimeError(
            "WideSeek train smoke required a Sub update but all Sub batches were empty"
        )
    final_revisions = registry.snapshot()
    final_hashes = {str(policy_id): backend.adapter_hash(policy_id) for policy_id in policy_ids}
    phase_reports = [
        {
            "phase": phase.phase,
            "tasks": len(phase.groups),
            "episodes": sum(len(group.traces) for group in phase.groups),
            "samples": len(phase.batch.samples),
            "zero_spawn_episodes": sum(
                trace.spawn_count == 0 for group in phase.groups for trace in group.traces
            ),
            "spawn_count": sum(
                trace.spawn_count for group in phase.groups for trace in group.traces
            ),
            "tool_calls": sum(
                len(trace.tool_outcomes) for group in phase.groups for trace in group.traces
            ),
            "degenerate_groups": phase.degenerate_groups,
            "batch_digest": phase.batch.batch_digest,
        }
        for phase in result.phases
    ]
    peak_bytes = 0
    gpu_name = None
    if str(local_config.device).startswith("cuda"):
        device_index = torch.device(local_config.device).index or 0
        gpu_name = torch.cuda.get_device_name(device_index)
        with torch.cuda.device(local_config.device):
            peak_bytes = int(torch.cuda.max_memory_allocated())
    report: dict[str, Any] = {
        "schema_revision": "heterospawn-wideseek-train-smoke-v1",
        "status": "passed",
        "comparable_to_official": False,
        "environment_mode": environment_mode,
        "topology": topology,
        "split": split,
        "task_ids": [str(task.task_id) for task in tasks],
        "rollouts_per_task": rollouts_per_task,
        "model_id": local_config.model_id,
        "model_revision": local_config.model_revision,
        "model_identity_kind": local_config.base_model_identity_kind,
        "model_identity": local_config.base_model_identity,
        "quantization": local_config.quantization,
        "gradient_checkpointing": local_config.gradient_checkpointing,
        "enable_thinking": local_config.enable_thinking,
        "sampling_params": dict(sampling_params),
        "sampling_logprob_semantics": "raw-policy",
        "tool_message_budgets": {
            "search_results": max_search_message_results,
            "search_content_characters": max_search_content_characters,
            "access_characters": max_access_characters,
        },
        "environment_revision": tools.provider_revision,
        "dataset_revision": dataset.revision,
        "reward_revision": reward.revision,
        "judge_mode": judge_mode,
        "judge_provider_requests": judge.provider_requests if judge is not None else 0,
        "elapsed_seconds": elapsed,
        "device": local_config.device,
        "gpu_name": gpu_name,
        "peak_allocated_vram_bytes": peak_bytes,
        "phases": phase_reports,
        "checks": {
            "exact_token_logprob_round_trip": exact_round_trip,
            "phase_commits_published": len(result.phase_commits) == len(result.phases),
            "checkpoint_restore": restored,
            "sub_updated_or_shared": sub_updated,
            "environment_bound_to_transactions": all(
                commit.input_digest for commit in result.phase_commits
            ),
        },
        "adapter_changed": {
            policy_id: initial_hashes[policy_id] != final_hashes[policy_id]
            for policy_id in initial_hashes
        },
        "versions": {
            "initial": [
                [str(policy_id), revision.model_dump(mode="json")]
                for policy_id, revision in initial_revisions
            ],
            "final": [
                [str(policy_id), revision.model_dump(mode="json")]
                for policy_id, revision in final_revisions
            ],
        },
        "updates": [
            {
                "policy_id": str(update.policy_id),
                "base_optimizer_step": update.base_version.optimizer_step,
                "trained_optimizer_step": update.trained_version.optimizer_step,
                "checkpoint_digest": update.checkpoint.weight_version.checkpoint_digest,
                "metrics": dict(update.metrics),
            }
            for update in updates
        ],
        "phase_commits": [
            {
                "transaction_id": commit.transaction_id,
                "phase": commit.phase_completed,
                "input_digest": commit.input_digest,
                "commit_digest": commit.manifest_digest,
            }
            for commit in result.phase_commits
        ],
    }
    if not exact_round_trip or not all(restored.values()):
        raise RuntimeError("WideSeek train smoke failed exact trajectory or restore validation")
    await asyncio.to_thread(_write_report, report_path, report)
    return report


def _rng_state() -> str:
    torch = importlib.import_module("torch")
    payload = {
        "python": repr(random.getstate()),
        "torch_cpu": torch.get_rng_state().tolist(),
        "torch_cuda": (
            [state.tolist() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }
    return base64.b64encode(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    ).decode()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
