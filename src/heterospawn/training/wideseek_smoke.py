"""Credential-safe WideSeek rollout and LocalHF short-training validation."""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import random
import time
from pathlib import Path
from typing import Any, Literal

from heterospawn.assets import load_asset_manifest
from heterospawn.backends.local_hf.backend import LocalHfLoraBackend
from heterospawn.backends.local_hf.config import LocalLoraConfig
from heterospawn.benchmarks.wideseek import WideSeekSplit, load_wideseek_dataset
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
from heterospawn.evaluation.wideseek import WideSeekEvaluator
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
