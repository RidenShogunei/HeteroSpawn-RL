"""Opt-in LocalHF training plus standalone vLLM rollout contract smoke."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

from heterospawn.backends.local_hf import LocalHfLoraBackend, LocalLoraConfig
from heterospawn.backends.vllm_rollout.backend import VllmRolloutBackend
from heterospawn.backends.vllm_rollout.models import (
    VllmPolicyDeployment,
    VllmRolloutConfig,
)
from heterospawn.backends.vllm_rollout.process import SubprocessVllmWorkerFactory
from heterospawn.benchmarks.xbench import BenchmarkTask
from heterospawn.domain.ids import (
    AgentInstanceId,
    EpisodeId,
    PolicyId,
    RolloutId,
    StepId,
    TaskId,
)
from heterospawn.domain.training import GenerationRequest, GenerationResult, TrajectoryStep
from heterospawn.domain.versions import RoleBinding, RolloutRevision
from heterospawn.errors import ConfigurationError, RolloutRevisionMismatch
from heterospawn.orchestration.trainable_episode import TrainableEpisodeOrchestrator
from heterospawn.orchestration.trainable_models import TrainableEpisodeTrace
from heterospawn.policies.base import Message
from heterospawn.search.mock import MockSearchService
from heterospawn.training import (
    PhaseRolloutResult,
    PolicyRegistry,
    RewardComposer,
    RewardConfig,
    TrainableAlternatingCycleRunner,
    TrainableCycleResult,
    TrainingBatchBuilder,
)


class _ContractOutcomeReward:
    """Non-scientific reward used only to exercise the optimizer contract."""

    revision = "vllm-trainable-cycle-contract-v1"

    async def score(self, task: BenchmarkTask, trace: TrainableEpisodeTrace) -> float:
        del task
        return 0.0 if str(trace.episode_id).endswith("episode-0") else 1.0


async def run_vllm_rollout_contract_smoke(
    *,
    local_config: LocalLoraConfig,
    vllm_python: Path,
    main_rollout_device: str,
    sub_rollout_device: str,
    runtime_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Prove exact rollout and restart synchronization without retaining model text."""

    if local_config.model_path is None:
        raise ConfigurationError("vLLM rollout smoke requires a verified local model path")
    if main_rollout_device == sub_rollout_device:
        raise ConfigurationError("Main and Sub vLLM workers require distinct GPU assignments")
    torch = importlib.import_module("torch")
    if not str(local_config.device).startswith("cuda") or not torch.cuda.is_available():
        raise ConfigurationError("vLLM rollout smoke requires an available training GPU")

    started = time.perf_counter()
    main = PolicyId("main")
    sub = PolicyId("sub")
    trainer = LocalHfLoraBackend.from_pretrained(
        config=local_config,
        policy_ids=(main, sub),
    )
    rollout_config = VllmRolloutConfig(
        python_executable=vllm_python,
        model_path=local_config.model_path,
        expected_model_weight_sha256=local_config.expected_model_weight_sha256,
        model_revision=local_config.model_revision,
        tokenizer_revision=trainer.prompt_encoder.tokenizer_revision,
        prompt_template_revision=trainer.prompt_encoder.prompt_template_revision,
        runtime_dir=runtime_dir,
        max_model_len=min(local_config.max_sequence_length, 512),
        max_new_tokens=min(local_config.max_new_tokens, 32),
        startup_timeout_seconds=180.0,
    )
    backend = await VllmRolloutBackend.create(
        config=rollout_config,
        training_backend=trainer,
        artifact_provider=trainer,
        deployments=(
            VllmPolicyDeployment(
                policy_id=main,
                cuda_device=main_rollout_device,
                deployment_id="main-vllm-rollout",
            ),
            VllmPolicyDeployment(
                policy_id=sub,
                cuda_device=sub_rollout_device,
                deployment_id="sub-vllm-rollout",
            ),
        ),
        worker_factory=SubprocessVllmWorkerFactory(),
    )
    try:
        report = await _run_contract(
            backend=backend,
            trainer=trainer,
            main=main,
            sub=sub,
            local_config=local_config,
            main_rollout_device=main_rollout_device,
            sub_rollout_device=sub_rollout_device,
            started=started,
        )
    finally:
        await backend.close()
    await asyncio.to_thread(_write_report, report_path, report)
    return report


async def _run_contract(
    *,
    backend: VllmRolloutBackend,
    trainer: LocalHfLoraBackend,
    main: PolicyId,
    sub: PolicyId,
    local_config: LocalLoraConfig,
    main_rollout_device: str,
    sub_rollout_device: str,
    started: float,
) -> dict[str, Any]:
    main_r0 = backend.rollout_revision(main)
    sub_r0 = backend.rollout_revision(sub)
    main_train_h0 = trainer.adapter_hash(main)
    sub_train_h0 = trainer.adapter_hash(sub)

    main_request = _request(
        trainer,
        role="main",
        index=0,
        prompt="Return a short JSON object with one key named status.",
    )
    main_result = await backend.endpoint(main).generate(main_request, main_r0)
    main_batch = TrainingBatchBuilder().build(
        batch_id="vllm-main-update-1",
        phase="main_update",
        target_policy_id=main,
        expected_base_version=main_r0.weight_version,
        steps=(_step(main_request, main_result, StepId("main-step-0")),),
        episode_advantages={main_request.episode_id: 1.0},
    )
    main_update = await backend.update_policy(main, main_batch, main_r0.weight_version)
    main_train_h1 = trainer.adapter_hash(main)
    sub_after_main = trainer.adapter_hash(sub)
    old_worker_result = await backend.endpoint(main).generate(main_request, main_r0)
    main_r1 = await backend.sync_rollout_weights(main, main_update.trained_version)
    stale_main_rejected = await _stale_rejected(backend, main, main_request, main_r0)

    fresh_main_request = _request(
        trainer,
        role="main",
        index=1,
        prompt="Return a short JSON object with one key named phase.",
    )
    fresh_main_result = await backend.endpoint(main).generate(fresh_main_request, main_r1)
    main_before_sub = trainer.adapter_hash(main)

    sub_requests = tuple(
        _request(
            trainer,
            role="sub",
            index=index,
            prompt=f"Return the integer {index} in a short JSON object.",
        )
        for index in range(4)
    )
    sub_results = await asyncio.gather(
        *(backend.endpoint(sub).generate(request, sub_r0) for request in sub_requests)
    )
    sub_steps = tuple(
        _step(
            request,
            result,
            StepId(f"sub-step-{index}"),
            partner=main_r1,
        )
        for index, (request, result) in enumerate(zip(sub_requests, sub_results, strict=True))
    )
    sub_batch = TrainingBatchBuilder().build(
        batch_id="vllm-sub-update-1",
        phase="sub_update",
        target_policy_id=sub,
        expected_base_version=sub_r0.weight_version,
        steps=sub_steps,
        episode_advantages={request.episode_id: 1.0 for request in sub_requests},
    )
    sub_update = await backend.update_policy(sub, sub_batch, sub_r0.weight_version)
    sub_train_h1 = trainer.adapter_hash(sub)
    main_after_sub = trainer.adapter_hash(main)
    sub_r1 = await backend.sync_rollout_weights(sub, sub_update.trained_version)
    stale_sub_rejected = await _stale_rejected(
        backend,
        sub,
        sub_requests[0],
        sub_r0,
    )
    fresh_sub_result = await backend.endpoint(sub).generate(sub_requests[0], sub_r1)
    restored = await backend.restore_checkpoint(sub_update.checkpoint)
    cycle_base = {
        "main": backend.rollout_revision(main),
        "sub": backend.rollout_revision(sub),
    }
    cycle = await _run_trainable_episode_cycle(
        backend=backend,
        trainer=trainer,
        main=main,
        sub=sub,
    )
    cycle_main_phase, cycle_sub_phase = cycle.phases
    cycle_checks = {
        "trainable_cycle_main_updated": cycle.updates.main_update is not None,
        "trainable_cycle_sub_updated": cycle.updates.sub_update is not None,
        "trainable_cycle_two_rollouts_per_phase": all(
            len(group.traces) == 2 for phase in cycle.phases for group in phase.groups
        ),
        "trainable_cycle_required_one_sub_path": all(
            trace.status == "success" and trace.spawn_count == 1
            for phase in cycle.phases
            for group in phase.groups
            for trace in group.traces
        ),
        "trainable_cycle_fresh_rollout_ids": {
            trace.rollout_id for group in cycle_main_phase.groups for trace in group.traces
        }.isdisjoint(
            {trace.rollout_id for group in cycle_sub_phase.groups for trace in group.traces}
        ),
        "trainable_cycle_exact_batch_roundtrip": all(
            _phase_batch_matches_raw_steps(phase) for phase in cycle.phases
        ),
        "trainable_cycle_episode_balanced": all(
            _phase_is_episode_balanced(phase) for phase in cycle.phases
        ),
        "trainable_cycle_main_first_versions": dict(cycle_main_phase.policy_revisions)[main]
        == cycle_base["main"],
        "trainable_cycle_sub_phase_is_fresh": (
            dict(cycle_sub_phase.policy_revisions)[main].weight_version.optimizer_step
            == cycle_base["main"].weight_version.optimizer_step + 1
            and dict(cycle_sub_phase.policy_revisions)[sub] == cycle_base["sub"]
        ),
    }
    runtime_metrics = {
        "main": (await backend.runtime_metrics(main)).model_dump(mode="json"),
        "sub": (await backend.runtime_metrics(sub)).model_dump(mode="json"),
    }

    checks = {
        "main_exact_token_logprob_alignment": _aligned(main_result),
        "old_worker_serves_old_revision_before_sync": old_worker_result.rollout_revision == main_r0,
        "main_train_changed": main_train_h1 != main_train_h0,
        "sub_unchanged_during_main_update": sub_after_main == sub_train_h0,
        "main_sync_advanced_revision": main_r1.replica_set_revision
        == main_r0.replica_set_revision + 1,
        "stale_main_revision_rejected": stale_main_rejected,
        "fresh_main_exact_token_logprob_alignment": _aligned(fresh_main_result),
        "four_subs_share_one_revision": len(sub_results) == 4
        and all(result.rollout_revision == sub_r0 for result in sub_results),
        "four_subs_exact_token_logprob_alignment": all(_aligned(result) for result in sub_results),
        "sub_train_changed": sub_train_h1 != sub_train_h0,
        "main_unchanged_during_sub_update": main_after_sub == main_before_sub,
        "sub_sync_advanced_revision": sub_r1.replica_set_revision
        == sub_r0.replica_set_revision + 1,
        "stale_sub_revision_rejected": stale_sub_rejected,
        "fresh_sub_exact_token_logprob_alignment": _aligned(fresh_sub_result),
        "checkpoint_restored": restored == sub_update.trained_version,
        **cycle_checks,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"vLLM rollout contract smoke failed: {', '.join(failed)}")
    return {
        "schema_version": 2,
        "status": "passed",
        "rollout_only": False,
        "training_backend": "local_hf_lora",
        "rollout_backend": "vllm-0.7.0-v0-xformers",
        "model_id": local_config.model_id,
        "model_revision": local_config.model_revision,
        "model_weight_sha256": local_config.expected_model_weight_sha256,
        "training_device": local_config.device,
        "rollout_devices": {
            "main": main_rollout_device,
            "sub": sub_rollout_device,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "runtime_versions": {
            "torch": torch_version(),
            "transformers": importlib.metadata.version("transformers"),
            "peft": importlib.metadata.version("peft"),
            "vllm": importlib.metadata.version("vllm"),
        },
        "token_counts": {
            "main_initial": len(main_result.response_ids),
            "main_fresh": len(fresh_main_result.response_ids),
            "subs": [len(result.response_ids) for result in sub_results],
            "sub_fresh": len(fresh_sub_result.response_ids),
        },
        "worker_runtime": runtime_metrics,
        "versions": {
            "main": {
                "before": main_r0.model_dump(mode="json"),
                "after": main_r1.model_dump(mode="json"),
            },
            "sub": {
                "before": sub_r0.model_dump(mode="json"),
                "after": sub_r1.model_dump(mode="json"),
            },
        },
        "trainable_episode_cycle": {
            "reward_revision": _ContractOutcomeReward.revision,
            "scientific_reward": False,
            "rollouts_per_task": 2,
            "phases": [
                {
                    "phase": phase.phase,
                    "sample_count": len(phase.batch.samples),
                    "degenerate_groups": phase.degenerate_groups,
                    "trace_count": sum(len(group.traces) for group in phase.groups),
                    "invalid_main_attempts": sum(
                        trace.invalid_main_attempts
                        for group in phase.groups
                        for trace in group.traces
                    ),
                    "base_versions": {
                        str(policy_id): revision.weight_version.model_dump(mode="json")
                        for policy_id, revision in phase.policy_revisions
                    },
                }
                for phase in cycle.phases
            ],
        },
        "checks": checks,
    }


async def _run_trainable_episode_cycle(
    *,
    backend: VllmRolloutBackend,
    trainer: LocalHfLoraBackend,
    main: PolicyId,
    sub: PolicyId,
) -> TrainableCycleResult:
    registry = PolicyRegistry(
        (
            RoleBinding(role="main", policy_id=main, trainable=True),
            RoleBinding(role="sub", policy_id=sub, trainable=True),
        ),
        (
            (main, backend.rollout_revision(main)),
            (sub, backend.rollout_revision(sub)),
        ),
    )
    orchestrator = TrainableEpisodeOrchestrator(
        registry,
        {"main": backend.endpoint(main), "sub": backend.endpoint(sub)},
        {"main": trainer.prompt_encoder, "sub": trainer.prompt_encoder},
        MockSearchService({"capital of France": "Paris is the capital of France."}),
        max_concurrency=2,
        max_spawn_per_episode=1,
        repair_attempts=2,
        sampling_params=(("max_new_tokens", 32), ("do_sample", False)),
        main_initial_sampling_params=(
            ("max_new_tokens", 32),
            ("do_sample", False),
            (
                "guided_regex",
                r'\{"kind":"spawn","subtasks":\["capital of France"\]\}',
            ),
        ),
        main_final_sampling_params=(
            ("max_new_tokens", 32),
            ("do_sample", False),
            ("guided_regex", r'\{"kind":"answer","answer":"Paris"\}'),
        ),
    )
    return await TrainableAlternatingCycleRunner(
        registry,
        backend,
        orchestrator,
        RewardComposer(_ContractOutcomeReward(), RewardConfig()),
        rollouts_per_task=2,
    ).run_cycle(
        cycle_id="vllm-trainable-cycle-1",
        tasks=(
            BenchmarkTask(
                task_id=TaskId("vllm-trainable-cycle-task"),
                prompt=(
                    "This is a strict orchestration contract check. On the initial turn, "
                    'return exactly {"kind":"spawn","subtasks":["capital of France"]}. '
                    "After the Sub result arrives, return exactly one ANSWER JSON object."
                ),
            ),
        ),
    )


def _phase_batch_matches_raw_steps(phase: PhaseRolloutResult) -> bool:
    raw = {
        step.step_id: step
        for group in phase.groups
        for trace in group.traces
        for step in trace.model_steps
    }
    return all(
        sample.source_step_id in raw
        and sample.prompt_ids == raw[sample.source_step_id].prompt_ids
        and sample.response_ids == raw[sample.source_step_id].response_ids
        and sample.old_log_probs == raw[sample.source_step_id].response_log_probs
        for sample in phase.batch.samples
    )


def _phase_is_episode_balanced(phase: PhaseRolloutResult) -> bool:
    episode_ids = {sample.episode_id for sample in phase.batch.samples}
    return all(
        abs(
            sum(
                sample.aggregation_weight
                for sample in phase.batch.samples
                if sample.episode_id == episode_id
            )
            - 1.0
        )
        < 1e-12
        for episode_id in episode_ids
    )


def torch_version() -> str:
    torch = importlib.import_module("torch")
    return str(torch.__version__)


async def _stale_rejected(
    backend: VllmRolloutBackend,
    policy_id: PolicyId,
    request: GenerationRequest,
    revision: RolloutRevision,
) -> bool:
    try:
        await backend.endpoint(policy_id).generate(request, revision)
    except RolloutRevisionMismatch:
        return True
    return False


def _aligned(result: GenerationResult) -> bool:
    return bool(result.response_ids) and len(result.response_ids) == len(result.response_log_probs)


def _request(
    trainer: LocalHfLoraBackend,
    *,
    role: str,
    index: int,
    prompt: str,
) -> GenerationRequest:
    encoding = trainer.prompt_encoder.encode(
        (
            Message(role="system", content="You are a concise contract-test model."),
            Message(role="user", content=prompt),
        )
    )
    return GenerationRequest(
        task_id=TaskId("vllm-rollout-contract-smoke"),
        episode_id=EpisodeId(f"{role}-episode-{index}"),
        rollout_id=RolloutId(f"{role}-rollout-{index}"),
        request_id=f"{role}-request-{index}",
        agent_role="main" if role == "main" else "sub",
        agent_instance_id=AgentInstanceId(f"{role}-{index}"),
        prompt_ids=encoding.prompt_ids,
        tokenizer_revision=encoding.tokenizer_revision,
        prompt_template_revision=encoding.prompt_template_revision,
        sampling_params=(("max_new_tokens", 16), ("do_sample", False)),
    )


def _step(
    request: GenerationRequest,
    result: GenerationResult,
    step_id: StepId,
    *,
    partner: RolloutRevision | None = None,
) -> TrajectoryStep:
    return TrajectoryStep(
        task_id=request.task_id,
        episode_id=request.episode_id,
        rollout_id=request.rollout_id,
        step_id=step_id,
        event_index=0,
        agent_role=request.agent_role,
        agent_instance_id=request.agent_instance_id,
        policy_id=result.policy_id,
        rollout_revision=result.rollout_revision,
        partner_rollout_revisions=() if partner is None else (partner,),
        prompt_ids=request.prompt_ids,
        response_ids=result.response_ids,
        response_log_probs=result.response_log_probs,
        tokenizer_revision=request.tokenizer_revision,
        prompt_template_revision=request.prompt_template_revision,
        sampling_params=request.sampling_params,
        stop_reason=result.stop_reason,
    )


def _write_report(report_path: Path, report: dict[str, Any]) -> None:
    destination = report_path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
