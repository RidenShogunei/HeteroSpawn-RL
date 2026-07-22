"""Opt-in real-GPU conformance run with credential-safe reporting."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

from heterospawn.backends.local_hf.backend import LocalHfLoraBackend
from heterospawn.backends.local_hf.config import LocalLoraConfig
from heterospawn.domain.ids import AgentInstanceId, EpisodeId, PolicyId, RolloutId, StepId, TaskId
from heterospawn.domain.training import GenerationRequest, GenerationResult, TrajectoryStep
from heterospawn.domain.versions import RolloutRevision
from heterospawn.errors import ConfigurationError, RolloutRevisionMismatch
from heterospawn.policies.base import Message
from heterospawn.training import TrainingBatchBuilder


async def run_local_contract_smoke(
    *,
    config: LocalLoraConfig,
    report_path: Path,
) -> dict[str, Any]:
    """Exercise two independent policies without retaining generated text."""

    torch = importlib.import_module("torch")
    if not str(config.device).startswith("cuda") or not torch.cuda.is_available():
        raise ConfigurationError("local GPU smoke requires an available CUDA device")
    device_index = torch.device(config.device).index or 0
    with torch.cuda.device(config.device):
        torch.cuda.reset_peak_memory_stats()
        free_before, total_memory = torch.cuda.mem_get_info()
    started = time.perf_counter()
    main = PolicyId("main")
    sub = PolicyId("sub")
    backend = LocalHfLoraBackend.from_pretrained(
        config=config,
        policy_ids=(main, sub),
    )
    main_r0 = backend.rollout_revision(main)
    sub_r0 = backend.rollout_revision(sub)
    main_train_h0 = backend.adapter_hash(main)
    main_rollout_h0 = backend.adapter_hash(main, rollout=True)
    sub_train_h0 = backend.adapter_hash(sub)
    sub_rollout_h0 = backend.adapter_hash(sub, rollout=True)

    main_request = _request(
        backend,
        policy_id=main,
        role="main",
        episode_index=0,
        prompt="Return a short JSON object with one key named status.",
    )
    main_result = await backend.endpoint(main).generate(main_request, main_r0)
    main_step = _step(main_request, main_result, StepId("main-step-0"))
    main_batch = TrainingBatchBuilder().build(
        batch_id="gpu-main-update-1",
        phase="main_update",
        target_policy_id=main,
        expected_base_version=main_r0.weight_version,
        steps=(main_step,),
        episode_advantages={main_request.episode_id: 1.0},
    )
    main_update = await backend.update_policy(main, main_batch, main_r0.weight_version)
    main_train_h1 = backend.adapter_hash(main)
    main_rollout_before_sync = backend.adapter_hash(main, rollout=True)
    main_r1 = await backend.sync_rollout_weights(main, main_update.trained_version)
    sub_train_after_main = backend.adapter_hash(sub)
    sub_rollout_after_main = backend.adapter_hash(sub, rollout=True)
    stale_rejected = False
    try:
        await backend.endpoint(main).generate(main_request, main_r0)
    except RolloutRevisionMismatch:
        stale_rejected = True

    fresh_main_request = _request(
        backend,
        policy_id=main,
        role="main",
        episode_index=1,
        prompt="Return a short JSON object with one key named phase.",
    )
    fresh_main_result = await backend.endpoint(main).generate(fresh_main_request, main_r1)
    main_train_before_sub = backend.adapter_hash(main)
    main_rollout_before_sub = backend.adapter_hash(main, rollout=True)

    sub_requests = tuple(
        _request(
            backend,
            policy_id=sub,
            role="sub",
            episode_index=index,
            prompt=f"Return the integer {index} in a short JSON object.",
        )
        for index in range(4)
    )
    sub_results = await asyncio.gather(
        *(backend.endpoint(sub).generate(request, sub_r0) for request in sub_requests)
    )
    sub_steps = tuple(
        _step(request, result, StepId(f"sub-step-{index}"), partner=main_r1)
        for index, (request, result) in enumerate(zip(sub_requests, sub_results, strict=True))
    )
    sub_batch = TrainingBatchBuilder().build(
        batch_id="gpu-sub-update-1",
        phase="sub_update",
        target_policy_id=sub,
        expected_base_version=sub_r0.weight_version,
        steps=sub_steps,
        episode_advantages={request.episode_id: 1.0 for request in sub_requests},
    )
    sub_update = await backend.update_policy(sub, sub_batch, sub_r0.weight_version)
    sub_train_h1 = backend.adapter_hash(sub)
    sub_rollout_before_sync = backend.adapter_hash(sub, rollout=True)
    sub_r1 = await backend.sync_rollout_weights(sub, sub_update.trained_version)
    restored = await backend.restore_checkpoint(sub_update.checkpoint)

    elapsed = time.perf_counter() - started
    with torch.cuda.device(config.device):
        peak_bytes = int(torch.cuda.max_memory_allocated())
    checks = {
        "main_token_logprob_alignment": len(main_result.response_ids)
        == len(main_result.response_log_probs),
        "fresh_main_token_logprob_alignment": len(fresh_main_result.response_ids)
        == len(fresh_main_result.response_log_probs),
        "four_subs_share_revision": len(sub_results) == 4
        and all(result.rollout_revision == sub_r0 for result in sub_results),
        "main_train_changed": main_train_h1 != main_train_h0,
        "main_rollout_unchanged_before_sync": main_rollout_before_sync == main_rollout_h0,
        "sub_unchanged_during_main_update": sub_train_after_main == sub_train_h0
        and sub_rollout_after_main == sub_rollout_h0,
        "stale_main_revision_rejected": stale_rejected,
        "main_sync_matches_train": backend.adapter_hash(main, rollout=True)
        == backend.adapter_hash(main),
        "sub_train_changed": sub_train_h1 != sub_train_h0,
        "sub_rollout_unchanged_before_sync": sub_rollout_before_sync == sub_rollout_h0,
        "sub_sync_matches_train": backend.adapter_hash(sub, rollout=True)
        == backend.adapter_hash(sub),
        "main_unchanged_during_sub_update": backend.adapter_hash(main) == main_train_before_sub
        and backend.adapter_hash(main, rollout=True) == main_rollout_before_sub,
        "checkpoint_restored": restored == sub_update.trained_version,
        "main_revision_advanced": main_r1.replica_set_revision == main_r0.replica_set_revision + 1,
        "sub_revision_advanced": sub_r1.replica_set_revision == sub_r0.replica_set_revision + 1,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"local contract smoke failed: {', '.join(failed)}")
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "reference_only": True,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "model_source": "verified_local_path" if config.model_path is not None else "pinned_hub",
        "model_weight_sha256": config.expected_model_weight_sha256,
        "device": config.device,
        "gpu_name": torch.cuda.get_device_name(device_index),
        "total_vram_bytes": int(total_memory),
        "free_vram_before_load_bytes": int(free_before),
        "peak_allocated_vram_bytes": peak_bytes,
        "elapsed_seconds": elapsed,
        "dtype": config.dtype,
        "runtime_versions": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
            "peft": importlib.metadata.version("peft"),
            "accelerate": importlib.metadata.version("accelerate"),
            "safetensors": importlib.metadata.version("safetensors"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "lora": {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": list(config.lora_target_modules),
        },
        "token_counts": {
            "main_initial": len(main_result.response_ids),
            "main_fresh": len(fresh_main_result.response_ids),
            "subs": [len(result.response_ids) for result in sub_results],
        },
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
        "checks": checks,
        "metrics": {
            "main": dict(main_update.metrics),
            "sub": dict(sub_update.metrics),
        },
    }
    await asyncio.to_thread(_write_report, report_path, report)
    return report


def _request(
    backend: LocalHfLoraBackend,
    *,
    policy_id: PolicyId,
    role: str,
    episode_index: int,
    prompt: str,
) -> GenerationRequest:
    encoding = backend.prompt_encoder.encode(
        (
            Message(role="system", content="You are a concise contract-test model."),
            Message(role="user", content=prompt),
        )
    )
    return GenerationRequest(
        task_id=TaskId("local-contract-smoke"),
        episode_id=EpisodeId(f"{role}-episode-{episode_index}"),
        rollout_id=RolloutId(f"{role}-rollout-{episode_index}"),
        request_id=f"{policy_id}-request-{episode_index}",
        agent_role="main" if role == "main" else "sub",
        agent_instance_id=AgentInstanceId(f"{role}-{episode_index}"),
        prompt_ids=encoding.prompt_ids,
        tokenizer_revision=encoding.tokenizer_revision,
        prompt_template_revision=encoding.prompt_template_revision,
        sampling_params=(("max_new_tokens", backend.config.max_new_tokens), ("do_sample", False)),
    )


def _step(
    request: GenerationRequest,
    result: GenerationResult,
    step_id: StepId,
    *,
    partner: RolloutRevision | None = None,
) -> TrajectoryStep:
    partners = () if partner is None else (partner,)
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
        partner_rollout_revisions=partners,
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
