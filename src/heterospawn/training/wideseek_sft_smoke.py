"""Opt-in QLoRA smoke for the role-targeted WideSeek supervised warm start."""

from __future__ import annotations

import asyncio
import gc
import importlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

from heterospawn.assets import load_asset_manifest
from heterospawn.backends.local_hf import LocalHfLoraBackend, LocalLoraConfig
from heterospawn.benchmarks.wideseek import WideSeekSplit, load_wideseek_dataset
from heterospawn.domain.ids import AgentInstanceId, EpisodeId, PolicyId, RolloutId
from heterospawn.domain.training import GenerationRequest, canonical_digest
from heterospawn.errors import ConfigurationError, RolloutRevisionMismatch
from heterospawn.training.wideseek_sft import (
    WideSeekRoleSftConstructor,
    build_supervised_training_batch,
    materialize_supervised_conversations,
)


async def run_wideseek_sft_smoke(
    *,
    split: WideSeekSplit,
    task_indices: tuple[int, ...],
    data_manifest_path: Path,
    data_dir: Path,
    local_config: LocalLoraConfig,
    report_path: Path,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Run one shared-policy SFT update without persisting plaintext examples."""

    torch = importlib.import_module("torch")
    if not str(local_config.device).startswith("cuda") or not torch.cuda.is_available():
        raise ConfigurationError("WideSeek SFT smoke requires an available CUDA device")
    if not task_indices or len(set(task_indices)) != len(task_indices):
        raise ValueError("SFT smoke task indices must be non-empty and unique")

    manifest = load_asset_manifest(data_manifest_path)
    filename = f"{split}.jsonl"
    expected = next(
        (file for file in manifest.files if file.path == filename and file.sha256 is not None),
        None,
    )
    if expected is None or expected.sha256 is None:
        raise ValueError(f"{split} is absent from the trusted manifest")
    dataset = load_wideseek_dataset(
        data_dir / filename,
        split=split,
        expected_sha256=expected.sha256,
        revision=manifest.revision,
    )
    construction = WideSeekRoleSftConstructor(max_workers=max_workers).build(
        dataset,
        task_indices=task_indices,
    )

    device_index = torch.device(local_config.device).index or 0
    with torch.cuda.device(local_config.device):
        torch.cuda.reset_peak_memory_stats()
        free_before, total_memory = torch.cuda.mem_get_info()
    started = time.perf_counter()
    policy_id = PolicyId("shared")
    backend = LocalHfLoraBackend.from_pretrained(
        config=local_config,
        policy_ids=(policy_id,),
    )
    base_revision = backend.rollout_revision(policy_id)
    train_hash_before = backend.adapter_hash(policy_id)
    rollout_hash_before = backend.adapter_hash(policy_id, rollout=True)
    examples = materialize_supervised_conversations(
        construction.conversations,
        backend.prompt_encoder,
    )
    del construction
    too_long = tuple(
        example.example_id
        for example in examples
        if (
            len(example.encoding.prompt_ids) + len(example.encoding.target_ids)
            > local_config.max_sequence_length
        )
    )
    if too_long:
        raise ValueError(f"{len(too_long)} supervised examples exceed max_sequence_length")
    batch = build_supervised_training_batch(
        batch_id=(
            "wideseek-sft:"
            + canonical_digest(
                {
                    "split": split,
                    "task_indices": task_indices,
                    "dataset_revision": dataset.revision,
                    "source_digest": dataset.source_digest,
                    "constructor_revision": examples[0].constructor_revision,
                }
            )
        ),
        target_policy_id=policy_id,
        expected_base_version=base_revision.weight_version,
        examples=examples,
    )
    update = await backend.update_supervised(
        policy_id,
        batch,
        base_revision.weight_version,
    )
    train_hash_after = backend.adapter_hash(policy_id)
    rollout_hash_before_sync = backend.adapter_hash(policy_id, rollout=True)
    replay = await backend.update_supervised(
        policy_id,
        batch,
        base_revision.weight_version,
    )
    synced_revision = await backend.sync_rollout_weights(
        policy_id,
        update.trained_version,
    )
    rollout_hash_after_sync = backend.adapter_hash(policy_id, rollout=True)
    stale_rejected = await _stale_revision_rejected(
        backend,
        policy_id,
        examples[0],
        base_revision,
    )
    checkpoint = update.checkpoint
    trained_hash = backend.adapter_hash(policy_id)
    first_deployment_id = synced_revision.deployment_id
    del backend
    gc.collect()
    with torch.cuda.device(local_config.device):
        torch.cuda.empty_cache()

    replacement = LocalHfLoraBackend.from_pretrained(
        config=local_config,
        policy_ids=(policy_id,),
    )
    restored_version = await replacement.restore_checkpoint(checkpoint)
    restored_train_hash = replacement.adapter_hash(policy_id)
    recovered_revision = await replacement.sync_rollout_weights(
        policy_id,
        restored_version,
    )
    restored_rollout_hash = replacement.adapter_hash(policy_id, rollout=True)
    elapsed = time.perf_counter() - started
    with torch.cuda.device(local_config.device):
        peak_bytes = int(torch.cuda.max_memory_allocated())

    behaviors = {example.behavior for example in examples}
    checks = {
        "both_behaviors_present": behaviors == {"main_final", "sub_summary"},
        "target_only_masks": all(
            example.loss_mask
            == (0,) * len(example.encoding.prompt_ids) + (1,) * len(example.encoding.target_ids)
            for example in examples
        ),
        "train_adapter_changed": train_hash_after != train_hash_before,
        "rollout_unchanged_before_sync": rollout_hash_before_sync == rollout_hash_before,
        "idempotent_replay": replay == update
        and replay.trained_version.optimizer_step
        == base_revision.weight_version.optimizer_step + 1,
        "sync_matches_train": rollout_hash_after_sync == train_hash_after,
        "stale_revision_rejected": stale_rejected,
        "checkpoint_restored": restored_version == update.trained_version
        and restored_train_hash == trained_hash,
        "replacement_sync_matches_train": restored_rollout_hash == restored_train_hash,
        "replacement_deployment_is_fresh": recovered_revision.deployment_id != first_deployment_id,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"WideSeek SFT smoke failed: {', '.join(failed)}")

    report: dict[str, Any] = {
        "schema_revision": "heterospawn-wideseek-sft-smoke-v1",
        "status": "passed",
        "comparable_to_official": False,
        "training_scope": "single-shared-policy-optimizer-step",
        "dataset_revision": dataset.revision,
        "source_digest": dataset.source_digest,
        "split": split,
        "task_indices": list(task_indices),
        "constructor_revision": examples[0].constructor_revision,
        "batch_digest": batch.batch_digest,
        "example_count": len(examples),
        "role_token_stats": _role_token_stats(examples),
        "model_id": local_config.model_id,
        "model_revision": local_config.model_revision,
        "model_identity_kind": local_config.base_model_identity_kind,
        "model_identity": local_config.base_model_identity,
        "device": local_config.device,
        "gpu_name": torch.cuda.get_device_name(device_index),
        "total_vram_bytes": int(total_memory),
        "free_vram_before_load_bytes": int(free_before),
        "peak_allocated_vram_bytes": peak_bytes,
        "elapsed_seconds": elapsed,
        "runtime_versions": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
            "peft": importlib.metadata.version("peft"),
            "accelerate": importlib.metadata.version("accelerate"),
            "safetensors": importlib.metadata.version("safetensors"),
            "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        },
        "versions": {
            "base": base_revision.model_dump(mode="json"),
            "trained": update.trained_version.model_dump(mode="json"),
            "synced": synced_revision.model_dump(mode="json"),
            "recovered": recovered_revision.model_dump(mode="json"),
        },
        "checkpoint": {
            "checkpoint_id": checkpoint.checkpoint_id,
            "optimizer_state_digest": checkpoint.optimizer_state_digest,
        },
        "metrics": dict(update.metrics),
        "checks": checks,
        "report_excludes": [
            "questions",
            "reference_answers",
            "constructed_conversations",
            "token_ids",
            "checkpoint_uri",
        ],
    }
    await asyncio.to_thread(_write_report, report_path, report)
    return report


async def _stale_revision_rejected(
    backend: LocalHfLoraBackend,
    policy_id: PolicyId,
    example: Any,
    stale_revision: Any,
) -> bool:
    request = GenerationRequest(
        task_id=example.task_id,
        episode_id=EpisodeId("sft-stale-probe"),
        rollout_id=RolloutId("sft-stale-probe"),
        request_id="sft-stale-probe",
        agent_role=example.agent_role,
        agent_instance_id=AgentInstanceId("sft-stale-probe"),
        prompt_ids=example.encoding.prompt_ids,
        tokenizer_revision=example.encoding.tokenizer_revision,
        prompt_template_revision=example.encoding.prompt_template_revision,
        sampling_params=(("max_new_tokens", 1), ("do_sample", False)),
    )
    try:
        await backend.endpoint(policy_id).generate(request, stale_revision)
    except RolloutRevisionMismatch:
        return True
    return False


def _role_token_stats(examples: tuple[Any, ...]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for behavior in ("main_final", "sub_summary"):
        selected = tuple(example for example in examples if example.behavior == behavior)
        if not selected:
            continue
        combined_lengths = tuple(
            len(example.encoding.prompt_ids) + len(example.encoding.target_ids)
            for example in selected
        )
        target_lengths = tuple(len(example.encoding.target_ids) for example in selected)
        result[behavior] = {
            "examples": len(selected),
            "combined_tokens": sum(combined_lengths),
            "target_tokens": sum(target_lengths),
            "max_sequence_tokens": max(combined_lengths),
        }
    return result


def _write_report(path: Path, report: dict[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
