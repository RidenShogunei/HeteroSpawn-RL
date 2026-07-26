"""Opt-in QLoRA smoke for the role-targeted WideSeek supervised warm start."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import importlib
import importlib.metadata
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from heterospawn.assets import load_asset_manifest
from heterospawn.backends.local_hf import LocalHfLoraBackend, LocalLoraConfig
from heterospawn.benchmarks.wideseek import WideSeekSplit, load_wideseek_dataset
from heterospawn.domain.ids import AgentInstanceId, EpisodeId, PolicyId, RolloutId
from heterospawn.domain.supervised import SupervisedTrainingExample
from heterospawn.domain.training import GenerationRequest, canonical_digest
from heterospawn.errors import ConfigurationError, RolloutRevisionMismatch
from heterospawn.training.wideseek_sft import (
    WideSeekRoleSftConstructor,
    build_supervised_training_batch,
    materialize_supervised_conversations,
)


@dataclass(frozen=True)
class _TaskExamples:
    task_index: int
    examples: tuple[SupervisedTrainingExample, ...]


async def run_wideseek_sft_smoke(
    *,
    split: WideSeekSplit,
    task_indices: tuple[int, ...],
    data_manifest_path: Path,
    data_dir: Path,
    local_config: LocalLoraConfig,
    report_path: Path,
    max_workers: int = 4,
    compliance_report_path: Path | None = None,
    service_url: str = "http://127.0.0.1:8000",
    qdrant_url: str = "http://127.0.0.1:6333",
    task_limit: int = 1,
    tasks_per_step: int | None = None,
    epochs: int = 1,
    training_max_sequence_length: int | None = None,
    exclude_compliance_selection: bool = False,
) -> dict[str, Any]:
    """Run a bounded shared-policy SFT schedule without persisting plaintext examples."""

    if task_indices and len(set(task_indices)) != len(task_indices):
        raise ValueError("SFT task indices must be unique")
    if not task_indices and task_limit < 1:
        raise ValueError("SFT task_limit must be positive")
    if tasks_per_step is not None and tasks_per_step < 1:
        raise ValueError("SFT tasks_per_step must be positive")
    if epochs < 1:
        raise ValueError("SFT epochs must be positive")
    training_sequence_limit = (
        local_config.max_sequence_length
        if training_max_sequence_length is None
        else training_max_sequence_length
    )
    if not 16 <= training_sequence_limit <= local_config.max_sequence_length:
        raise ValueError("training_max_sequence_length must be in 16..backend max_sequence_length")
    heldout_isolation = exclude_compliance_selection or compliance_report_path is not None
    if heldout_isolation and task_indices:
        _validate_compliance_selection(split, task_indices)
    torch = importlib.import_module("torch")
    if not str(local_config.device).startswith("cuda") or not torch.cuda.is_available():
        raise ConfigurationError("WideSeek SFT smoke requires an available CUDA device")

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
    task_groups, skipped_overlength = _materialize_task_groups(
        dataset=dataset,
        explicit_task_indices=task_indices,
        task_limit=task_limit,
        constructor=WideSeekRoleSftConstructor(max_workers=max_workers),
        codec=backend.prompt_encoder,
        training_max_sequence_length=training_sequence_limit,
        exclude_compliance=heldout_isolation,
    )
    selected_task_indices = tuple(group.task_index for group in task_groups)
    examples = tuple(example for group in task_groups for example in group.examples)
    effective_tasks_per_step = tasks_per_step or len(task_groups)
    schedule = _planned_task_batches(
        selected_task_indices,
        tasks_per_step=effective_tasks_per_step,
        epochs=epochs,
        seed=local_config.seed,
    )
    schedule_digest = canonical_digest(
        {
            "split": split,
            "task_indices": selected_task_indices,
            "dataset_revision": dataset.revision,
            "source_digest": dataset.source_digest,
            "constructor_revision": examples[0].constructor_revision,
            "tasks_per_step": effective_tasks_per_step,
            "epochs": epochs,
            "training_max_sequence_length": training_sequence_limit,
            "seed": local_config.seed,
        }
    )
    groups_by_index = {group.task_index: group for group in task_groups}
    current_version = base_revision.weight_version
    step_records: list[dict[str, Any]] = []
    update: Any = None
    last_batch: Any = None
    last_base_version: Any = None
    for epoch_index, step_in_epoch, batch_indices in schedule:
        batch_examples = tuple(
            example
            for task_index in batch_indices
            for example in groups_by_index[task_index].examples
        )
        batch = build_supervised_training_batch(
            batch_id=(f"wideseek-sft:{schedule_digest}:epoch-{epoch_index}:step-{step_in_epoch}"),
            target_policy_id=policy_id,
            expected_base_version=current_version,
            examples=batch_examples,
        )
        last_base_version = current_version
        update = await backend.update_supervised(
            policy_id,
            batch,
            current_version,
        )
        current_version = update.trained_version
        last_batch = batch
        step_records.append(
            {
                "epoch": epoch_index,
                "step_in_epoch": step_in_epoch,
                "optimizer_step": current_version.optimizer_step,
                "task_count": len(batch_indices),
                "example_count": len(batch_examples),
                "target_token_count": sum(
                    len(example.encoding.target_ids) for example in batch_examples
                ),
                "metrics": dict(update.metrics),
            }
        )
    if update is None or last_batch is None or last_base_version is None:
        raise RuntimeError("SFT schedule produced no optimizer updates")
    train_hash_after = backend.adapter_hash(policy_id)
    rollout_hash_before_sync = backend.adapter_hash(policy_id, rollout=True)
    replay = await backend.update_supervised(
        policy_id,
        last_batch,
        last_base_version,
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
        == base_revision.weight_version.optimizer_step + len(schedule),
        "optimizer_steps_match_schedule": update.trained_version.optimizer_step
        == base_revision.weight_version.optimizer_step + len(schedule),
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

    compliance: dict[str, Any] | None = None
    if compliance_report_path is not None:
        from heterospawn.training.wideseek_smoke import (
            WIDESEEK_COMPLIANCE_PROFILE_V1,
            run_wideseek_compliance_baseline,
        )

        compliance_result = await run_wideseek_compliance_baseline(
            topology="shared",
            task_selection=WIDESEEK_COMPLIANCE_PROFILE_V1,
            rollouts_per_task=1,
            data_manifest_path=data_manifest_path,
            data_dir=data_dir,
            service_url=service_url,
            qdrant_url=qdrant_url,
            local_config=local_config,
            report_path=compliance_report_path,
            backend=replacement,
            do_sample=True,
            max_search_message_results=3,
            max_search_content_characters=600,
            max_access_characters=800,
        )
        compliance = {
            "selection_profile": compliance_result["selection_profile"],
            "summary": compliance_result["summary"],
            "checks": compliance_result["checks"],
            "report_digest": _file_sha256(compliance_report_path),
        }

    report: dict[str, Any] = {
        "schema_revision": "heterospawn-wideseek-sft-smoke-v2",
        "status": "passed",
        "comparable_to_official": False,
        "training_scope": "bounded-shared-policy-sft-schedule",
        "dataset_revision": dataset.revision,
        "source_digest": dataset.source_digest,
        "split": split,
        "task_indices": list(selected_task_indices),
        "selected_task_count": len(selected_task_indices),
        "heldout_compliance_selection_excluded": heldout_isolation,
        "skipped_overlength_task_indices": list(skipped_overlength),
        "training_max_sequence_length": training_sequence_limit,
        "rollout_max_sequence_length": local_config.max_sequence_length,
        "tasks_per_step": effective_tasks_per_step,
        "epochs": epochs,
        "optimizer_steps": len(schedule),
        "schedule_digest": schedule_digest,
        "constructor_revision": examples[0].constructor_revision,
        "final_batch_digest": last_batch.batch_digest,
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
        "step_records": step_records,
        "checks": checks,
        "post_sft_compliance": compliance,
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


def _materialize_task_groups(
    *,
    dataset: Any,
    explicit_task_indices: tuple[int, ...],
    task_limit: int,
    constructor: WideSeekRoleSftConstructor,
    codec: Any,
    training_max_sequence_length: int,
    exclude_compliance: bool,
) -> tuple[tuple[_TaskExamples, ...], tuple[int, ...]]:
    if explicit_task_indices:
        candidates = explicit_task_indices
        required_count = len(explicit_task_indices)
    else:
        held_out = _compliance_indices(dataset.split) if exclude_compliance else set()
        candidates = tuple(index for index in range(len(dataset.tasks)) if index not in held_out)
        required_count = task_limit

    groups: list[_TaskExamples] = []
    skipped: list[int] = []
    for task_index in candidates:
        construction = constructor.build(dataset, task_indices=(task_index,))
        examples = materialize_supervised_conversations(
            construction.conversations,
            codec,
        )
        longest = max(
            len(example.encoding.prompt_ids) + len(example.encoding.target_ids)
            for example in examples
        )
        if longest > training_max_sequence_length:
            if explicit_task_indices:
                raise ValueError(
                    f"SFT task index {task_index} exceeds training_max_sequence_length"
                )
            skipped.append(task_index)
            continue
        groups.append(_TaskExamples(task_index=task_index, examples=examples))
        if not explicit_task_indices and len(groups) == required_count:
            break
    if len(groups) != required_count:
        raise ValueError(
            f"only {len(groups)} eligible SFT tasks were found; required {required_count}"
        )
    return tuple(groups), tuple(skipped)


def _planned_task_batches(
    task_indices: tuple[int, ...],
    *,
    tasks_per_step: int,
    epochs: int,
    seed: int,
) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    if not task_indices or len(set(task_indices)) != len(task_indices):
        raise ValueError("planned SFT task indices must be non-empty and unique")
    if tasks_per_step < 1 or epochs < 1:
        raise ValueError("SFT tasks_per_step and epochs must be positive")
    result: list[tuple[int, int, tuple[int, ...]]] = []
    for epoch_index in range(epochs):
        ordered = tuple(
            sorted(
                task_indices,
                key=lambda task_index: canonical_digest(
                    {
                        "seed": seed,
                        "epoch": epoch_index,
                        "task_index": task_index,
                    }
                ),
            )
        )
        for step_in_epoch, offset in enumerate(range(0, len(ordered), tasks_per_step)):
            result.append(
                (
                    epoch_index,
                    step_in_epoch,
                    ordered[offset : offset + tasks_per_step],
                )
            )
    return tuple(result)


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


def _validate_compliance_selection(
    split: WideSeekSplit,
    task_indices: tuple[int, ...],
) -> None:
    overlap = tuple(sorted(set(task_indices) & _compliance_indices(split)))
    if overlap:
        raise ValueError(f"SFT training selection overlaps the fixed compliance profile: {overlap}")


def _compliance_indices(split: WideSeekSplit) -> set[int]:
    from heterospawn.training.wideseek_smoke import (
        WIDESEEK_COMPLIANCE_PROFILE_V1,
    )

    return {
        index
        for compliance_split, index in WIDESEEK_COMPLIANCE_PROFILE_V1
        if compliance_split == split
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
