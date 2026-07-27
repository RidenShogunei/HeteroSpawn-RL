from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from heterospawn.backends.local_hf import load_local_checkpoint_ref
from heterospawn.backends.local_hf.config import LocalLoraConfig
from heterospawn.cli import _local_lora_config, build_parser
from heterospawn.domain.training import canonical_digest
from heterospawn.errors import CheckpointIntegrityError


def test_model_manifest_identity_is_complete_and_preferred() -> None:
    config = LocalLoraConfig(
        model_path=Path("model"),
        model_manifest_path=Path("manifest.json"),
        expected_model_manifest_digest="a" * 64,
    )

    assert config.base_model_identity_kind == "hf-asset-manifest"
    assert config.base_model_identity == "a" * 64


@pytest.mark.parametrize(
    ("manifest_path", "manifest_digest"),
    (
        (Path("manifest.json"), None),
        (None, "a" * 64),
    ),
)
def test_model_manifest_path_and_digest_are_atomic(
    manifest_path: Path | None,
    manifest_digest: str | None,
) -> None:
    with pytest.raises(ValidationError, match="must be set together"):
        LocalLoraConfig(
            model_path=Path("model"),
            model_manifest_path=manifest_path,
            expected_model_manifest_digest=manifest_digest,
        )


def test_bnb_4bit_is_restricted_to_cuda_fp16() -> None:
    with pytest.raises(ValidationError, match="only on CUDA"):
        LocalLoraConfig(device="cpu", quantization="bnb-4bit")
    with pytest.raises(ValidationError, match="requires float16"):
        LocalLoraConfig(device="cuda:0", dtype="float32", quantization="bnb-4bit")


def test_qwen3_cli_profile_selects_verified_qlora_defaults(tmp_path: Path) -> None:
    config = _local_lora_config(
        model_profile="qwen3-4b",
        device="cuda:0",
        model_path=tmp_path / "model",
        model_manifest=None,
        artifact_dir=tmp_path / "checkpoints",
        max_sequence_length=4096,
        max_new_tokens=512,
    )

    assert config.model_id == "Qwen/Qwen3-4B"
    assert config.model_revision == "1cfa9a7208912126459214e8b04321603b3df60c"
    assert config.model_manifest_path == Path("manifests/qwen3-4b.json")
    assert config.quantization == "bnb-4bit"
    assert config.attention_implementation == "sdpa"
    assert config.response_only_logits is True
    assert config.gradient_checkpointing is True
    assert config.gradient_checkpointing_use_reentrant is False
    assert config.enable_thinking is False


def test_qwen3_cli_profile_requires_verified_local_model_path(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="requires --model-path"):
        _local_lora_config(
            model_profile="qwen3-4b",
            device="cuda:0",
            model_path=None,
            model_manifest=None,
            artifact_dir=tmp_path,
            max_sequence_length=4096,
            max_new_tokens=512,
        )


def test_wideseek_train_cli_exposes_sampled_rollout_controls() -> None:
    args = build_parser().parse_args(
        [
            "wideseek-train-smoke",
            "--model-path",
            "model",
            "--do-sample",
        ]
    )

    assert args.do_sample is True


def test_wideseek_train_cli_exposes_shared_checkpoint_initialization() -> None:
    args = build_parser().parse_args(
        [
            "wideseek-train-smoke",
            "--topology",
            "shared",
            "--model-path",
            "model",
            "--checkpoint-dir",
            "sft-checkpoint",
            "--max-new-tokens",
            "1024",
            "--require-learning-signal",
        ]
    )

    assert args.checkpoint_dir == Path("sft-checkpoint")
    assert args.max_sequence_length == 4096
    assert args.max_new_tokens == 1024
    assert args.require_learning_signal is True


def test_local_checkpoint_ref_is_reconstructed_from_manifest(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "shared_step-7"
    checkpoint_dir.mkdir()
    payload = {
        "schema_version": 1,
        "policy_id": "shared",
        "optimizer_step": 7,
        "file_digests": {"optimizer.pt": "optimizer-digest"},
    }
    digest = canonical_digest(payload)
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps({**payload, "checkpoint_digest": digest}),
        encoding="utf-8",
    )

    checkpoint = load_local_checkpoint_ref(checkpoint_dir)

    assert checkpoint.checkpoint_id == f"shared:step-7:{digest[:12]}"
    assert checkpoint.policy_id == "shared"
    assert checkpoint.weight_version.optimizer_step == 7
    assert checkpoint.weight_version.checkpoint_digest == digest
    assert checkpoint.optimizer_state_digest == "optimizer-digest"
    assert checkpoint.uri == checkpoint_dir.resolve().as_uri()


def test_local_checkpoint_ref_rejects_manifest_identity_mismatch(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "manifest.json").write_text(
        """
        {
          "schema_version": 1,
          "policy_id": "shared",
          "optimizer_step": 7,
          "file_digests": {"optimizer.pt": "optimizer-digest"},
          "checkpoint_digest": "wrong"
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(CheckpointIntegrityError, match="manifest digest mismatch"):
        load_local_checkpoint_ref(checkpoint_dir)


def test_wideseek_sft_smoke_defaults_to_qwen3_shared_update() -> None:
    args = build_parser().parse_args(
        [
            "wideseek-sft-smoke",
            "--model-path",
            "model",
        ]
    )

    assert args.model_profile == "qwen3-4b"
    assert args.task_indices is None
    assert args.max_sequence_length == 4096
    assert args.max_new_tokens == 512
    assert args.max_workers == 4
    assert args.run_compliance is False


def test_wideseek_multistep_sft_separates_training_and_rollout_limits() -> None:
    args = build_parser().parse_args(
        [
            "wideseek-sft-train",
            "--model-path",
            "model",
        ]
    )

    assert args.model_profile == "qwen3-4b"
    assert args.task_limit == 192
    assert args.tasks_per_step == 4
    assert args.epochs == 1
    assert args.training_max_sequence_length == 2304
    assert args.max_sequence_length == 4096


def test_wideseek_compliance_exposes_checkpoint_only_rollout() -> None:
    args = build_parser().parse_args(
        [
            "wideseek-compliance-baseline",
            "--model-path",
            "model",
            "--checkpoint-dir",
            "checkpoint",
            "--max-sequence-length",
            "8192",
        ]
    )

    assert args.checkpoint_dir == Path("checkpoint")
    assert args.max_sequence_length == 8192


def test_wideseek_compliance_exposes_independent_checkpoint_pair() -> None:
    args = build_parser().parse_args(
        [
            "wideseek-compliance-baseline",
            "--topology",
            "independent",
            "--model-path",
            "model",
            "--main-checkpoint-dir",
            "main",
            "--sub-checkpoint-dir",
            "sub",
        ]
    )

    assert args.main_checkpoint_dir == Path("main")
    assert args.sub_checkpoint_dir == Path("sub")


def test_wideseek_recovery_cli_exposes_durable_phase_identity() -> None:
    args = build_parser().parse_args(
        [
            "wideseek-recover-phase",
            "--transaction-id",
            "experiment:cycle:joint_update",
            "--model-path",
            "model",
            "--require-learning-signal",
        ]
    )

    assert args.transaction_id == "experiment:cycle:joint_update"
    assert args.require_learning_signal is True
