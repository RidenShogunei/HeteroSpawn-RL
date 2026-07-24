from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from heterospawn.backends.local_hf.config import LocalLoraConfig
from heterospawn.cli import _local_lora_config, build_parser


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
    assert config.gradient_checkpointing is True
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
