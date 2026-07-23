from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
from typing import Any, ClassVar

import pytest

from heterospawn.backends.local_hf import LocalHfLoraBackend, LocalLoraConfig
from heterospawn.backends.vllm_rollout.models import rollout_artifact_path
from heterospawn.domain.ids import AgentInstanceId, EpisodeId, PolicyId, RolloutId, StepId, TaskId
from heterospawn.domain.training import GenerationRequest, TrajectoryStep
from heterospawn.errors import (
    CheckpointIntegrityError,
    RolloutRevisionMismatch,
    TrainingBatchError,
)
from heterospawn.policies.base import Message
from heterospawn.policies.trainable import ToolDefinition
from heterospawn.training import TrainingBatchBuilder

if os.environ.get("HETEROSPAWN_RUN_LOCAL_BACKEND_TESTS") != "1":
    pytest.skip(
        "set HETEROSPAWN_RUN_LOCAL_BACKEND_TESTS=1 for optional local backend tests",
        allow_module_level=True,
    )

transformers = importlib.import_module("transformers")
torch = importlib.import_module("torch")
safetensors_torch = importlib.import_module("safetensors.torch")

pytestmark = pytest.mark.local_backend


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    special_tokens_map: ClassVar[dict[str, str]] = {
        "eos_token": "<eos>",
        "pad_token": "<pad>",
    }
    chat_template = "tiny-contract-template-v1"

    def get_vocab(self) -> dict[str, int]:
        return {"<pad>": 0, "<bos>": 1, "<eos>": 2, "tiny": 5, "prompt": 6}

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        tools: list[dict[str, object]] | None = None,
    ) -> list[int]:
        assert tokenize and add_generation_prompt and messages
        if tools is not None:
            assert tools
        return [1, 5, 6]


def _backend(
    tmp_path: Path,
    *,
    dtype: str = "float32",
) -> LocalHfLoraBackend:
    model_config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = transformers.Qwen2ForCausalLM(model_config)
    if dtype == "float16":
        model = model.half()
    return LocalHfLoraBackend(
        config=LocalLoraConfig(
            model_id="tiny-random-qwen2",
            model_revision="fixture-v1",
            device="cpu",
            dtype=dtype,
            max_sequence_length=64,
            max_new_tokens=3,
            artifact_dir=tmp_path / f"{dtype}-checkpoints",
        ),
        model=model,
        tokenizer=TinyTokenizer(),
        policy_ids=(PolicyId("main"), PolicyId("sub")),
    )


def _request(
    backend: LocalHfLoraBackend,
    *,
    role: str,
    request_id: str,
) -> GenerationRequest:
    return GenerationRequest(
        task_id=TaskId("task"),
        episode_id=EpisodeId(f"episode-{request_id}"),
        rollout_id=RolloutId(f"rollout-{request_id}"),
        request_id=request_id,
        agent_role="main" if role == "main" else "sub",
        agent_instance_id=AgentInstanceId(f"{role}-{request_id}"),
        prompt_ids=(1, 5, 6),
        tokenizer_revision=backend.prompt_encoder.tokenizer_revision,
        prompt_template_revision=backend.prompt_encoder.prompt_template_revision,
        sampling_params=(("max_new_tokens", 3), ("do_sample", False)),
    )


def _step(
    request: GenerationRequest,
    result: Any,
    *,
    step_id: str,
) -> TrajectoryStep:
    return TrajectoryStep(
        task_id=request.task_id,
        episode_id=request.episode_id,
        rollout_id=request.rollout_id,
        step_id=StepId(step_id),
        event_index=0,
        agent_role=request.agent_role,
        agent_instance_id=request.agent_instance_id,
        policy_id=result.policy_id,
        rollout_revision=result.rollout_revision,
        prompt_ids=request.prompt_ids,
        response_ids=result.response_ids,
        response_log_probs=result.response_log_probs,
        tokenizer_revision=request.tokenizer_revision,
        prompt_template_revision=request.prompt_template_revision,
        sampling_params=request.sampling_params,
        stop_reason=result.stop_reason,
    )


@pytest.mark.asyncio
async def test_exact_generate_update_sync_and_partner_isolation(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    main = PolicyId("main")
    sub = PolicyId("sub")
    main_revision = backend.rollout_revision(main)
    main_rollout_hash = backend.adapter_hash(main, rollout=True)
    sub_train_hash = backend.adapter_hash(sub)
    sub_rollout_hash = backend.adapter_hash(sub, rollout=True)
    request = _request(backend, role="main", request_id="main-0")

    result = await backend.endpoint(main).generate(request, main_revision)
    assert len(result.response_ids) == len(result.response_log_probs)
    step = _step(request, result, step_id="main-step")
    batch = TrainingBatchBuilder().build(
        batch_id="main-update-1",
        phase="main_update",
        target_policy_id=main,
        expected_base_version=main_revision.weight_version,
        steps=(step,),
        episode_advantages={request.episode_id: 1.0},
    )

    update = await backend.update_policy(main, batch, main_revision.weight_version)

    assert update.trained_version.optimizer_step == 1
    assert backend.adapter_hash(main) != main_rollout_hash
    assert backend.adapter_hash(main, rollout=True) == main_rollout_hash
    assert backend.adapter_hash(sub) == sub_train_hash
    assert backend.adapter_hash(sub, rollout=True) == sub_rollout_hash
    assert backend.rollout_revision(main) == main_revision

    synced = await backend.sync_rollout_weights(main, update.trained_version)
    assert synced.replica_set_revision == 1
    assert backend.adapter_hash(main, rollout=True) == backend.adapter_hash(main)
    with pytest.raises(RolloutRevisionMismatch):
        await backend.endpoint(main).generate(request, main_revision)


@pytest.mark.asyncio
async def test_generation_accepts_only_prompt_revisions_issued_for_tool_schema(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    main = PolicyId("main")
    encoding = backend.prompt_encoder.encode(
        (Message(role="user", content="use the tool"),),
        (
            ToolDefinition(
                name="search",
                description="Search the fixture.",
                parameters_json='{"type":"object","properties":{}}',
            ),
        ),
    )
    request = _request(backend, role="main", request_id="tools").model_copy(
        update={
            "prompt_ids": encoding.prompt_ids,
            "prompt_template_revision": encoding.prompt_template_revision,
        }
    )
    await backend.endpoint(main).generate(request, backend.rollout_revision(main))

    forged = request.model_copy(update={"prompt_template_revision": "unissued-revision"})
    with pytest.raises(TrainingBatchError):
        await backend.endpoint(main).generate(forged, backend.rollout_revision(main))


@pytest.mark.asyncio
async def test_four_sub_requests_share_endpoint_and_checkpoint_restores(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    sub = PolicyId("sub")
    endpoint = backend.endpoint(sub)
    revision = backend.rollout_revision(sub)
    requests = tuple(_request(backend, role="sub", request_id=f"sub-{index}") for index in range(4))
    results = await asyncio.gather(*(endpoint.generate(request, revision) for request in requests))

    assert len(results) == 4
    assert all(result.rollout_revision == revision for result in results)
    steps = tuple(
        _step(request, result, step_id=f"sub-step-{index}")
        for index, (request, result) in enumerate(zip(requests, results, strict=True))
    )
    batch = TrainingBatchBuilder().build(
        batch_id="sub-update-1",
        phase="sub_update",
        target_policy_id=sub,
        expected_base_version=revision.weight_version,
        steps=steps,
        episode_advantages={request.episode_id: 1.0 for request in requests},
    )
    update = await backend.update_policy(sub, batch, revision.weight_version)
    trained_hash = backend.adapter_hash(sub)

    restored = await backend.restore_checkpoint(update.checkpoint)

    assert restored == update.trained_version
    assert backend.adapter_hash(sub) == trained_hash
    assert update.checkpoint.optimizer_state_digest


@pytest.mark.asyncio
async def test_restart_restores_weights_under_a_new_deployment_identity(tmp_path: Path) -> None:
    backend = _backend(tmp_path / "original")
    main = PolicyId("main")
    request = _request(backend, role="main", request_id="restart")
    original_revision = backend.rollout_revision(main)
    result = await backend.endpoint(main).generate(request, original_revision)
    batch = TrainingBatchBuilder().build(
        batch_id="restart-update-1",
        phase="main_update",
        target_policy_id=main,
        expected_base_version=original_revision.weight_version,
        steps=(_step(request, result, step_id="restart-step"),),
        episode_advantages={request.episode_id: 1.0},
    )
    update = await backend.update_policy(main, batch, original_revision.weight_version)
    committed_revision = await backend.sync_rollout_weights(main, update.trained_version)

    replacement = _backend(tmp_path / "replacement")
    await replacement.restore_checkpoint(update.checkpoint)
    recovered_revision = await replacement.sync_rollout_weights(main, update.trained_version)

    assert recovered_revision.weight_version == committed_revision.weight_version
    assert recovered_revision.deployment_id != committed_revision.deployment_id
    assert replacement.adapter_hash(main) == replacement.adapter_hash(main, rollout=True)
    with pytest.raises(RolloutRevisionMismatch):
        await replacement.endpoint(main).generate(request, committed_revision)


@pytest.mark.asyncio
async def test_export_rollout_artifact_is_exact_and_idempotent(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    main = PolicyId("main")
    version = backend.rollout_revision(main).weight_version

    first = await backend.export_rollout_artifact(main, version)
    second = await backend.export_rollout_artifact(main, version)

    assert first == second
    artifact_path = rollout_artifact_path(first)
    assert (artifact_path / "adapter_config.json").is_file()
    assert (artifact_path / "adapter_model.safetensors").is_file()
    assert first.format_revision == "peft-lora-v1"

    (artifact_path / "adapter_model.safetensors").write_bytes(b"corrupted")
    with pytest.raises(CheckpointIntegrityError, match="differs from its training checkpoint"):
        await backend.export_rollout_artifact(main, version)


@pytest.mark.asyncio
async def test_all_lora_adapters_remain_float32_with_fp16_base(tmp_path: Path) -> None:
    backend = _backend(tmp_path, dtype="float16")

    for policy_id in (PolicyId("main"), PolicyId("sub")):
        version = backend.rollout_revision(policy_id).weight_version
        artifact = await backend.export_rollout_artifact(policy_id, version)
        tensors = safetensors_torch.load_file(
            str(rollout_artifact_path(artifact) / "adapter_model.safetensors")
        )
        assert tensors
        assert {tensor.dtype for tensor in tensors.values()} == {torch.float32}
        assert all(torch.isfinite(tensor).all() for tensor in tensors.values())
