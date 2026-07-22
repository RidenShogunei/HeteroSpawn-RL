from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
from typing import Any, ClassVar

import pytest

from heterospawn.backends.local_hf import LocalHfLoraBackend, LocalLoraConfig
from heterospawn.domain.ids import AgentInstanceId, EpisodeId, PolicyId, RolloutId, StepId, TaskId
from heterospawn.domain.training import GenerationRequest, TrajectoryStep
from heterospawn.errors import RolloutRevisionMismatch
from heterospawn.training import TrainingBatchBuilder

if os.environ.get("HETEROSPAWN_RUN_LOCAL_BACKEND_TESTS") != "1":
    pytest.skip(
        "set HETEROSPAWN_RUN_LOCAL_BACKEND_TESTS=1 for optional local backend tests",
        allow_module_level=True,
    )

torch = importlib.import_module("torch")
transformers = importlib.import_module("transformers")
importlib.import_module("peft")
importlib.import_module("safetensors")

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
    ) -> list[int]:
        assert tokenize and add_generation_prompt and messages
        return [1, 5, 6]


def _backend(tmp_path: Path) -> LocalHfLoraBackend:
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
    return LocalHfLoraBackend(
        config=LocalLoraConfig(
            model_id="tiny-random-qwen2",
            model_revision="fixture-v1",
            device="cpu",
            dtype="float32",
            max_sequence_length=64,
            max_new_tokens=3,
            artifact_dir=tmp_path / "checkpoints",
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
