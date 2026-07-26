from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
from typing import Any, ClassVar, Literal
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

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
from heterospawn.training.wideseek_sft import (
    SupervisedConversation,
    build_supervised_training_batch,
    materialize_supervised_conversations,
)

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
        return {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "tiny": 5,
            "prompt": 6,
            "target": 7,
        }

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        tools: list[dict[str, object]] | None = None,
    ) -> list[int]:
        assert tokenize and messages
        if tools is not None:
            assert tools
        if add_generation_prompt:
            return [1, 5, 6]
        target = messages[-1]
        assert target["role"] == "assistant"
        target_ids = [10 + (byte % 50) for byte in target["content"].encode()]
        return [1, 5, 6, *target_ids, 2]


def _backend(
    tmp_path: Path,
    *,
    dtype: str = "float32",
    architecture: str = "qwen2",
    policy_ids: tuple[PolicyId, ...] = (PolicyId("main"), PolicyId("sub")),
) -> LocalHfLoraBackend:
    config_class = transformers.Qwen3Config if architecture == "qwen3" else transformers.Qwen2Config
    model_class = (
        transformers.Qwen3ForCausalLM if architecture == "qwen3" else transformers.Qwen2ForCausalLM
    )
    model_config = config_class(
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
        head_dim=8,
    )
    model = model_class(model_config)
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
        policy_ids=policy_ids,
    )


@pytest.mark.asyncio
async def test_qwen3_architecture_generates_exact_trainable_trajectory(tmp_path: Path) -> None:
    backend = _backend(tmp_path, architecture="qwen3")
    main = PolicyId("main")
    request = _request(backend, role="main", request_id="qwen3")

    result = await backend.endpoint(main).generate(
        request,
        backend.rollout_revision(main),
    )

    assert result.response_ids
    assert len(result.response_ids) == len(result.response_log_probs)


@pytest.mark.asyncio
async def test_raw_policy_sampling_preserves_update_logprob_semantics(tmp_path: Path) -> None:
    backend = _backend(tmp_path, architecture="qwen3")
    main = PolicyId("main")
    revision = backend.rollout_revision(main)
    request = _request(backend, role="main", request_id="sampled").model_copy(
        update={
            "sampling_params": (
                ("max_new_tokens", 3),
                ("do_sample", True),
                ("temperature", 1.0),
                ("top_p", 1.0),
                ("top_k", 0),
            )
        }
    )

    result = await backend.endpoint(main).generate(request, revision)
    batch = TrainingBatchBuilder().build(
        batch_id="sampled-main-update",
        phase="main_update",
        target_policy_id=main,
        expected_base_version=revision.weight_version,
        steps=(_step(request, result, step_id="sampled-main-step"),),
        episode_advantages={request.episode_id: 1.0},
    )
    update = await backend.update_policy(main, batch, revision.weight_version)

    metrics = dict(update.metrics)
    assert metrics["old_new_ratio_mean"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["approx_kl_mean"] == pytest.approx(0.0, abs=1e-5)


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


def _supervised_conversation(
    role: Literal["main", "sub"],
    behavior: Literal["main_final", "sub_summary"],
    target: str,
) -> SupervisedConversation:
    return SupervisedConversation(
        example_id=f"sft-{behavior}",
        task_id=TaskId("sft-task"),
        agent_role=role,
        behavior=behavior,
        dataset_revision="fixture-dataset",
        source_digest="a" * 64,
        constructor_revision="b" * 64,
        messages=(
            Message(role="system", content="fixture system"),
            Message(role="user", content="fixture evidence"),
        ),
        tools=(),
        target=target,
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
async def test_supervised_update_is_role_balanced_idempotent_and_restorable(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path / "original")
    main = PolicyId("main")
    sub = PolicyId("sub")
    main_revision = backend.rollout_revision(main)
    main_train_before = backend.adapter_hash(main)
    main_rollout_before = backend.adapter_hash(main, rollout=True)
    sub_train_before = backend.adapter_hash(sub)
    sub_rollout_before = backend.adapter_hash(sub, rollout=True)
    conversations = (
        _supervised_conversation("main", "main_final", "answer"),
        _supervised_conversation("sub", "sub_summary", "evidence"),
    )
    examples = materialize_supervised_conversations(
        conversations,
        backend.prompt_encoder,
    )
    batch = build_supervised_training_batch(
        batch_id="shared-sft-update-1",
        target_policy_id=main,
        expected_base_version=main_revision.weight_version,
        examples=examples,
    )

    update = await backend.update_supervised(
        main,
        batch,
        main_revision.weight_version,
    )

    metrics = dict(update.metrics)
    assert update.trained_version.optimizer_step == 1
    assert backend.adapter_hash(main) != main_train_before
    assert backend.adapter_hash(main, rollout=True) == main_rollout_before
    assert backend.adapter_hash(sub) == sub_train_before
    assert backend.adapter_hash(sub, rollout=True) == sub_rollout_before
    assert metrics["gradient_norm"] > 0
    assert metrics["loss"] == pytest.approx(
        (metrics["main_final_token_loss"] + metrics["sub_summary_token_loss"]) / 2
    )

    replay = await backend.update_supervised(
        main,
        batch,
        main_revision.weight_version,
    )
    assert replay == update
    assert backend.weight_version(main).optimizer_step == 1

    conflicting_examples = materialize_supervised_conversations(
        (
            _supervised_conversation("main", "main_final", "different answer"),
            conversations[1],
        ),
        backend.prompt_encoder,
    )
    conflicting_batch = build_supervised_training_batch(
        batch_id=batch.batch_id,
        target_policy_id=main,
        expected_base_version=main_revision.weight_version,
        examples=conflicting_examples,
    )
    with pytest.raises(TrainingBatchError, match="another digest"):
        await backend.update_supervised(
            main,
            conflicting_batch,
            main_revision.weight_version,
        )

    second_batch = build_supervised_training_batch(
        batch_id="shared-sft-update-2",
        target_policy_id=main,
        expected_base_version=update.trained_version,
        examples=examples,
    )
    second_update = await backend.update_supervised(
        main,
        second_batch,
        update.trained_version,
    )
    assert second_update.trained_version.optimizer_step == 2

    synced = await backend.sync_rollout_weights(main, second_update.trained_version)
    assert synced.replica_set_revision == main_revision.replica_set_revision + 1
    assert backend.adapter_hash(main, rollout=True) == backend.adapter_hash(main)

    replacement = _backend(tmp_path / "replacement")
    restored = await replacement.restore_checkpoint(second_update.checkpoint)
    recovered = await replacement.sync_rollout_weights(main, restored)
    assert restored == second_update.trained_version
    assert recovered.weight_version == synced.weight_version
    assert recovered.deployment_id != synced.deployment_id
    assert replacement.adapter_hash(main) == backend.adapter_hash(main)
    assert replacement.adapter_hash(main, rollout=True) == replacement.adapter_hash(main)


@pytest.mark.asyncio
async def test_shared_checkpoint_forks_into_independent_policy_lineages(
    tmp_path: Path,
) -> None:
    shared = PolicyId("shared")
    main = PolicyId("main")
    sub = PolicyId("sub")
    source_backend = _backend(
        tmp_path / "source",
        policy_ids=(shared,),
    )
    source_revision = source_backend.rollout_revision(shared)
    examples = materialize_supervised_conversations(
        (
            _supervised_conversation("main", "main_final", "answer"),
            _supervised_conversation("sub", "sub_summary", "evidence"),
        ),
        source_backend.prompt_encoder,
    )
    batch = build_supervised_training_batch(
        batch_id="shared-sft-source",
        target_policy_id=shared,
        expected_base_version=source_revision.weight_version,
        examples=examples,
    )
    source_update = await source_backend.update_supervised(
        shared,
        batch,
        source_revision.weight_version,
    )
    source_hash = source_backend.adapter_hash(shared)

    backend = _backend(tmp_path / "forked")
    old_main_rollout = backend.rollout_revision(main)
    old_sub_rollout = backend.rollout_revision(sub)
    forked = await backend.fork_checkpoint(
        source_update.checkpoint,
        (main, sub),
    )
    main_checkpoint, sub_checkpoint = forked

    assert backend.adapter_hash(main) == source_hash
    assert backend.adapter_hash(sub) == source_hash
    assert backend.rollout_revision(main) == old_main_rollout
    assert backend.rollout_revision(sub) == old_sub_rollout
    assert main_checkpoint.policy_id == main
    assert sub_checkpoint.policy_id == sub
    assert main_checkpoint.weight_version.optimizer_step == 1
    assert sub_checkpoint.weight_version.optimizer_step == 1
    assert (
        main_checkpoint.weight_version.checkpoint_digest
        != sub_checkpoint.weight_version.checkpoint_digest
    )
    expected_lineage = {
        "kind": "policy_fork",
        "source_policy_id": "shared",
        "source_optimizer_step": 1,
        "source_checkpoint_digest": source_update.trained_version.checkpoint_digest,
        "source_optimizer_state_digest": source_update.checkpoint.optimizer_state_digest,
    }
    assert backend.checkpoint_lineage(main_checkpoint) == expected_lineage
    assert backend.checkpoint_lineage(sub_checkpoint) == expected_lineage
    source_optimizer = torch.load(
        _checkpoint_path(source_update.checkpoint.uri) / "optimizer.pt",
        map_location="cpu",
        weights_only=True,
    )
    main_optimizer = torch.load(
        _checkpoint_path(main_checkpoint.uri) / "optimizer.pt",
        map_location="cpu",
        weights_only=True,
    )
    sub_optimizer = torch.load(
        _checkpoint_path(sub_checkpoint.uri) / "optimizer.pt",
        map_location="cpu",
        weights_only=True,
    )
    _assert_optimizer_state_equal(source_optimizer, main_optimizer)
    _assert_optimizer_state_equal(source_optimizer, sub_optimizer)

    main_synced = await backend.sync_rollout_weights(main, main_checkpoint.weight_version)
    assert backend.adapter_hash(main, rollout=True) == source_hash
    assert backend.rollout_revision(sub) == old_sub_rollout
    sub_synced = await backend.sync_rollout_weights(sub, sub_checkpoint.weight_version)
    assert backend.adapter_hash(sub, rollout=True) == source_hash
    assert main_synced.weight_version.policy_id == main
    assert sub_synced.weight_version.policy_id == sub

    sub_hash_before_main_update = backend.adapter_hash(sub)
    sub_version_before_main_update = backend.weight_version(sub)
    request = _request(backend, role="main", request_id="forked-main")
    result = await backend.endpoint(main).generate(request, main_synced)
    update_batch = TrainingBatchBuilder().build(
        batch_id="forked-main-update",
        phase="main_update",
        target_policy_id=main,
        expected_base_version=main_synced.weight_version,
        steps=(_step(request, result, step_id="forked-main-step"),),
        episode_advantages={request.episode_id: 1.0},
    )
    main_update = await backend.update_policy(
        main,
        update_batch,
        main_synced.weight_version,
    )

    assert main_update.trained_version.optimizer_step == 2
    assert backend.weight_version(sub) == sub_version_before_main_update
    assert backend.adapter_hash(sub) == sub_hash_before_main_update


def _assert_optimizer_state_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert torch.equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_optimizer_state_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_optimizer_state_equal(left_item, right_item)
        return
    assert left == right


def _checkpoint_path(uri: str) -> Path:
    parsed = urlparse(uri)
    path = url2pathname(unquote(parsed.path))
    if os.name == "nt" and path.startswith("\\") and len(path) > 2 and path[2] == ":":
        path = path[1:]
    return Path(path)


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
