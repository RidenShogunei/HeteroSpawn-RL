from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import ValidationError

from heterospawn.backends.local_hf.config import LocalLoraConfig, LocalPromptEncoder
from heterospawn.benchmarks.wideseek import load_wideseek_dataset
from heterospawn.domain.ids import PolicyId
from heterospawn.domain.supervised import (
    SupervisedPromptEncoding,
    SupervisedTrainingBatch,
    SupervisedTrainingExample,
)
from heterospawn.domain.versions import WeightVersion
from heterospawn.orchestration.wideseek_actions import parse_main_turn, parse_sub_turn
from heterospawn.training.wideseek_sft import (
    WideSeekRoleSftConstructor,
    build_supervised_training_batch,
    materialize_supervised_conversations,
)
from heterospawn.training.wideseek_sft_smoke import (
    _validate_compliance_selection,
)


class _DeterministicSupervisedCodec:
    def encode_supervised(
        self,
        messages: tuple[object, ...],
        target: str,
        tools: tuple[object, ...] = (),
    ) -> SupervisedPromptEncoding:
        prompt_text = json.dumps(
            {
                "messages": [
                    message.model_dump(mode="json")  # type: ignore[attr-defined]
                    for message in messages
                ],
                "tool_count": len(tools),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return SupervisedPromptEncoding(
            prompt_ids=tuple(prompt_text.encode()),
            target_ids=tuple((target + "<end>").encode()),
            tokenizer_revision="fixture-tokenizer",
            prompt_template_revision="fixture-template",
        )

    def encode(self, messages: tuple[object, ...], tools: tuple[object, ...] = ()) -> object:
        raise AssertionError("supervised materialization must use encode_supervised")

    def decode(self, response_ids: tuple[int, ...]) -> str:
        raise AssertionError("supervised materialization must not decode")


class _PrefixTokenizer:
    chat_template = "fixture-chat-template"
    special_tokens_map: ClassVar[dict[str, str]] = {"eos_token": "<end>"}

    def get_vocab(self) -> dict[str, int]:
        return {"<end>": 0, "x": 1}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **_: object,
    ) -> list[int]:
        assert tokenize is True
        encoded: list[int] = [90]
        role_tokens = {"system": 10, "user": 20, "assistant": 30}
        for message in messages:
            encoded.append(role_tokens[message["role"]])
            encoded.extend(message["content"].encode())
            encoded.append(0)
        if add_generation_prompt:
            encoded.append(role_tokens["assistant"])
        return encoded

    def decode(self, _: list[int], **__: object) -> str:
        return ""


def _write_hybrid_fixture(path: Path) -> tuple[str, str]:
    sentinel = "PRIVATE_REFERENCE_SENTINEL"
    records = [
        {
            "question": "Build the requested table.",
            "answer": (
                "```markdown\n"
                "| Name | Value |\n"
                "|---|---|\n"
                f"| {sentinel}-A | 1 |\n"
                "| B | 2 |\n"
                "| C | 3 |\n"
                "| D | 4 |\n"
                "| E | 5 |\n"
                "```"
            ),
            "unique_columns": ["Name"],
            "is_markdown": True,
            "instance_id": 11,
        },
        {
            "question": "Return the requested fact.",
            "answer": f"{sentinel}-FACT",
            "unique_columns": None,
            "is_markdown": False,
            "instance_id": 12,
        },
    ]
    content = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    )
    path.write_bytes(content.encode())
    return hashlib.sha256(content.encode()).hexdigest(), sentinel


def test_role_targeted_construction_is_deterministic_legal_and_answer_safe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "hybrid_20k.jsonl"
    digest, sentinel = _write_hybrid_fixture(path)
    dataset = load_wideseek_dataset(
        path,
        split="hybrid_20k",
        expected_sha256=digest,
    )
    constructor = WideSeekRoleSftConstructor(max_workers=4)

    first = constructor.build(dataset, task_indices=(0, 1))
    second = constructor.build(dataset, task_indices=(0, 1))

    assert first == second
    assert first.summary.main_final_examples == 2
    assert first.summary.sub_summary_examples == 5
    assert first.summary.worker_count_histogram == ((1, 1), (4, 1))
    assert sentinel not in first.summary.model_dump_json()
    assert sentinel not in repr(first)
    assert {item.behavior for item in first.conversations} == {
        "main_final",
        "sub_summary",
    }

    for conversation in first.conversations:
        if conversation.behavior == "main_final":
            assert conversation.agent_role == "main"
            assert len(conversation.messages) == 4
            assert parse_main_turn(conversation.messages[2].content).kind == "spawn"
            assert "<tool_call>" not in conversation.target
        else:
            assert conversation.agent_role == "sub"
            assert len(conversation.messages) == 6
            assert parse_sub_turn(conversation.messages[2].content).kind == "tools"
            assert parse_sub_turn(conversation.messages[4].content).kind == "tools"
            assert "<tool_call>" not in conversation.target


def test_materialization_masks_only_targets_and_batch_digest_is_enforced(
    tmp_path: Path,
) -> None:
    path = tmp_path / "hybrid_20k.jsonl"
    digest, _ = _write_hybrid_fixture(path)
    dataset = load_wideseek_dataset(
        path,
        split="hybrid_20k",
        expected_sha256=digest,
    )
    construction = WideSeekRoleSftConstructor(max_workers=2).build(
        dataset,
        task_indices=(0,),
    )
    examples = materialize_supervised_conversations(
        construction.conversations,
        _DeterministicSupervisedCodec(),  # type: ignore[arg-type]
    )

    assert len(examples) == 3
    for example in examples:
        prompt_length = len(example.encoding.prompt_ids)
        target_length = len(example.encoding.target_ids)
        assert example.loss_mask == (0,) * prompt_length + (1,) * target_length
        assert sum(example.loss_mask) == target_length

    policy_id = PolicyId("shared-policy")
    version = WeightVersion(
        policy_id=policy_id,
        optimizer_step=0,
        checkpoint_digest="base",
    )
    batch = build_supervised_training_batch(
        batch_id="sft-batch-0",
        target_policy_id=policy_id,
        expected_base_version=version,
        examples=examples,
    )
    assert batch.objective == "causal_sft"

    invalid = batch.model_dump(mode="python")
    invalid["batch_digest"] = "0" * 64
    with pytest.raises(ValidationError, match="batch_digest"):
        SupervisedTrainingBatch.model_validate(invalid, strict=True)

    invalid_example = examples[0].model_dump(mode="python")
    invalid_example["loss_mask"] = (1,) * len(examples[0].loss_mask)
    with pytest.raises(ValidationError, match="loss_mask"):
        SupervisedTrainingExample.model_validate(invalid_example, strict=True)


def test_local_prompt_encoder_preserves_exact_supervised_boundary() -> None:
    tokenizer = _PrefixTokenizer()
    config = LocalLoraConfig(device="cpu", dtype="float32")
    encoder = LocalPromptEncoder(tokenizer, config)
    construction_messages = (
        # Existing assistant/tool history is part of the prompt and must remain masked.
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "prior tool call"},
        {"role": "user", "content": "tool result"},
    )
    from heterospawn.policies.base import Message

    messages = tuple(Message.model_validate(item, strict=True) for item in construction_messages)
    target = "final answer"
    encoding = encoder.encode_supervised(messages, target)
    expected_full = tuple(
        tokenizer.apply_chat_template(
            [
                *construction_messages,
                {"role": "assistant", "content": target},
            ],
            tokenize=True,
            add_generation_prompt=False,
        )
    )

    assert encoding.prompt_ids + encoding.target_ids == expected_full
    assert encoding.target_ids[-1] == 0


def test_sft_selection_cannot_overlap_fixed_compliance_profile() -> None:
    with pytest.raises(ValueError, match="overlaps"):
        _validate_compliance_selection("hybrid_20k", (0, 1))

    _validate_compliance_selection("hybrid_20k", (1, 2, 3))
