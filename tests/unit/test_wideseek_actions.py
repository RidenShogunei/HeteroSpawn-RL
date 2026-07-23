from __future__ import annotations

from typing import ClassVar

import pytest

from heterospawn.backends.local_hf.config import LocalLoraConfig, LocalPromptEncoder
from heterospawn.errors import InvalidActionError
from heterospawn.orchestration.wideseek_actions import (
    MAIN_TOOLS,
    MainAnswerTurn,
    MainSpawnTurn,
    SubSummaryTurn,
    SubToolsTurn,
    parse_main_turn,
    parse_sub_turn,
)
from heterospawn.policies.base import Message


def _call(name: str, arguments: str) -> str:
    return f'<tool_call>{{"name":"{name}","arguments":{arguments}}}</tool_call>'


def test_main_no_tool_is_answer_and_one_to_four_subtasks_spawn() -> None:
    answer = parse_main_turn("final answer")
    assert isinstance(answer, MainAnswerTurn)
    assert answer.answer == "final answer"

    spawn = parse_main_turn(
        _call("subtask", '{"subtask":"a"}')
        + _call("subtask", '{"subtask":"b"}')
        + _call("subtask", '{"subtask":"c"}')
        + _call("subtask", '{"subtask":"d"}')
    )
    assert isinstance(spawn, MainSpawnTurn)
    assert spawn.subtasks == ("a", "b", "c", "d")


@pytest.mark.parametrize(
    "content",
    [
        "",
        "<tool_call></tool_call>",
        _call("subtask", '{"subtask":""}'),
        _call("search", '{"query":"not allowed for Main"}'),
        _call("subtask", '{"subtask":"1"}') * 5,
        '<tool_call>{"name":"subtask","arguments":',
    ],
)
def test_main_rejects_empty_malformed_unknown_and_over_limit_calls(content: str) -> None:
    with pytest.raises(InvalidActionError):
        parse_main_turn(content)


def test_sub_parses_search_access_or_direct_summary() -> None:
    tools = parse_sub_turn(
        _call("search", '{"query":"alpha","topk":3}')
        + _call(
            "access",
            '{"url":"https://docs/alpha","info_to_extract":"dates"}',
        )
    )
    assert isinstance(tools, SubToolsTurn)
    assert [call.name for call in tools.calls] == ["search", "access"]

    summary = parse_sub_turn("evidence summary")
    assert isinstance(summary, SubSummaryTurn)
    assert summary.summary == "evidence summary"


def test_sub_rejects_more_than_three_calls() -> None:
    with pytest.raises(InvalidActionError):
        parse_sub_turn(_call("search", '{"query":"alpha"}') * 4)


def test_local_prompt_encoder_versions_and_passes_tool_schema() -> None:
    class _Tokenizer:
        special_tokens_map: ClassVar[dict[str, str]] = {}
        chat_template = "tool-template"

        def __init__(self) -> None:
            self.tools: object = None

        def get_vocab(self) -> dict[str, int]:
            return {"token": 1}

        def apply_chat_template(
            self,
            messages: list[dict[str, object]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            tools: object = None,
        ) -> list[int]:
            assert messages and tokenize and add_generation_prompt
            self.tools = tools
            return [1, 2, 3]

    tokenizer = _Tokenizer()
    encoder = LocalPromptEncoder(tokenizer, LocalLoraConfig())
    messages = (Message(role="user", content="question"),)
    plain = encoder.encode(messages)
    with_tools = encoder.encode(messages, MAIN_TOOLS)

    assert tokenizer.tools is not None
    assert with_tools.prompt_ids == (1, 2, 3)
    assert with_tools.prompt_template_revision != plain.prompt_template_revision
