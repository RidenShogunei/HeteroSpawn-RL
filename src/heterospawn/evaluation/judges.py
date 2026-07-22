"""Provider-neutral xbench judge contract and MiniMax development adapter."""

# The pinned upstream prompt is intentionally kept byte-for-byte, including long lines
# and full-width punctuation.
# ruff: noqa: E501, RUF001

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from heterospawn.domain.ids import TaskId
from heterospawn.errors import JudgeRequestError
from heterospawn.policies.base import JsonScalar, Message, TokenUsage
from heterospawn.policies.minimax import MiniMaxChatClient, MiniMaxChatRequest

Clock = Callable[[], float]

_XBENCH_JUDGE_PROMPT = """
你是一个通用人工智能助手。根据下面给出的[正确答案], 判断以下对[原问题]的[回答]的回答是否正确。

[原问题]: {question}

[正确答案]: {correct_answer}

[回答]:{response}

你的判断必须按照以下格式和标准进行:
最终答案: 从[回答]中提取出的最终准确答案。如果[回答]中没有明确的最终答案, 则填写'无'。
解释: 根据[正确]解释为什么[最终答案]是正确的或错误的。只关注[最终答案]与[正确答案]之间是否存在实质性差异, 不要评论题目的背景, 不要尝试重新解题, 不要为任何不同于[正确答案]的答案辩护, 只专注于判断答案是否一致。
结论: 如果[最终答案]与上方给出的[正确答案]一致, 或者在数值题目中处于可接受的微小误差范围内, 则填写'正确'; 否则（即存在任何不一致、歧义、不等价或提取出的答案错误的情况）填写'错误'。
""".strip()
XBENCH_JUDGE_PROMPT_REVISION = (
    "xbench-evals@17c562192cc7e62215bfb98b65e9f8806fb95504:"
    + hashlib.sha256(_XBENCH_JUDGE_PROMPT.encode("utf-8")).hexdigest()
)
_FORMAT_REPAIR_PROMPT = """The verdict format was invalid. Return exactly three lines:
最终答案:<从回答提取的答案或无>
解释:<简短解释>
结论: 正确或错误
Do not add markdown or any other text."""
MINIMAX_JUDGE_REPAIR_REVISION = hashlib.sha256(_FORMAT_REPAIR_PROMPT.encode("utf-8")).hexdigest()


class JudgeRevision(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    mode: Literal["minimax-development", "gemini-official"]
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_revision: str = Field(min_length=1)
    sampling_params: tuple[tuple[str, JsonScalar], ...]
    max_format_repair_attempts: int = Field(ge=0)
    format_repair_revision: str | None
    comparable_to_official: bool


class JudgeRequest(BaseModel):
    """Sensitive in-memory evaluator input; never include this model in reports."""

    model_config = ConfigDict(frozen=True, strict=True)

    request_id: str = Field(min_length=1)
    task_id: TaskId
    question: str = Field(min_length=1)
    correct_answer: str = Field(min_length=1)
    response: str = Field(min_length=1)


class JudgeResult(BaseModel):
    """Parsed result with text reduced to digests before crossing the adapter boundary."""

    model_config = ConfigDict(frozen=True, strict=True)

    request_id: str
    correct: bool
    extracted_answer_digest: str
    reason_digest: str
    provider_response_digest: str
    usage: TokenUsage
    latency_ms: int = Field(ge=0)


class JudgeService(Protocol):
    @property
    def revision(self) -> JudgeRevision: ...

    async def judge(self, request: JudgeRequest) -> JudgeResult: ...


class MiniMaxDevelopmentJudge:
    """Runs the pinned xbench judge prompt through MiniMax for development only."""

    _SAMPLING_PARAMS: tuple[tuple[str, JsonScalar], ...] = (
        ("temperature", 1.0),
        ("top_p", 0.95),
        ("max_completion_tokens", 2048),
        ("reasoning_split", True),
    )

    def __init__(self, chat: MiniMaxChatClient, *, clock: Clock = time.monotonic) -> None:
        self._chat = chat
        self._clock = clock
        self._revision = JudgeRevision(
            mode="minimax-development",
            provider=chat.revision.provider,
            model=chat.revision.model,
            prompt_revision=XBENCH_JUDGE_PROMPT_REVISION,
            sampling_params=self._SAMPLING_PARAMS,
            max_format_repair_attempts=1,
            format_repair_revision=MINIMAX_JUDGE_REPAIR_REVISION,
            comparable_to_official=False,
        )

    @property
    def revision(self) -> JudgeRevision:
        return self._revision

    async def judge(self, request: JudgeRequest) -> JudgeResult:
        prompt = _XBENCH_JUDGE_PROMPT.format(
            question=request.question,
            correct_answer=request.correct_answer,
            response=request.response,
        )
        messages: tuple[Message, ...] = (Message(role="user", content=prompt),)
        start = self._clock()
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        response_digests: list[str] = []
        for attempt in range(self._revision.max_format_repair_attempts + 1):
            try:
                result = await self._chat.complete(
                    MiniMaxChatRequest(
                        messages=messages,
                        sampling_params=self._SAMPLING_PARAMS,
                    )
                )
            except Exception:
                raise JudgeRequestError("MiniMax development judge failed") from None
            prompt_tokens += result.usage.prompt_tokens
            completion_tokens += result.usage.completion_tokens
            total_tokens += result.usage.total_tokens
            response_digests.append(result.raw_response_digest)
            try:
                extracted_answer, explanation, correct = parse_xbench_judge_response(result.content)
            except JudgeRequestError:
                if attempt == self._revision.max_format_repair_attempts:
                    raise JudgeRequestError("MiniMax development judge failed") from None
                messages = (
                    *messages,
                    Message(role="assistant", content=result.content),
                    Message(role="user", content=_FORMAT_REPAIR_PROMPT),
                )
                continue
            return JudgeResult(
                request_id=request.request_id,
                correct=correct,
                extracted_answer_digest=_text_digest(extracted_answer),
                reason_digest=_text_digest(explanation),
                provider_response_digest=_text_digest(":".join(response_digests)),
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                ),
                latency_ms=max(0, round((self._clock() - start) * 1000)),
            )

        raise AssertionError("judge repair loop must return or raise")


def parse_xbench_judge_response(response: str) -> tuple[str, str, bool]:
    """Mirror the pinned upstream regex/parser behavior and require all fields."""

    extracted_answer = _parse_match(re.search(r"最终答案:*(.*)", response))
    conclusion = _parse_match(re.search(r"结论:*.(正确|错误)", response))
    explanation = _parse_match(re.search(r"解释:*(.*)", response))
    if extracted_answer is None or conclusion not in {"正确", "错误"} or explanation is None:
        raise JudgeRequestError("xbench judge response is missing required fields")
    return extracted_answer, explanation, conclusion == "正确"


def _parse_match(match: re.Match[str] | None) -> str | None:
    if match is None:
        return None
    matched_text = match.group(0)
    try:
        return matched_text.split(":")[1].strip()
    except IndexError:
        return matched_text


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
