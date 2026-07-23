"""Provider-neutral semantic judging with revision-bound safe caching."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from heterospawn.domain.ids import TaskId
from heterospawn.domain.training import canonical_digest
from heterospawn.errors import JudgeRequestError, ProviderRequestError
from heterospawn.policies.base import Message, TokenUsage
from heterospawn.policies.minimax import MiniMaxChatClient, MiniMaxChatRequest

SemanticJudgeOperation = Literal[
    "answer_equivalence",
    "cell_equivalence",
    "key_equivalence",
    "column_equivalence",
]


class SemanticJudgeRevision(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    mode: Literal["minimax-development", "fake"]
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_revision: str = Field(min_length=1)
    cache_revision: Literal["heterospawn-semantic-judge-cache-v1"] = (
        "heterospawn-semantic-judge-cache-v1"
    )
    sampling_params: tuple[tuple[str, None | bool | int | float | str], ...]
    max_format_attempts: int = Field(ge=1)
    comparable_to_official: Literal[False] = False


class SemanticJudgeRequest(BaseModel):
    """Sensitive evaluator input that must never be persisted in reports or caches."""

    model_config = ConfigDict(frozen=True, strict=True)

    request_id: str = Field(min_length=1)
    task_id: TaskId
    operation: SemanticJudgeOperation
    question: str = Field(min_length=1)
    candidates: tuple[str, ...] = Field(min_length=1)
    references: tuple[str, ...] = Field(min_length=1)

    def model_post_init(self, __context: object) -> None:
        if len(self.candidates) != len(self.references):
            raise ValueError("semantic Judge candidate/reference pairs must align")


class SemanticJudgeResult(BaseModel):
    """Safe binary scores with no question, candidate, reference, or raw response."""

    model_config = ConfigDict(frozen=True, strict=True)

    request_id: str
    scores: tuple[Literal[0, 1], ...] = Field(min_length=1)
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    cache_hit: bool
    provider_response_digest: str = Field(min_length=1)
    usage: TokenUsage


class SemanticJudge(Protocol):
    @property
    def revision(self) -> SemanticJudgeRevision: ...

    async def judge(self, request: SemanticJudgeRequest) -> SemanticJudgeResult: ...


class SemanticJudgeCache:
    """Digest-only cache; evaluator plaintext is deliberately unrecoverable."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._values: dict[str, tuple[tuple[Literal[0, 1], ...], str]] = {}
        if path is not None and path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                self._values = {
                    key: (
                        tuple(value["scores"]),
                        str(value["provider_response_digest"]),
                    )
                    for key, value in payload.items()
                }
            except (OSError, ValueError, TypeError, KeyError):
                raise JudgeRequestError("semantic Judge cache is corrupt") from None

    def get(self, key: str) -> tuple[tuple[Literal[0, 1], ...], str] | None:
        return self._values.get(key)

    def put(
        self,
        key: str,
        scores: tuple[Literal[0, 1], ...],
        provider_response_digest: str,
    ) -> None:
        existing = self._values.get(key)
        value = (scores, provider_response_digest)
        if existing is not None and existing != value:
            raise JudgeRequestError("semantic Judge cache key has conflicting results")
        self._values[key] = value
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                cache_key: {
                    "scores": list(cache_value[0]),
                    "provider_response_digest": cache_value[1],
                }
                for cache_key, cache_value in sorted(self._values.items())
            }
            temporary = self._path.with_name(f".{self._path.name}.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            temporary.replace(self._path)


_SEMANTIC_PROMPT = """You are an evaluation assistant. Compare each candidate/reference pair for
semantic equivalence under the requested operation. Return only one JSON object:
{"scores":[0,1,...]}
Each score must be 1 when the pair identifies the same answer/entity/value and 0 otherwise. The
number and order of scores must exactly match the input pairs."""
SEMANTIC_PROMPT_REVISION = canonical_digest(
    {
        "upstream": "RLinf@d9f3d8a9db4d7aad1d641029293295503dd3eb2c",
        "prompt": _SEMANTIC_PROMPT,
        "schema": {"scores": "0|1[]"},
    }
)


class MiniMaxSemanticJudge:
    """Temperature-zero, concurrency-bounded, non-official development Judge."""

    _SAMPLING = (
        ("temperature", 0.0),
        ("top_p", 1.0),
        ("max_completion_tokens", 1024),
        ("reasoning_split", False),
    )

    def __init__(
        self,
        chat: MiniMaxChatClient,
        *,
        cache: SemanticJudgeCache | None = None,
        max_concurrency: int = 4,
        max_format_attempts: int = 2,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("semantic Judge max_concurrency must be positive")
        if max_format_attempts < 1:
            raise ValueError("semantic Judge max_format_attempts must be positive")
        self._chat = chat
        self._cache = cache or SemanticJudgeCache()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._revision = SemanticJudgeRevision(
            mode="minimax-development",
            provider=chat.revision.provider,
            model=chat.revision.model,
            prompt_revision=SEMANTIC_PROMPT_REVISION,
            sampling_params=self._SAMPLING,
            max_format_attempts=max_format_attempts,
        )

    @property
    def revision(self) -> SemanticJudgeRevision:
        return self._revision

    async def judge(self, request: SemanticJudgeRequest) -> SemanticJudgeResult:
        cache_key = canonical_digest(
            {
                "revision": self._revision.model_dump(mode="json"),
                "operation": request.operation,
                "question": request.question,
                "candidates": request.candidates,
                "references": request.references,
            }
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return SemanticJudgeResult(
                request_id=request.request_id,
                scores=cached[0],
                cache_key=cache_key,
                cache_hit=True,
                provider_response_digest=cached[1],
                usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )

        pairs = [
            {"idx": index, "candidate": candidate, "reference": reference}
            for index, (candidate, reference) in enumerate(
                zip(request.candidates, request.references, strict=True)
            )
        ]
        messages: tuple[Message, ...] = (
            Message(role="system", content=_SEMANTIC_PROMPT),
            Message(
                role="user",
                content=json.dumps(
                    {
                        "operation": request.operation,
                        "question": request.question,
                        "pairs": pairs,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ),
        )
        prompt_tokens = completion_tokens = total_tokens = 0
        response_digest = ""
        async with self._semaphore:
            for attempt in range(self._revision.max_format_attempts):
                try:
                    result = await self._chat.complete(
                        MiniMaxChatRequest(
                            messages=messages,
                            sampling_params=self._revision.sampling_params,
                        )
                    )
                except ProviderRequestError as exc:
                    raise JudgeRequestError("semantic Judge provider request failed") from exc
                prompt_tokens += result.usage.prompt_tokens
                completion_tokens += result.usage.completion_tokens
                total_tokens += result.usage.total_tokens
                response_digest = result.raw_response_digest
                scores = _parse_scores(result.content, len(request.candidates))
                if scores is not None:
                    self._cache.put(cache_key, scores, response_digest)
                    return SemanticJudgeResult(
                        request_id=request.request_id,
                        scores=scores,
                        cache_key=cache_key,
                        cache_hit=False,
                        provider_response_digest=response_digest,
                        usage=TokenUsage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                        ),
                    )
                messages = (
                    *messages,
                    Message(role="assistant", content=result.content),
                    Message(
                        role="user",
                        content=(
                            f"Invalid schema on attempt {attempt + 1}. Return only "
                            f'{{"scores":[...]}} with {len(request.candidates)} binary values.'
                        ),
                    ),
                )
        raise JudgeRequestError("semantic Judge returned invalid output after bounded repairs")


def _parse_scores(
    content: str,
    expected: int,
) -> tuple[Literal[0, 1], ...] | None:
    stripped = content.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != {"scores"}:
        return None
    scores = payload["scores"]
    if (
        not isinstance(scores, list)
        or len(scores) != expected
        or any(type(score) is not int or score not in (0, 1) for score in scores)
    ):
        return None
    return tuple(scores)
