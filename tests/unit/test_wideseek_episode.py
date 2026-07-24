from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import pytest

from heterospawn.domain.ids import EpisodeId, PolicyId, RolloutId, TaskId
from heterospawn.domain.tasks import ResearchTask
from heterospawn.domain.training import (
    GenerationRequest,
    GenerationResult,
    PromptEncoding,
    canonical_digest,
)
from heterospawn.domain.versions import RoleBinding, RolloutRevision
from heterospawn.orchestration import WideSeekEpisodeOrchestrator
from heterospawn.policies.base import Message
from heterospawn.policies.trainable import ToolDefinition
from heterospawn.search.base import (
    AccessRequest,
    AccessResponse,
    SearchItem,
    SearchRequest,
    SearchResponse,
)
from heterospawn.training import MockTrainingBackend, PolicyRegistry, TrainingBatchBuilder

ResponseScript = Callable[[GenerationRequest], str]


def _call(name: str, arguments: dict[str, object]) -> str:
    return (
        "<tool_call>"
        + json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"))
        + "</tool_call>"
    )


class _Utf8ToolCodec:
    def __init__(self) -> None:
        self.tool_sets: list[tuple[str, ...]] = []

    def encode(
        self,
        messages: tuple[Message, ...],
        tools: tuple[ToolDefinition, ...] = (),
    ) -> PromptEncoding:
        self.tool_sets.append(tuple(tool.name for tool in tools))
        payload = {
            "messages": [message.model_dump(mode="json") for message in messages],
            "tools": [tool.model_dump(mode="json") for tool in tools],
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        return PromptEncoding(
            prompt_ids=tuple(encoded),
            tokenizer_revision="utf8-tools-v1",
            prompt_template_revision=canonical_digest(payload["tools"]),
        )

    def decode(self, response_ids: tuple[int, ...]) -> str:
        return bytes(response_ids).decode()


class _ScriptedPolicy:
    def __init__(
        self,
        backend: MockTrainingBackend,
        policy_id: PolicyId,
        script: ResponseScript,
    ) -> None:
        self._backend = backend
        self._policy_id = policy_id
        self._script = script
        self.stop_reason = "eos"
        self.results: dict[str, GenerationResult] = {}

    @property
    def policy_id(self) -> PolicyId:
        return self._policy_id

    async def current_rollout_revision(self) -> RolloutRevision:
        return self._backend.rollout_revision(self._policy_id)

    async def generate(
        self,
        request: GenerationRequest,
        expected_revision: RolloutRevision,
    ) -> GenerationResult:
        if expected_revision != self._backend.rollout_revision(self._policy_id):
            raise RuntimeError("stale scripted revision")
        content = self._script(request)
        response_ids = tuple(content.encode())
        result = GenerationResult(
            request_id=request.request_id,
            policy_id=self._policy_id,
            rollout_revision=expected_revision,
            response_ids=response_ids,
            response_log_probs=tuple(-0.01 for _ in response_ids),
            stop_reason=self.stop_reason,
        )
        self.results[request.request_id] = result
        return result


class _Tools:
    async def search(self, request: SearchRequest) -> SearchResponse:
        await asyncio.sleep(0.002 if request.query.endswith("0") else 0)
        if request.query == "fail-search":
            raise RuntimeError("synthetic search failure")
        url = f"https://docs/{request.query}"
        return SearchResponse(
            request_id=request.request_id,
            provider="memory",
            provider_revision="memory@1",
            provider_request_id=f"provider:{request.request_id}",
            query=request.query,
            results=(
                SearchItem(
                    title=request.query,
                    url=url,
                    content=f"snippet:{request.query}",
                    score=1.0,
                ),
            ),
            raw_response_digest=canonical_digest({"query": request.query, "url": url}),
        )

    async def access(self, request: AccessRequest) -> AccessResponse:
        return AccessResponse(
            request_id=request.request_id,
            provider="memory",
            provider_revision="memory@1",
            provider_request_id=f"provider:{request.request_id}",
            url=request.url,
            content=f"page:{request.url}:{request.info_to_extract}",
            truncated=False,
            raw_response_digest=canonical_digest(
                {"url": request.url, "extract": request.info_to_extract}
            ),
        )


def _setup(
    main_script: ResponseScript,
    sub_script: ResponseScript,
    **kwargs: object,
) -> tuple[
    WideSeekEpisodeOrchestrator,
    PolicyRegistry,
    _ScriptedPolicy,
    _ScriptedPolicy,
    _Utf8ToolCodec,
    _Utf8ToolCodec,
]:
    main_id = PolicyId("main")
    sub_id = PolicyId("sub")
    backend = MockTrainingBackend((main_id, sub_id))
    registry = PolicyRegistry(
        (
            RoleBinding(role="main", policy_id=main_id, trainable=True),
            RoleBinding(role="sub", policy_id=sub_id, trainable=True),
        ),
        (
            (main_id, backend.rollout_revision(main_id)),
            (sub_id, backend.rollout_revision(sub_id)),
        ),
    )
    main = _ScriptedPolicy(backend, main_id, main_script)
    sub = _ScriptedPolicy(backend, sub_id, sub_script)
    main_codec = _Utf8ToolCodec()
    sub_codec = _Utf8ToolCodec()
    orchestrator = WideSeekEpisodeOrchestrator(
        registry,
        {"main": main, "sub": sub},
        {"main": main_codec, "sub": sub_codec},
        _Tools(),
        max_concurrency=2,
        sampling_params=(("temperature", 0.0),),
        **kwargs,
    )
    return orchestrator, registry, main, sub, main_codec, sub_codec


def test_tool_message_budgets_are_deterministic_and_versioned() -> None:
    orchestrator, *_ = _setup(
        lambda request: "answer",
        lambda request: "summary",
        max_search_message_results=2,
        max_search_content_characters=6,
        max_access_characters=7,
    )
    other, *_ = _setup(
        lambda request: "answer",
        lambda request: "summary",
        max_search_content_characters=8,
        max_access_characters=7,
    )
    response = SearchResponse(
        request_id="search",
        provider="memory",
        provider_revision="memory@1",
        provider_request_id="provider:search",
        query="query",
        results=(
            SearchItem(title="one", url="https://docs/one", content="abcdefgh", score=1.0),
            SearchItem(title="two", url="https://docs/two", content="ijklmnop", score=0.5),
        ),
        raw_response_digest="digest",
    )

    displayed = orchestrator._search_message_results(response)

    assert [item["content"] for item in displayed] == ["abc", "ijk"]
    assert all(item["content_truncated"] is True for item in displayed)
    assert orchestrator.prompt_revision != other.prompt_revision


@pytest.mark.asyncio
async def test_direct_answer_is_zero_spawn() -> None:
    orchestrator, registry, main, _, main_codec, _ = _setup(
        lambda request: "direct answer",
        lambda request: "unused",
    )
    trace = await orchestrator.run(
        ResearchTask(task_id=TaskId("direct"), prompt="question"),
        EpisodeId("episode-direct"),
        RolloutId("rollout-direct"),
        registry.snapshot(),
    )

    assert trace.status == "success"
    assert trace.answer == "direct answer"
    assert trace.spawn_count == 0
    assert trace.spawn_rounds == ()
    assert len(trace.model_steps) == 1
    assert (
        trace.model_steps[0].response_ids
        == main.results[str(trace.model_steps[0].step_id)].response_ids
    )
    assert main_codec.tool_sets == [("subtask",)]


@pytest.mark.asyncio
async def test_length_truncated_direct_answer_is_invalid_and_retried() -> None:
    orchestrator, registry, main, _, _, _ = _setup(
        lambda request: "unfinished direct answer",
        lambda request: "unused",
    )
    main.stop_reason = "length"

    trace = await orchestrator.run(
        ResearchTask(task_id=TaskId("truncated"), prompt="question"),
        EpisodeId("episode-truncated"),
        RolloutId("rollout-truncated"),
        registry.snapshot(),
    )

    assert trace.status == "failed"
    assert trace.failure_code == "invalid_main_action"
    assert trace.invalid_main_attempts == 2
    assert all(
        attempt.error_code == "truncated Main output cannot be ANSWER"
        for attempt in trace.main_attempts
    )


@pytest.mark.asyncio
async def test_multi_round_spawn_search_access_provenance_and_stable_order() -> None:
    def main_script(request: GenerationRequest) -> str:
        if "round-0:attempt-0" in request.request_id:
            return _call("subtask", {"subtask": ""})
        if "round-0:attempt-1" in request.request_id:
            return "".join(_call("subtask", {"subtask": f"task-{index}"}) for index in range(4))
        if "round-1" in request.request_id:
            return _call("subtask", {"subtask": "task-4"})
        return "integrated final answer"

    def sub_script(request: GenerationRequest) -> str:
        instance = str(request.agent_instance_id)
        index = int(instance.rsplit("-", 1)[1])
        if "turn-0" in request.request_id:
            query = "fail-search" if index == 2 else f"q{index}"
            return _call("search", {"query": query, "topk": 2})
        if "turn-1" in request.request_id:
            url = "https://forged.invalid" if index == 0 else f"https://docs/q{index}"
            return _call("access", {"url": url, "info_to_extract": "facts"})
        return f"summary-{index}"

    orchestrator, registry, main, sub, main_codec, sub_codec = _setup(
        main_script,
        sub_script,
    )
    trace = await orchestrator.run(
        ResearchTask(task_id=TaskId("multi"), prompt="question"),
        EpisodeId("episode-multi"),
        RolloutId("rollout-multi"),
        registry.snapshot(),
    )

    assert trace.status == "success"
    assert trace.answer == "integrated final answer"
    assert trace.spawn_count == 5
    assert [len(item.agent_instance_ids) for item in trace.spawn_rounds] == [4, 1]
    assert trace.invalid_main_attempts == 1
    assert trace.failed_subs == 0
    assert tuple(event.event_index for event in trace.events) == tuple(range(len(trace.events)))
    assert [str(outcome.agent_instance_id) for outcome in trace.sub_outcomes] == [
        "sub-r0-0",
        "sub-r0-1",
        "sub-r0-2",
        "sub-r0-3",
        "sub-r1-4",
    ]
    forged = next(
        outcome for outcome in trace.tool_outcomes if outcome.url == "https://forged.invalid"
    )
    assert forged.status == "failed"
    assert forged.error_code == "access_url_not_discovered"
    successful_accesses = [
        outcome
        for outcome in trace.tool_outcomes
        if outcome.tool_name == "access" and outcome.status == "success"
    ]
    assert successful_accesses
    assert all(outcome.source_search_step_id is not None for outcome in successful_accesses)
    assert any(
        outcome.error_code == "RuntimeError" and outcome.tool_name == "search"
        for outcome in trace.tool_outcomes
    )
    for outcome in trace.tool_outcomes:
        assert canonical_digest(json.loads(outcome.request_json)) == outcome.request_digest
        assert canonical_digest(json.loads(outcome.result_json)) == outcome.result_digest
    assert all(outcome.status == "success" for outcome in trace.sub_outcomes)
    assert all(tool_set == ("subtask",) for tool_set in main_codec.tool_sets)
    assert all(tool_set == ("search", "access") for tool_set in sub_codec.tool_sets)

    generated = {**main.results, **sub.results}
    for step in trace.model_steps:
        assert step.response_ids == generated[str(step.step_id)].response_ids
        assert step.response_log_probs == generated[str(step.step_id)].response_log_probs

    builder = TrainingBatchBuilder()
    batches = {}
    for phase, policy_id in (
        ("main_update", PolicyId("main")),
        ("sub_update", PolicyId("sub")),
    ):
        batch = builder.build(
            batch_id=f"batch:{phase}",
            phase=phase,
            target_policy_id=policy_id,
            expected_base_version=registry.revision(policy_id).weight_version,
            steps=trace.model_steps,
            episode_advantages={trace.episode_id: 1.0},
        )
        batches[phase] = batch
        for sample in batch.samples:
            source = generated[str(sample.source_step_id)]
            assert sample.response_ids == source.response_ids
            assert sample.old_log_probs == source.response_log_probs
    assert all(
        sample.aggregation_weight == pytest.approx(0.25)
        for sample in batches["main_update"].samples
    )
    assert all(
        sample.aggregation_weight == pytest.approx(1 / 15)
        for sample in batches["sub_update"].samples
    )


@pytest.mark.asyncio
async def test_failed_sub_does_not_cancel_sibling() -> None:
    def main_script(request: GenerationRequest) -> str:
        if "round-0" in request.request_id:
            return _call("subtask", {"subtask": "ok"}) + _call("subtask", {"subtask": "boom"})
        return "answer despite one failed worker"

    def sub_script(request: GenerationRequest) -> str:
        if str(request.agent_instance_id) == "sub-r0-1":
            raise RuntimeError("synthetic policy failure")
        return "usable evidence"

    orchestrator, registry, _, _, _, _ = _setup(main_script, sub_script)
    trace = await orchestrator.run(
        ResearchTask(task_id=TaskId("partial"), prompt="question"),
        EpisodeId("episode-partial"),
        RolloutId("rollout-partial"),
        registry.snapshot(),
    )

    assert trace.status == "success"
    assert trace.spawn_count == 2
    assert [outcome.status for outcome in trace.sub_outcomes] == ["success", "failed"]
    assert trace.sub_outcomes[1].error_code == "RuntimeError"
    assert any(event.kind == "sub_failure" for event in trace.events)


@pytest.mark.asyncio
async def test_episode_spawn_limit_violation_is_retained_and_repaired() -> None:
    def main_script(request: GenerationRequest) -> str:
        if "round-0" in request.request_id:
            return "".join(_call("subtask", {"subtask": f"first-{index}"}) for index in range(4))
        if "round-1:attempt-0" in request.request_id:
            return "".join(_call("subtask", {"subtask": f"overflow-{index}"}) for index in range(4))
        if "round-1:attempt-1" in request.request_id:
            return "".join(_call("subtask", {"subtask": f"allowed-{index}"}) for index in range(2))
        return "answer"

    orchestrator, registry, _, _, _, _ = _setup(
        main_script,
        lambda request: "summary",
        max_spawn_per_episode=6,
    )
    trace = await orchestrator.run(
        ResearchTask(task_id=TaskId("budget"), prompt="question"),
        EpisodeId("episode-budget"),
        RolloutId("rollout-budget"),
        registry.snapshot(),
    )

    assert trace.status == "success"
    assert trace.spawn_count == 6
    assert trace.invalid_main_attempts == 1
    overflow = next(
        attempt
        for attempt in trace.main_attempts
        if attempt.round_index == 1 and attempt.attempt_index == 0
    )
    assert overflow.valid is False
    assert overflow.error_code == "Main spawn exceeds episode limit"
