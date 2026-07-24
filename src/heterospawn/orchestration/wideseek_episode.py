"""Multi-round WideSeek Main/Sub orchestration with exact trainable trajectories."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from heterospawn.domain.ids import (
    AgentInstanceId,
    EpisodeId,
    PolicyId,
    RolloutId,
    StepId,
)
from heterospawn.domain.tasks import ResearchTask
from heterospawn.domain.training import (
    GenerationRequest,
    GenerationResult,
    JsonScalar,
    PromptEncoding,
    TrajectoryStep,
    canonical_digest,
)
from heterospawn.domain.versions import AgentRole, RolloutRevision
from heterospawn.errors import ConfigurationError, InvalidActionError
from heterospawn.orchestration.budget import ConcurrencyBudgetLedger
from heterospawn.orchestration.trainable_models import (
    EvidenceRecord,
    SpawnRoundRecord,
    ToolOutcomeRecord,
    TrainableEnvironmentSnapshot,
    TrainableEpisodeEvent,
    TrainableEpisodeTrace,
    TrainableMainAttempt,
    TrainableSubOutcome,
)
from heterospawn.orchestration.wideseek_actions import (
    MAIN_TOOLS,
    SUB_TOOLS,
    WIDESEEK_PARSER_REVISION,
    WIDESEEK_TOOL_SCHEMA_REVISION,
    WIDESEEK_UPSTREAM_REVISION,
    AccessToolCall,
    MainAnswerTurn,
    MainSpawnTurn,
    SearchToolCall,
    SubSummaryTurn,
    SubToolCall,
    SubToolsTurn,
    parse_main_turn,
    parse_sub_turn,
)
from heterospawn.policies.base import Message
from heterospawn.policies.trainable import TrainablePolicyCodec
from heterospawn.search.base import (
    AccessRequest,
    AccessResponse,
    ResearchToolService,
    SearchRequest,
    SearchResponse,
)
from heterospawn.training.base import PolicyService
from heterospawn.training.registry import PolicyRegistry

_MAIN_SYSTEM_PROMPT = """You are the lead researcher. Use the subtask tool only when delegation is
needed. You may emit 1-4 subtask tool calls in one turn. When sufficient evidence is available,
return the final answer directly with no tool call. Never invent tool results."""

_SUB_SYSTEM_PROMPT = """You are a research worker. Use search to discover sources and access to read
only URLs returned by your own earlier searches. You may emit at most three tool calls in one turn.
When the subtask is complete, return a concise evidence summary directly with no tool call."""
WIDESEEK_PROMPT_REVISION = canonical_digest(
    {
        "upstream_revision": WIDESEEK_UPSTREAM_REVISION,
        "main_system_prompt": _MAIN_SYSTEM_PROMPT,
        "sub_system_prompt": _SUB_SYSTEM_PROMPT,
    }
)


@dataclass(frozen=True)
class _ToolExecution:
    event: TrainableEpisodeEvent
    outcome: ToolOutcomeRecord
    message_payload: dict[str, object]
    discovered_urls: tuple[str, ...] = ()


@dataclass(frozen=True)
class _SubExecution:
    outcome: TrainableSubOutcome
    evidence: EvidenceRecord | None
    model_steps: tuple[TrajectoryStep, ...]
    events: tuple[TrainableEpisodeEvent, ...]
    tool_outcomes: tuple[ToolOutcomeRecord, ...]
    final_step_id: StepId


class WideSeekEpisodeOrchestrator:
    """Runs bounded WideSeek-style multi-round episodes without backend coupling."""

    def __init__(
        self,
        registry: PolicyRegistry,
        policy_services: Mapping[AgentRole, PolicyService],
        codecs: Mapping[AgentRole, TrainablePolicyCodec],
        tools: ResearchToolService,
        *,
        max_concurrency: int = 4,
        max_main_rounds: int = 3,
        max_sub_turns: int = 4,
        max_spawn_per_round: int = 4,
        max_spawn_per_episode: int = 8,
        max_tools_per_sub_turn: int = 3,
        main_repair_attempts: int = 1,
        sub_repair_attempts: int = 1,
        max_search_message_results: int = 3,
        max_search_content_characters: int = 3000,
        max_access_characters: int = 2000,
        sampling_params: tuple[tuple[str, JsonScalar], ...] = (),
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if max_main_rounds < 1:
            raise ValueError("max_main_rounds must be positive")
        if max_sub_turns < 1:
            raise ValueError("max_sub_turns must be positive")
        if not 1 <= max_spawn_per_round <= 4:
            raise ValueError("max_spawn_per_round must be in 1..4")
        if max_spawn_per_episode < 1:
            raise ValueError("max_spawn_per_episode must be positive")
        if not 1 <= max_tools_per_sub_turn <= 3:
            raise ValueError("max_tools_per_sub_turn must be in 1..3")
        if main_repair_attempts < 0 or sub_repair_attempts < 0:
            raise ValueError("repair attempts cannot be negative")
        if max_search_message_results < 1:
            raise ValueError("model-visible search result count must be positive")
        if max_search_content_characters < 1 or max_access_characters < 1:
            raise ValueError("tool message character budgets must be positive")
        for role in ("main", "sub"):
            binding = registry.binding(role)
            service = policy_services.get(role)
            if service is None or service.policy_id != binding.policy_id:
                raise ConfigurationError(f"{role} policy service does not match its role binding")
            if role not in codecs:
                raise ConfigurationError(f"{role} trainable policy codec is required")

        self._registry = registry
        self._services = dict(policy_services)
        self._codecs = dict(codecs)
        self._tools = tools
        self._max_concurrency = max_concurrency
        self._max_main_rounds = max_main_rounds
        self._max_sub_turns = max_sub_turns
        self._max_spawn_per_round = max_spawn_per_round
        self._max_spawn_per_episode = max_spawn_per_episode
        self._max_tools_per_sub_turn = max_tools_per_sub_turn
        self._main_repair_attempts = main_repair_attempts
        self._sub_repair_attempts = sub_repair_attempts
        self._max_search_message_results = max_search_message_results
        self._max_search_content_characters = max_search_content_characters
        self._max_access_characters = max_access_characters
        self._sampling_params = sampling_params
        self.prompt_revision = canonical_digest(
            {
                "base_revision": WIDESEEK_PROMPT_REVISION,
                "max_search_message_results": max_search_message_results,
                "max_search_content_characters": max_search_content_characters,
                "max_access_characters": max_access_characters,
            }
        )

    async def run(
        self,
        task: ResearchTask,
        episode_id: EpisodeId,
        rollout_id: RolloutId,
        policy_revisions: tuple[tuple[PolicyId, RolloutRevision], ...],
    ) -> TrainableEpisodeTrace:
        revisions = self._validate_revisions(policy_revisions)
        ledger = ConcurrencyBudgetLedger(self._max_concurrency)
        tool_semaphore = asyncio.Semaphore(self._max_concurrency)
        events: list[TrainableEpisodeEvent] = []
        model_steps: list[TrajectoryStep] = []
        attempts: list[TrainableMainAttempt] = []
        spawn_rounds: list[SpawnRoundRecord] = []
        sub_outcomes: list[TrainableSubOutcome] = []
        evidence: list[EvidenceRecord] = []
        tool_outcomes: list[ToolOutcomeRecord] = []
        messages: tuple[Message, ...] = (
            Message(role="system", content=self._main_prompt),
            Message(role="user", content=task.prompt),
        )
        main_causes: tuple[StepId, ...] = ()
        total_spawned = 0

        for round_index in range(self._max_main_rounds):
            allow_spawn = round_index < self._max_main_rounds - 1
            turn, main_step_id, accepted_content = await self._generate_main_turn(
                task=task,
                episode_id=episode_id,
                rollout_id=rollout_id,
                revisions=revisions,
                round_index=round_index,
                messages=messages,
                causal_step_ids=main_causes,
                allow_spawn=allow_spawn,
                remaining_spawn_slots=self._max_spawn_per_episode - total_spawned,
                ledger=ledger,
                events=events,
                model_steps=model_steps,
                attempts=attempts,
            )
            if turn is None:
                return self._trace(
                    task,
                    episode_id,
                    rollout_id,
                    policy_revisions,
                    status="failed",
                    failure_code="invalid_main_action",
                    attempts=attempts,
                    spawn_rounds=spawn_rounds,
                    sub_outcomes=sub_outcomes,
                    evidence=evidence,
                    tool_outcomes=tool_outcomes,
                    model_steps=model_steps,
                    events=events,
                )
            if isinstance(turn, MainAnswerTurn):
                return self._trace(
                    task,
                    episode_id,
                    rollout_id,
                    policy_revisions,
                    status="success",
                    answer=turn.answer,
                    attempts=attempts,
                    spawn_rounds=spawn_rounds,
                    sub_outcomes=sub_outcomes,
                    evidence=evidence,
                    tool_outcomes=tool_outcomes,
                    model_steps=model_steps,
                    events=events,
                )
            if not isinstance(turn, MainSpawnTurn):
                raise AssertionError("Main parser returned an unsupported turn")

            instance_ids = tuple(
                AgentInstanceId(f"sub-r{round_index}-{total_spawned + index}")
                for index in range(len(turn.subtasks))
            )
            spawn_rounds.append(
                SpawnRoundRecord(
                    round_index=round_index,
                    main_step_id=main_step_id,
                    agent_instance_ids=instance_ids,
                )
            )
            executions = await asyncio.gather(
                *(
                    self._run_sub(
                        task=task,
                        episode_id=episode_id,
                        rollout_id=rollout_id,
                        revisions=revisions,
                        spawn_round=round_index,
                        instance_id=instance_id,
                        subtask=subtask,
                        spawn_step_id=main_step_id,
                        ledger=ledger,
                        tool_semaphore=tool_semaphore,
                    )
                    for instance_id, subtask in zip(instance_ids, turn.subtasks, strict=True)
                )
            )
            final_causes: list[StepId] = []
            for execution in executions:
                self._merge_sub_execution(execution, events, model_steps)
                sub_outcomes.append(execution.outcome)
                tool_outcomes.extend(execution.tool_outcomes)
                if execution.evidence is not None:
                    evidence.append(execution.evidence)
                final_causes.append(execution.final_step_id)
            total_spawned += len(executions)
            result_payload = [
                {
                    "agent_instance_id": str(execution.outcome.agent_instance_id),
                    "subtask": execution.outcome.subtask,
                    "status": execution.outcome.status,
                    "content": execution.outcome.content,
                    "error_code": execution.outcome.error_code,
                }
                for execution in executions
            ]
            messages = (
                *messages,
                Message(role="assistant", content=accepted_content),
                Message(
                    role="user",
                    content=(
                        "Delegated worker results, in request order:\n"
                        + json.dumps(result_payload, ensure_ascii=False, sort_keys=True)
                    ),
                ),
            )
            main_causes = (main_step_id, *final_causes)

        return self._trace(
            task,
            episode_id,
            rollout_id,
            policy_revisions,
            status="failed",
            failure_code="main_round_limit",
            attempts=attempts,
            spawn_rounds=spawn_rounds,
            sub_outcomes=sub_outcomes,
            evidence=evidence,
            tool_outcomes=tool_outcomes,
            model_steps=model_steps,
            events=events,
        )

    async def _generate_main_turn(
        self,
        *,
        task: ResearchTask,
        episode_id: EpisodeId,
        rollout_id: RolloutId,
        revisions: Mapping[PolicyId, RolloutRevision],
        round_index: int,
        messages: tuple[Message, ...],
        causal_step_ids: tuple[StepId, ...],
        allow_spawn: bool,
        remaining_spawn_slots: int,
        ledger: ConcurrencyBudgetLedger,
        events: list[TrainableEpisodeEvent],
        model_steps: list[TrajectoryStep],
        attempts: list[TrainableMainAttempt],
    ) -> tuple[MainAnswerTurn | MainSpawnTurn | None, StepId, str]:
        service = self._services["main"]
        codec = self._codecs["main"]
        expected_revision = revisions[service.policy_id]
        current_messages = messages
        attempt_causes = causal_step_ids
        last_step_id: StepId | None = None
        last_content = ""
        for attempt_index in range(self._main_repair_attempts + 1):
            prompt = codec.encode(current_messages, MAIN_TOOLS)
            step_id = StepId(f"{rollout_id}:main:round-{round_index}:attempt-{attempt_index}")
            request_id = str(step_id)
            result = await service.generate(
                self._generation_request(
                    task,
                    episode_id,
                    rollout_id,
                    request_id,
                    "main",
                    AgentInstanceId("main"),
                    prompt,
                ),
                expected_revision,
            )
            self._validate_generation_result(
                result,
                request_id,
                service.policy_id,
                expected_revision,
            )
            content = codec.decode(result.response_ids)
            step = self._trajectory_step(
                task,
                episode_id,
                rollout_id,
                step_id,
                len(events),
                attempt_causes,
                "main",
                AgentInstanceId("main"),
                prompt,
                result,
                revisions,
            )
            action: MainAnswerTurn | MainSpawnTurn | None = None
            error_code: str | None = None
            try:
                candidate = parse_main_turn(content)
                if result.stop_reason == "length" and isinstance(candidate, MainAnswerTurn):
                    raise InvalidActionError("truncated Main output cannot be ANSWER")
                if isinstance(candidate, MainSpawnTurn):
                    if not allow_spawn:
                        raise InvalidActionError("last Main round must answer")
                    if len(candidate.subtasks) > self._max_spawn_per_round:
                        raise InvalidActionError("Main spawn exceeds per-round limit")
                    if len(candidate.subtasks) > remaining_spawn_slots:
                        raise InvalidActionError("Main spawn exceeds episode limit")
                action = candidate
            except InvalidActionError as exc:
                error_code = str(exc)
            valid = action is not None
            model_steps.append(step)
            events.append(
                self._model_event(
                    step,
                    "valid" if valid else "invalid",
                    self._main_phase(round_index, allow_spawn),
                    await self._environment_snapshot(ledger),
                )
            )
            attempts.append(
                TrainableMainAttempt(
                    phase=self._main_phase(round_index, allow_spawn),
                    round_index=round_index,
                    attempt_index=attempt_index,
                    step_id=step_id,
                    content=content,
                    valid=valid,
                    action_kind=action.kind if action is not None else None,
                    error_code=error_code,
                )
            )
            last_step_id = step_id
            last_content = content
            if action is not None:
                return action, step_id, content
            attempt_causes = (*causal_step_ids, step_id)
            current_messages = (
                *current_messages,
                Message(role="assistant", content=content),
                Message(
                    role="user",
                    content=(
                        "Repair the invalid action. Use 1-4 valid subtask tool calls or return "
                        "a direct non-empty answer with no tool call."
                    ),
                ),
            )
        if last_step_id is None:
            raise AssertionError("at least one Main attempt must execute")
        return None, last_step_id, last_content

    async def _run_sub(
        self,
        *,
        task: ResearchTask,
        episode_id: EpisodeId,
        rollout_id: RolloutId,
        revisions: Mapping[PolicyId, RolloutRevision],
        spawn_round: int,
        instance_id: AgentInstanceId,
        subtask: str,
        spawn_step_id: StepId,
        ledger: ConcurrencyBudgetLedger,
        tool_semaphore: asyncio.Semaphore,
    ) -> _SubExecution:
        service = self._services["sub"]
        codec = self._codecs["sub"]
        expected_revision = revisions[service.policy_id]
        messages: tuple[Message, ...] = (
            Message(role="system", content=_SUB_SYSTEM_PROMPT),
            Message(role="user", content=subtask),
        )
        known_urls: dict[str, StepId] = {}
        events: list[TrainableEpisodeEvent] = []
        model_steps: list[TrajectoryStep] = []
        tool_outcomes: list[ToolOutcomeRecord] = []
        causal_step_ids: tuple[StepId, ...] = (spawn_step_id,)

        try:
            for turn_index in range(self._max_sub_turns):
                parsed: SubSummaryTurn | SubToolsTurn | None = None
                accepted_content = ""
                for attempt_index in range(self._sub_repair_attempts + 1):
                    prompt = codec.encode(messages, SUB_TOOLS)
                    step_id = StepId(
                        f"{rollout_id}:{instance_id}:turn-{turn_index}:attempt-{attempt_index}"
                    )
                    request_id = str(step_id)
                    result = await service.generate(
                        self._generation_request(
                            task,
                            episode_id,
                            rollout_id,
                            request_id,
                            "sub",
                            instance_id,
                            prompt,
                        ),
                        expected_revision,
                    )
                    self._validate_generation_result(
                        result,
                        request_id,
                        service.policy_id,
                        expected_revision,
                    )
                    content = codec.decode(result.response_ids)
                    step = self._trajectory_step(
                        task,
                        episode_id,
                        rollout_id,
                        step_id,
                        0,
                        causal_step_ids,
                        "sub",
                        instance_id,
                        prompt,
                        result,
                        revisions,
                    )
                    model_steps.append(step)
                    try:
                        candidate = parse_sub_turn(content)
                        if result.stop_reason == "length" and isinstance(candidate, SubSummaryTurn):
                            raise InvalidActionError(
                                "truncated Sub output cannot be an evidence summary"
                            )
                        if (
                            isinstance(candidate, SubToolsTurn)
                            and len(candidate.calls) > self._max_tools_per_sub_turn
                        ):
                            raise InvalidActionError("Sub tool calls exceed per-turn limit")
                        parsed = candidate
                    except InvalidActionError:
                        parsed = None
                    events.append(
                        self._model_event(
                            step,
                            "valid" if parsed is not None else "invalid",
                            "sub",
                            await self._environment_snapshot(ledger),
                        )
                    )
                    accepted_content = content
                    if parsed is not None:
                        break
                    causal_step_ids = (*causal_step_ids, step_id)
                    messages = (
                        *messages,
                        Message(role="assistant", content=content),
                        Message(
                            role="user",
                            content=(
                                "Repair the invalid turn. Use 1-3 valid search/access tool calls "
                                "or return a direct non-empty evidence summary."
                            ),
                        ),
                    )

                if parsed is None:
                    return await self._failed_sub(
                        rollout_id,
                        instance_id,
                        subtask,
                        spawn_round,
                        spawn_step_id,
                        "invalid_sub_action",
                        model_steps,
                        events,
                        tool_outcomes,
                        ledger,
                    )
                accepted_model_step_id = model_steps[-1].step_id
                if isinstance(parsed, SubSummaryTurn):
                    producer_tool = next(
                        (
                            outcome.step_id
                            for outcome in reversed(tool_outcomes)
                            if outcome.status == "success"
                        ),
                        None,
                    )
                    outcome = TrainableSubOutcome(
                        agent_instance_id=instance_id,
                        subtask=subtask,
                        spawn_round=spawn_round,
                        status="success",
                        content=parsed.summary,
                        search_step_id=next(
                            (
                                item.step_id
                                for item in tool_outcomes
                                if item.tool_name == "search" and item.status == "success"
                            ),
                            None,
                        ),
                        model_step_id=accepted_model_step_id,
                        tool_step_ids=tuple(item.step_id for item in tool_outcomes),
                        model_step_ids=tuple(step.step_id for step in model_steps),
                    )
                    return _SubExecution(
                        outcome=outcome,
                        evidence=EvidenceRecord(
                            agent_instance_id=instance_id,
                            subtask=subtask,
                            content=parsed.summary,
                            producer_tool_step_id=producer_tool,
                            producer_model_step_id=accepted_model_step_id,
                        ),
                        model_steps=tuple(model_steps),
                        events=tuple(events),
                        tool_outcomes=tuple(tool_outcomes),
                        final_step_id=accepted_model_step_id,
                    )
                if not isinstance(parsed, SubToolsTurn):
                    raise AssertionError("Sub parser returned an unsupported turn")

                known_before_turn = dict(known_urls)
                executions = await asyncio.gather(
                    *(
                        self._execute_tool_call(
                            call=call,
                            request_index=request_index,
                            rollout_id=rollout_id,
                            instance_id=instance_id,
                            spawn_round=spawn_round,
                            sub_turn=turn_index,
                            causal_step_id=accepted_model_step_id,
                            known_urls=known_before_turn,
                            ledger=ledger,
                            semaphore=tool_semaphore,
                        )
                        for request_index, call in enumerate(parsed.calls)
                    )
                )
                tool_payload: list[dict[str, object]] = []
                for execution in executions:
                    events.append(execution.event)
                    tool_outcomes.append(execution.outcome)
                    tool_payload.append(execution.message_payload)
                    for url in execution.discovered_urls:
                        known_urls[url] = execution.outcome.step_id
                messages = (
                    *messages,
                    Message(role="assistant", content=accepted_content),
                    Message(
                        role="user",
                        content=json.dumps(tool_payload, ensure_ascii=False, sort_keys=True),
                    ),
                )
                causal_step_ids = tuple(execution.outcome.step_id for execution in executions)

            return await self._failed_sub(
                rollout_id,
                instance_id,
                subtask,
                spawn_round,
                spawn_step_id,
                "sub_turn_limit",
                model_steps,
                events,
                tool_outcomes,
                ledger,
            )
        except Exception as exc:
            return await self._failed_sub(
                rollout_id,
                instance_id,
                subtask,
                spawn_round,
                spawn_step_id,
                type(exc).__name__,
                model_steps,
                events,
                tool_outcomes,
                ledger,
            )

    async def _execute_tool_call(
        self,
        *,
        call: SubToolCall,
        request_index: int,
        rollout_id: RolloutId,
        instance_id: AgentInstanceId,
        spawn_round: int,
        sub_turn: int,
        causal_step_id: StepId,
        known_urls: Mapping[str, StepId],
        ledger: ConcurrencyBudgetLedger,
        semaphore: asyncio.Semaphore,
    ) -> _ToolExecution:
        tool_name: Literal["search", "access"] = call.name
        step_id = StepId(
            f"{rollout_id}:{instance_id}:turn-{sub_turn}:tool-{request_index}:{tool_name}"
        )
        request_id = str(step_id)
        request_payload = call.model_dump(mode="json")
        request_json = self._canonical_json(request_payload)
        request_digest = canonical_digest(request_payload)
        provider_revision: str | None = None
        provider_response_digest: str | None = None
        result_payload: object = {"error_code": "not_executed"}
        result_digest = canonical_digest(result_payload)
        error_code: str | None = None
        message_payload: dict[str, object]
        discovered_urls: tuple[str, ...] = ()
        source_search_step_id: StepId | None = None
        query: str | None = None
        url: str | None = None

        if isinstance(call, AccessToolCall):
            url = call.arguments.url
            source_search_step_id = known_urls.get(url)
            if source_search_step_id is None:
                error_code = "access_url_not_discovered"
                result_payload = {
                    "error_code": error_code,
                    "url": url,
                    "request_id": request_id,
                }
                result_digest = canonical_digest(result_payload)
                message_payload = {
                    "request_index": request_index,
                    "tool": tool_name,
                    "status": "failed",
                    "error_code": error_code,
                }
                return _ToolExecution(
                    event=TrainableEpisodeEvent(
                        event_index=0,
                        step_id=step_id,
                        kind=tool_name,
                        agent_role="sub",
                        agent_instance_id=instance_id,
                        causal_step_ids=(causal_step_id,),
                        status="failed",
                        phase="sub",
                        payload_digest=result_digest,
                        environment=await self._environment_snapshot(ledger),
                    ),
                    outcome=ToolOutcomeRecord(
                        step_id=step_id,
                        agent_instance_id=instance_id,
                        spawn_round=spawn_round,
                        sub_turn=sub_turn,
                        request_index=request_index,
                        tool_name=tool_name,
                        status="failed",
                        request_json=request_json,
                        request_digest=request_digest,
                        result_json=self._canonical_json(result_payload),
                        result_digest=result_digest,
                        url=url,
                        error_code=error_code,
                    ),
                    message_payload=message_payload,
                )

        reservation_id = f"{request_id}:budget"
        async with semaphore:
            await ledger.reserve(reservation_id)
            try:
                await ledger.commit(reservation_id)
                if isinstance(call, SearchToolCall):
                    query = call.arguments.query
                    search_result = await self._tools.search(
                        SearchRequest(
                            request_id=request_id,
                            query=query,
                            max_results=call.arguments.topk,
                        )
                    )
                    self._validate_search_response(search_result, request_id)
                    provider_revision = search_result.provider_revision
                    provider_response_digest = search_result.raw_response_digest
                    result_payload = search_result.model_dump(mode="json")
                    result_digest = canonical_digest(result_payload)
                    discovered_urls = tuple(item.url for item in search_result.results)
                    message_payload = {
                        "request_index": request_index,
                        "tool": tool_name,
                        "status": "success",
                        "results": self._search_message_results(search_result),
                    }
                else:
                    access_result = await self._tools.access(
                        AccessRequest(
                            request_id=request_id,
                            url=call.arguments.url,
                            info_to_extract=call.arguments.info_to_extract,
                            max_characters=self._max_access_characters,
                        )
                    )
                    self._validate_access_response(access_result, request_id, call.arguments.url)
                    provider_revision = access_result.provider_revision
                    provider_response_digest = access_result.raw_response_digest
                    result_payload = access_result.model_dump(mode="json")
                    result_digest = canonical_digest(result_payload)
                    message_payload = {
                        "request_index": request_index,
                        "tool": tool_name,
                        "status": "success",
                        "url": access_result.url,
                        "content": access_result.content,
                        "truncated": access_result.truncated,
                    }
            except Exception as exc:
                error_code = type(exc).__name__
                result_payload = {"error_code": error_code, "request_id": request_id}
                result_digest = canonical_digest(result_payload)
                message_payload = {
                    "request_index": request_index,
                    "tool": tool_name,
                    "status": "failed",
                    "error_code": error_code,
                }
            finally:
                await ledger.release(reservation_id)

        status: Literal["success", "failed"] = "failed" if error_code else "success"
        environment = await self._environment_snapshot(
            ledger,
            provider_revision=provider_revision,
            response_digest=provider_response_digest or result_digest,
        )
        return _ToolExecution(
            event=TrainableEpisodeEvent(
                event_index=0,
                step_id=step_id,
                kind=tool_name,
                agent_role="sub",
                agent_instance_id=instance_id,
                causal_step_ids=(causal_step_id,),
                status=status,
                phase="sub",
                payload_digest=result_digest,
                environment=environment,
            ),
            outcome=ToolOutcomeRecord(
                step_id=step_id,
                agent_instance_id=instance_id,
                spawn_round=spawn_round,
                sub_turn=sub_turn,
                request_index=request_index,
                tool_name=tool_name,
                status=status,
                request_json=request_json,
                request_digest=request_digest,
                result_json=self._canonical_json(result_payload),
                result_digest=result_digest,
                provider_response_digest=provider_response_digest,
                query=query,
                url=url,
                source_search_step_id=source_search_step_id,
                provider_revision=provider_revision,
                error_code=error_code,
            ),
            message_payload=message_payload,
            discovered_urls=discovered_urls,
        )

    async def _failed_sub(
        self,
        rollout_id: RolloutId,
        instance_id: AgentInstanceId,
        subtask: str,
        spawn_round: int,
        spawn_step_id: StepId,
        error_code: str,
        model_steps: list[TrajectoryStep],
        events: list[TrainableEpisodeEvent],
        tool_outcomes: list[ToolOutcomeRecord],
        ledger: ConcurrencyBudgetLedger,
    ) -> _SubExecution:
        cause = events[-1].step_id if events else spawn_step_id
        failure_step_id = StepId(f"{rollout_id}:{instance_id}:failure")
        events.append(
            TrainableEpisodeEvent(
                event_index=0,
                step_id=failure_step_id,
                kind="sub_failure",
                agent_role="sub",
                agent_instance_id=instance_id,
                causal_step_ids=(cause,),
                status="failed",
                phase="sub",
                payload_digest=canonical_digest({"error_code": error_code}),
                environment=await self._environment_snapshot(ledger),
            )
        )
        outcome = TrainableSubOutcome(
            agent_instance_id=instance_id,
            subtask=subtask,
            spawn_round=spawn_round,
            status="failed",
            content="subtask failed",
            search_step_id=next(
                (
                    item.step_id
                    for item in tool_outcomes
                    if item.tool_name == "search" and item.status == "success"
                ),
                None,
            ),
            model_step_id=model_steps[-1].step_id if model_steps else None,
            tool_step_ids=tuple(item.step_id for item in tool_outcomes),
            model_step_ids=tuple(step.step_id for step in model_steps),
            error_code=error_code,
        )
        return _SubExecution(
            outcome=outcome,
            evidence=None,
            model_steps=tuple(model_steps),
            events=tuple(events),
            tool_outcomes=tuple(tool_outcomes),
            final_step_id=failure_step_id,
        )

    @staticmethod
    def _merge_sub_execution(
        execution: _SubExecution,
        events: list[TrainableEpisodeEvent],
        model_steps: list[TrajectoryStep],
    ) -> None:
        steps = {step.step_id: step for step in execution.model_steps}
        for event in execution.events:
            event_index = len(events)
            events.append(event.model_copy(update={"event_index": event_index}))
            step = steps.get(event.step_id)
            if step is not None:
                model_steps.append(step.model_copy(update={"event_index": event_index}))

    def _generation_request(
        self,
        task: ResearchTask,
        episode_id: EpisodeId,
        rollout_id: RolloutId,
        request_id: str,
        role: AgentRole,
        instance_id: AgentInstanceId,
        prompt: PromptEncoding,
    ) -> GenerationRequest:
        return GenerationRequest(
            request_id=request_id,
            task_id=task.task_id,
            episode_id=episode_id,
            rollout_id=rollout_id,
            agent_role=role,
            agent_instance_id=instance_id,
            prompt_ids=prompt.prompt_ids,
            tokenizer_revision=prompt.tokenizer_revision,
            prompt_template_revision=prompt.prompt_template_revision,
            sampling_params=self._sampling_params,
        )

    def _trajectory_step(
        self,
        task: ResearchTask,
        episode_id: EpisodeId,
        rollout_id: RolloutId,
        step_id: StepId,
        event_index: int,
        causal_step_ids: tuple[StepId, ...],
        role: AgentRole,
        instance_id: AgentInstanceId,
        prompt: PromptEncoding,
        result: GenerationResult,
        revisions: Mapping[PolicyId, RolloutRevision],
    ) -> TrajectoryStep:
        partners = tuple(
            revision
            for policy_id, revision in sorted(revisions.items(), key=lambda item: str(item[0]))
            if policy_id != result.policy_id
        )
        return TrajectoryStep(
            task_id=task.task_id,
            episode_id=episode_id,
            rollout_id=rollout_id,
            step_id=step_id,
            event_index=event_index,
            causal_step_ids=causal_step_ids,
            agent_role=role,
            agent_instance_id=instance_id,
            policy_id=result.policy_id,
            rollout_revision=result.rollout_revision,
            partner_rollout_revisions=partners,
            prompt_ids=prompt.prompt_ids,
            response_ids=result.response_ids,
            response_log_probs=result.response_log_probs,
            tokenizer_revision=prompt.tokenizer_revision,
            prompt_template_revision=prompt.prompt_template_revision,
            sampling_params=self._sampling_params,
            stop_reason=result.stop_reason,
        )

    @staticmethod
    def _model_event(
        step: TrajectoryStep,
        status: Literal["valid", "invalid", "success", "failed"],
        phase: Literal["initial", "main", "sub", "final"],
        environment: TrainableEnvironmentSnapshot,
    ) -> TrainableEpisodeEvent:
        return TrainableEpisodeEvent(
            event_index=step.event_index,
            step_id=step.step_id,
            kind="model",
            agent_role=step.agent_role,
            agent_instance_id=step.agent_instance_id,
            causal_step_ids=step.causal_step_ids,
            status=status,
            phase=phase,
            payload_digest=canonical_digest({"response_ids": step.response_ids}),
            environment=environment,
        )

    async def _environment_snapshot(
        self,
        ledger: ConcurrencyBudgetLedger,
        *,
        provider_revision: str | None = None,
        response_digest: str | None = None,
    ) -> TrainableEnvironmentSnapshot:
        return TrainableEnvironmentSnapshot(
            budget=await ledger.snapshot(),
            search_provider_revision=provider_revision,
            search_response_digest=response_digest,
            prompt_revision=self.prompt_revision,
            tool_schema_revision=WIDESEEK_TOOL_SCHEMA_REVISION,
            parser_revision=WIDESEEK_PARSER_REVISION,
        )

    def _search_message_results(self, result: SearchResponse) -> list[dict[str, object]]:
        visible_results = result.results[: self._max_search_message_results]
        if not visible_results:
            return []
        per_result_budget = max(
            1,
            self._max_search_content_characters // len(visible_results),
        )
        return [
            {
                "title": item.title,
                "url": item.url,
                "content": item.content[:per_result_budget],
                "content_truncated": len(item.content) > per_result_budget,
                "score": item.score,
            }
            for item in visible_results
        ]

    def _validate_revisions(
        self,
        policy_revisions: tuple[tuple[PolicyId, RolloutRevision], ...],
    ) -> dict[PolicyId, RolloutRevision]:
        revisions = dict(policy_revisions)
        if len(revisions) != len(policy_revisions):
            raise ConfigurationError("episode policy revision map contains duplicates")
        for role in ("main", "sub"):
            service = self._services[role]
            revision = revisions.get(service.policy_id)
            if revision is None or revision != self._registry.revision(service.policy_id):
                raise ConfigurationError(f"{role} episode revision is missing or stale")
        return revisions

    @staticmethod
    def _validate_generation_result(
        result: GenerationResult,
        request_id: str,
        policy_id: PolicyId,
        revision: RolloutRevision,
    ) -> None:
        if result.request_id != request_id:
            raise RuntimeError("policy returned a result for another request")
        if result.policy_id != policy_id or result.rollout_revision != revision:
            raise RuntimeError("policy returned a result from another rollout revision")

    @staticmethod
    def _validate_search_response(result: SearchResponse, request_id: str) -> None:
        if result.request_id != request_id or not result.provider_revision:
            raise RuntimeError("search returned mismatched or unversioned response")

    @staticmethod
    def _canonical_json(value: object) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _validate_access_response(
        result: AccessResponse,
        request_id: str,
        url: str,
    ) -> None:
        if result.request_id != request_id or result.url != url or not result.provider_revision:
            raise RuntimeError("access returned mismatched or unversioned response")

    @staticmethod
    def _main_phase(
        round_index: int,
        allow_spawn: bool,
    ) -> Literal["initial", "main", "final"]:
        if round_index == 0:
            return "initial"
        return "main" if allow_spawn else "final"

    @property
    def _main_prompt(self) -> str:
        return (
            f"{_MAIN_SYSTEM_PROMPT}\n"
            f"Episode limits: {self._max_main_rounds} Main turns, "
            f"{self._max_spawn_per_round} workers per spawn round, "
            f"{self._max_spawn_per_episode} workers total."
        )

    @staticmethod
    def _trace(
        task: ResearchTask,
        episode_id: EpisodeId,
        rollout_id: RolloutId,
        revisions: tuple[tuple[PolicyId, RolloutRevision], ...],
        *,
        status: Literal["success", "failed"],
        attempts: list[TrainableMainAttempt],
        spawn_rounds: list[SpawnRoundRecord],
        sub_outcomes: list[TrainableSubOutcome],
        evidence: list[EvidenceRecord],
        tool_outcomes: list[ToolOutcomeRecord],
        model_steps: list[TrajectoryStep],
        events: list[TrainableEpisodeEvent],
        answer: str | None = None,
        failure_code: str | None = None,
    ) -> TrainableEpisodeTrace:
        return TrainableEpisodeTrace(
            task_id=task.task_id,
            episode_id=episode_id,
            rollout_id=rollout_id,
            status=status,
            answer=answer,
            failure_code=failure_code,
            spawn_count=len(sub_outcomes),
            spawn_rounds=tuple(spawn_rounds),
            main_attempts=tuple(attempts),
            sub_outcomes=tuple(sub_outcomes),
            tool_outcomes=tuple(tool_outcomes),
            evidence=tuple(evidence),
            model_steps=tuple(model_steps),
            events=tuple(events),
            policy_revisions=revisions,
        )
