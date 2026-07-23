"""Exact-token Main/Sub orchestration for trainable system rollouts."""

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
from heterospawn.orchestration.models import (
    AnswerAction,
    MainAction,
    SpawnAction,
    parse_main_action,
)
from heterospawn.orchestration.trainable_models import (
    EvidenceRecord,
    TrainableEnvironmentSnapshot,
    TrainableEpisodeEvent,
    TrainableEpisodeTrace,
    TrainableMainAttempt,
    TrainableSubOutcome,
)
from heterospawn.policies.base import Message
from heterospawn.policies.trainable import TrainablePolicyCodec
from heterospawn.search.base import SearchRequest, SearchResponse, SearchService
from heterospawn.training.base import PolicyService
from heterospawn.training.registry import PolicyRegistry

_MAIN_SYSTEM_PROMPT = """You are the Main research policy. Return exactly one JSON object.
To answer without delegation: {"kind":"answer","answer":"..."}
To delegate one or more tasks: {"kind":"spawn","subtasks":["..."]}
An empty subtasks list is illegal. Do not wrap JSON in markdown."""

_SUB_SYSTEM_PROMPT = "Synthesize concise evidence for Main from the supplied sources."


@dataclass(frozen=True)
class _SubExecution:
    agent_instance_id: AgentInstanceId
    subtask: str
    environment: TrainableEnvironmentSnapshot
    search_response: SearchResponse | None
    prompt: PromptEncoding | None
    generation: GenerationResult | None
    content: str
    error_code: str | None


class TrainableEpisodeOrchestrator:
    """Runs one-level dynamic spawning while retaining every exact MODEL trajectory."""

    def __init__(
        self,
        registry: PolicyRegistry,
        policy_services: Mapping[AgentRole, PolicyService],
        codecs: Mapping[AgentRole, TrainablePolicyCodec],
        search: SearchService,
        *,
        max_concurrency: int = 4,
        max_spawn_per_episode: int = 4,
        repair_attempts: int = 1,
        sampling_params: tuple[tuple[str, JsonScalar], ...] = (),
        main_initial_sampling_params: tuple[tuple[str, JsonScalar], ...] | None = None,
        sub_sampling_params: tuple[tuple[str, JsonScalar], ...] | None = None,
        main_final_sampling_params: tuple[tuple[str, JsonScalar], ...] | None = None,
    ) -> None:
        if repair_attempts < 0:
            raise ValueError("repair_attempts cannot be negative")
        if max_spawn_per_episode < 1:
            raise ValueError("max_spawn_per_episode must be positive")
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
        self._search = search
        self._repair_attempts = repair_attempts
        self._max_spawn_per_episode = max_spawn_per_episode
        self._sampling_params = sampling_params
        self._main_initial_sampling_params = (
            sampling_params
            if main_initial_sampling_params is None
            else main_initial_sampling_params
        )
        self._sub_sampling_params = (
            sampling_params if sub_sampling_params is None else sub_sampling_params
        )
        self._main_final_sampling_params = (
            sampling_params if main_final_sampling_params is None else main_final_sampling_params
        )
        self._ledger = ConcurrencyBudgetLedger(max_concurrency)
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def run(
        self,
        task: ResearchTask,
        episode_id: EpisodeId,
        rollout_id: RolloutId,
        policy_revisions: tuple[tuple[PolicyId, RolloutRevision], ...],
    ) -> TrainableEpisodeTrace:
        revisions = self._validate_revisions(policy_revisions)
        events: list[TrainableEpisodeEvent] = []
        model_steps: list[TrajectoryStep] = []
        attempts: list[TrainableMainAttempt] = []

        initial_action, initial_step_id = await self._generate_main_action(
            task=task,
            episode_id=episode_id,
            rollout_id=rollout_id,
            revisions=revisions,
            phase="initial",
            messages=(
                Message(role="system", content=self._main_system_prompt),
                Message(role="user", content=task.prompt),
            ),
            events=events,
            model_steps=model_steps,
            attempts=attempts,
            causal_step_ids=(),
            require_answer=False,
        )
        if initial_action is None:
            return self._trace(
                task=task,
                episode_id=episode_id,
                rollout_id=rollout_id,
                revisions=policy_revisions,
                status="failed",
                failure_code="invalid_initial_main_action",
                attempts=attempts,
                model_steps=model_steps,
                events=events,
            )
        if isinstance(initial_action, AnswerAction):
            return self._trace(
                task=task,
                episode_id=episode_id,
                rollout_id=rollout_id,
                revisions=policy_revisions,
                status="success",
                answer=initial_action.answer,
                attempts=attempts,
                model_steps=model_steps,
                events=events,
            )

        executions = await asyncio.gather(
            *(
                self._run_sub(
                    task=task,
                    episode_id=episode_id,
                    rollout_id=rollout_id,
                    revisions=revisions,
                    index=index,
                    subtask=subtask,
                )
                for index, subtask in enumerate(initial_action.subtasks)
            )
        )
        sub_outcomes: list[TrainableSubOutcome] = []
        evidence: list[EvidenceRecord] = []
        final_causes: list[StepId] = []
        for execution in executions:
            search_step_id = StepId(f"{rollout_id}:{execution.agent_instance_id}:search")
            search_status: Literal["success", "failed"] = (
                "success" if execution.search_response is not None else "failed"
            )
            search_digest = (
                execution.search_response.raw_response_digest
                if execution.search_response is not None
                else canonical_digest({"error_code": execution.error_code})
            )
            events.append(
                TrainableEpisodeEvent(
                    event_index=len(events),
                    step_id=search_step_id,
                    kind="search",
                    agent_role="sub",
                    agent_instance_id=execution.agent_instance_id,
                    causal_step_ids=(initial_step_id,),
                    status=search_status,
                    phase="sub",
                    payload_digest=search_digest,
                    environment=execution.environment,
                )
            )
            if execution.generation is None or execution.prompt is None:
                failure_step_id = StepId(f"{rollout_id}:{execution.agent_instance_id}:failure")
                events.append(
                    TrainableEpisodeEvent(
                        event_index=len(events),
                        step_id=failure_step_id,
                        kind="sub_failure",
                        agent_role="sub",
                        agent_instance_id=execution.agent_instance_id,
                        causal_step_ids=(search_step_id,),
                        status="failed",
                        phase="sub",
                        payload_digest=canonical_digest(
                            {"error_code": execution.error_code or "unknown_sub_failure"}
                        ),
                        environment=execution.environment,
                    )
                )
                final_causes.append(failure_step_id)
                sub_outcomes.append(
                    TrainableSubOutcome(
                        agent_instance_id=execution.agent_instance_id,
                        subtask=execution.subtask,
                        status="failed",
                        content="subtask failed",
                        search_step_id=search_step_id,
                        error_code=execution.error_code or "unknown_sub_failure",
                    )
                )
                continue

            model_step_id = StepId(f"{rollout_id}:{execution.agent_instance_id}:model")
            step = self._trajectory_step(
                task=task,
                episode_id=episode_id,
                rollout_id=rollout_id,
                step_id=model_step_id,
                event_index=len(events),
                causal_step_ids=(search_step_id,),
                role="sub",
                instance_id=execution.agent_instance_id,
                prompt=execution.prompt,
                result=execution.generation,
                revisions=revisions,
                sampling_params=self._sub_sampling_params,
            )
            model_steps.append(step)
            events.append(
                self._model_event(
                    step=step,
                    status="failed" if execution.error_code is not None else "success",
                    phase="sub",
                    environment=execution.environment,
                )
            )
            if execution.error_code is not None:
                failure_step_id = StepId(f"{rollout_id}:{execution.agent_instance_id}:failure")
                events.append(
                    TrainableEpisodeEvent(
                        event_index=len(events),
                        step_id=failure_step_id,
                        kind="sub_failure",
                        agent_role="sub",
                        agent_instance_id=execution.agent_instance_id,
                        causal_step_ids=(model_step_id,),
                        status="failed",
                        phase="sub",
                        payload_digest=canonical_digest({"error_code": execution.error_code}),
                        environment=execution.environment,
                    )
                )
                final_causes.append(failure_step_id)
                sub_outcomes.append(
                    TrainableSubOutcome(
                        agent_instance_id=execution.agent_instance_id,
                        subtask=execution.subtask,
                        status="failed",
                        content="subtask failed",
                        search_step_id=search_step_id,
                        model_step_id=model_step_id,
                        error_code=execution.error_code,
                    )
                )
                continue
            final_causes.append(model_step_id)
            sub_outcomes.append(
                TrainableSubOutcome(
                    agent_instance_id=execution.agent_instance_id,
                    subtask=execution.subtask,
                    status="success",
                    content=execution.content,
                    search_step_id=search_step_id,
                    model_step_id=model_step_id,
                )
            )
            evidence.append(
                EvidenceRecord(
                    agent_instance_id=execution.agent_instance_id,
                    subtask=execution.subtask,
                    content=execution.content,
                    producer_tool_step_id=search_step_id,
                    producer_model_step_id=model_step_id,
                )
            )

        evidence_payload = [outcome.model_dump(mode="json") for outcome in sub_outcomes]
        final_action, _ = await self._generate_main_action(
            task=task,
            episode_id=episode_id,
            rollout_id=rollout_id,
            revisions=revisions,
            phase="final",
            messages=(
                Message(role="system", content=self._main_system_prompt),
                Message(role="user", content=task.prompt),
                Message(
                    role="user",
                    content=(
                        "Sub results follow. Return an ANSWER action only.\n"
                        + json.dumps(evidence_payload, ensure_ascii=False, sort_keys=True)
                    ),
                ),
            ),
            events=events,
            model_steps=model_steps,
            attempts=attempts,
            causal_step_ids=tuple(final_causes),
            require_answer=True,
        )
        if final_action is None:
            return self._trace(
                task=task,
                episode_id=episode_id,
                rollout_id=rollout_id,
                revisions=policy_revisions,
                status="failed",
                failure_code="invalid_final_main_action",
                attempts=attempts,
                sub_outcomes=sub_outcomes,
                evidence=evidence,
                model_steps=model_steps,
                events=events,
            )
        if not isinstance(final_action, AnswerAction):
            raise AssertionError("final action validation must require ANSWER")
        return self._trace(
            task=task,
            episode_id=episode_id,
            rollout_id=rollout_id,
            revisions=policy_revisions,
            status="success",
            answer=final_action.answer,
            attempts=attempts,
            sub_outcomes=sub_outcomes,
            evidence=evidence,
            model_steps=model_steps,
            events=events,
        )

    async def _generate_main_action(
        self,
        *,
        task: ResearchTask,
        episode_id: EpisodeId,
        rollout_id: RolloutId,
        revisions: Mapping[PolicyId, RolloutRevision],
        phase: Literal["initial", "final"],
        messages: tuple[Message, ...],
        events: list[TrainableEpisodeEvent],
        model_steps: list[TrajectoryStep],
        attempts: list[TrainableMainAttempt],
        causal_step_ids: tuple[StepId, ...],
        require_answer: bool,
    ) -> tuple[MainAction | None, StepId]:
        service = self._services["main"]
        codec = self._codecs["main"]
        expected_revision = revisions[service.policy_id]
        sampling_params = (
            self._main_initial_sampling_params
            if phase == "initial"
            else self._main_final_sampling_params
        )
        current_messages = messages
        attempt_causes = causal_step_ids
        last_step_id: StepId | None = None
        for attempt_index in range(self._repair_attempts + 1):
            prompt = codec.encode(current_messages)
            request_id = f"{rollout_id}:main:{phase}:{attempt_index}"
            result = await service.generate(
                GenerationRequest(
                    request_id=request_id,
                    task_id=task.task_id,
                    episode_id=episode_id,
                    rollout_id=rollout_id,
                    agent_role="main",
                    agent_instance_id=AgentInstanceId("main-0"),
                    prompt_ids=prompt.prompt_ids,
                    tokenizer_revision=prompt.tokenizer_revision,
                    prompt_template_revision=prompt.prompt_template_revision,
                    sampling_params=sampling_params,
                ),
                expected_revision,
            )
            self._validate_generation_result(
                result,
                request_id,
                service.policy_id,
                expected_revision,
            )
            error_code: str | None = None
            try:
                content = codec.decode(result.response_ids)
            except Exception:
                content = ""
                action = None
                valid = False
                error_code = "invalid_main_action"
            else:
                try:
                    action = parse_main_action(content)
                    if (
                        isinstance(action, SpawnAction)
                        and len(action.subtasks) > self._max_spawn_per_episode
                    ):
                        raise InvalidActionError("Main spawn exceeds the configured episode limit")
                    if require_answer and isinstance(action, SpawnAction):
                        raise InvalidActionError("final Main output must be ANSWER")
                    valid = True
                except InvalidActionError:
                    action = None
                    valid = False
                    error_code = "invalid_main_action"

            step_id = StepId(f"{rollout_id}:main:{phase}:{attempt_index}")
            last_step_id = step_id
            step = self._trajectory_step(
                task=task,
                episode_id=episode_id,
                rollout_id=rollout_id,
                step_id=step_id,
                event_index=len(events),
                causal_step_ids=attempt_causes,
                role="main",
                instance_id=AgentInstanceId("main-0"),
                prompt=prompt,
                result=result,
                revisions=revisions,
                sampling_params=sampling_params,
            )
            model_steps.append(step)
            events.append(
                self._model_event(
                    step=step,
                    status="valid" if valid else "invalid",
                    phase=phase,
                    environment=await self._environment_snapshot(),
                )
            )
            attempts.append(
                TrainableMainAttempt(
                    phase=phase,
                    attempt_index=attempt_index,
                    step_id=step_id,
                    content=content,
                    valid=valid,
                    action_kind=action.kind if action is not None else None,
                    error_code=error_code,
                )
            )
            if action is not None:
                return action, step_id
            attempt_causes = (*causal_step_ids, step_id)
            current_messages = (
                *current_messages,
                Message(role="assistant", content=content),
                Message(
                    role="user",
                    content="Repair the invalid action. Return one schema-valid JSON object only.",
                ),
            )

        if last_step_id is None:
            raise AssertionError("at least one Main attempt must execute")
        return None, last_step_id

    async def _run_sub(
        self,
        *,
        task: ResearchTask,
        episode_id: EpisodeId,
        rollout_id: RolloutId,
        revisions: Mapping[PolicyId, RolloutRevision],
        index: int,
        subtask: str,
    ) -> _SubExecution:
        instance_id = AgentInstanceId(f"sub-{index}")
        reservation_id = f"{rollout_id}:{instance_id}"
        search_response: SearchResponse | None = None
        prompt: PromptEncoding | None = None
        generation: GenerationResult | None = None
        content = ""
        error_code: str | None = None
        environment: TrainableEnvironmentSnapshot | None = None
        async with self._semaphore:
            await self._ledger.reserve(reservation_id)
            try:
                await self._ledger.commit(reservation_id)
                search_response = await self._search.search(
                    SearchRequest(
                        request_id=f"{rollout_id}:search:{index}",
                        query=subtask,
                    )
                )
                environment = await self._environment_snapshot(search_response)
                sources = [item.model_dump(mode="json") for item in search_response.results]
                codec = self._codecs["sub"]
                prompt = codec.encode(
                    (
                        Message(role="system", content=_SUB_SYSTEM_PROMPT),
                        Message(role="user", content=subtask),
                        Message(
                            role="user",
                            content=json.dumps(sources, ensure_ascii=False, sort_keys=True),
                        ),
                    )
                )
                service = self._services["sub"]
                expected_revision = revisions[service.policy_id]
                request_id = f"{rollout_id}:sub:{index}"
                candidate = await service.generate(
                    GenerationRequest(
                        request_id=request_id,
                        task_id=task.task_id,
                        episode_id=episode_id,
                        rollout_id=rollout_id,
                        agent_role="sub",
                        agent_instance_id=instance_id,
                        prompt_ids=prompt.prompt_ids,
                        tokenizer_revision=prompt.tokenizer_revision,
                        prompt_template_revision=prompt.prompt_template_revision,
                        sampling_params=self._sub_sampling_params,
                    ),
                    expected_revision,
                )
                self._validate_generation_result(
                    candidate,
                    request_id,
                    service.policy_id,
                    expected_revision,
                )
                generation = candidate
                content = codec.decode(generation.response_ids)
            except Exception as exc:
                error_code = type(exc).__name__
                if environment is None:
                    environment = await self._environment_snapshot(search_response)
            finally:
                await self._ledger.release(reservation_id)
        if environment is None:
            environment = await self._environment_snapshot(search_response)
        return _SubExecution(
            agent_instance_id=instance_id,
            subtask=subtask,
            environment=environment,
            search_response=search_response,
            prompt=prompt,
            generation=generation,
            content=content,
            error_code=error_code,
        )

    def _trajectory_step(
        self,
        *,
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
        sampling_params: tuple[tuple[str, JsonScalar], ...],
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
            sampling_params=sampling_params,
            stop_reason=result.stop_reason,
        )

    @staticmethod
    def _model_event(
        *,
        step: TrajectoryStep,
        status: Literal["valid", "invalid", "success", "failed"],
        phase: Literal["initial", "sub", "final"],
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
        search_response: SearchResponse | None = None,
    ) -> TrainableEnvironmentSnapshot:
        return TrainableEnvironmentSnapshot(
            budget=await self._ledger.snapshot(),
            search_provider_revision=(
                search_response.provider_revision if search_response is not None else None
            ),
            search_response_digest=(
                search_response.raw_response_digest if search_response is not None else None
            ),
        )

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

    @property
    def _main_system_prompt(self) -> str:
        return (
            f"{_MAIN_SYSTEM_PROMPT}\n"
            f"At most {self._max_spawn_per_episode} subtasks may be spawned in this episode."
        )

    @staticmethod
    def _trace(
        *,
        task: ResearchTask,
        episode_id: EpisodeId,
        rollout_id: RolloutId,
        revisions: tuple[tuple[PolicyId, RolloutRevision], ...],
        status: Literal["success", "failed"],
        answer: str | None = None,
        failure_code: str | None = None,
        attempts: list[TrainableMainAttempt],
        sub_outcomes: list[TrainableSubOutcome] | None = None,
        evidence: list[EvidenceRecord] | None = None,
        model_steps: list[TrajectoryStep],
        events: list[TrainableEpisodeEvent],
    ) -> TrainableEpisodeTrace:
        outcomes = tuple(sub_outcomes or ())
        return TrainableEpisodeTrace(
            task_id=task.task_id,
            episode_id=episode_id,
            rollout_id=rollout_id,
            status=status,
            answer=answer,
            failure_code=failure_code,
            spawn_count=len(outcomes),
            main_attempts=tuple(attempts),
            sub_outcomes=outcomes,
            evidence=tuple(evidence or ()),
            model_steps=tuple(model_steps),
            events=tuple(events),
            policy_revisions=revisions,
        )
