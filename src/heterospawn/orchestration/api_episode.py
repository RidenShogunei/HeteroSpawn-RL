"""One-round API-first Main/Sub episode used to validate benchmark semantics."""

from __future__ import annotations

import asyncio
import json

from heterospawn.domain.ids import AgentInstanceId, EpisodeId, RolloutId
from heterospawn.domain.tasks import ResearchTask
from heterospawn.errors import EpisodeRunError, InvalidActionError
from heterospawn.orchestration.budget import ConcurrencyBudgetLedger
from heterospawn.orchestration.models import (
    AnswerAction,
    EpisodeEvent,
    EpisodeTrace,
    MainAction,
    MainAttempt,
    SpawnAction,
    SubResult,
    parse_main_action,
)
from heterospawn.policies.base import (
    EvaluationGenerationRequest,
    EvaluationPolicyService,
    JsonScalar,
    Message,
)
from heterospawn.search.base import SearchRequest, SearchService

_MAIN_SYSTEM_PROMPT = """You are the Main research policy. Return exactly one JSON object.
To answer without delegation: {"kind":"answer","answer":"..."}
To delegate one or more tasks: {"kind":"spawn","subtasks":["..."]}
An empty subtasks list is illegal. Do not wrap JSON in markdown."""


class ApiEpisodeOrchestrator:
    """Validates shared-policy dynamic spawning without creating RL samples."""

    def __init__(
        self,
        policy: EvaluationPolicyService,
        search: SearchService,
        *,
        max_concurrency: int = 4,
        max_spawn_per_episode: int = 4,
        repair_attempts: int = 1,
        sampling_params: tuple[tuple[str, JsonScalar], ...] = (),
    ) -> None:
        if repair_attempts < 0:
            raise ValueError("repair_attempts cannot be negative")
        if max_spawn_per_episode < 1:
            raise ValueError("max_spawn_per_episode must be positive")
        self._policy = policy
        self._search = search
        self._repair_attempts = repair_attempts
        self._max_spawn_per_episode = max_spawn_per_episode
        self._sampling_params = sampling_params
        self._ledger = ConcurrencyBudgetLedger(max_concurrency)
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def run(self, task: ResearchTask, episode_id: EpisodeId) -> EpisodeTrace:
        events: list[EpisodeEvent] = []
        attempts: list[MainAttempt] = []
        try:
            initial_action, initial_event_index = await self._generate_main_action(
                task,
                episode_id,
                phase="initial",
                messages=(
                    Message(role="system", content=self._main_system_prompt),
                    Message(role="user", content=task.prompt),
                ),
                events=events,
                attempts=attempts,
                causal_event_indices=(),
                require_answer=False,
            )
        except InvalidActionError:
            raise _episode_run_error(attempts, (), events) from None

        if isinstance(initial_action, AnswerAction):
            return self._trace(
                task,
                episode_id,
                initial_action.answer,
                attempts,
                (),
                events,
            )

        sub_results = await asyncio.gather(
            *(
                self._run_sub(task, episode_id, index, subtask)
                for index, subtask in enumerate(initial_action.subtasks)
            )
        )
        stable_snapshot = await self._ledger.snapshot()
        sub_event_indices: list[int] = []
        for result in sub_results:
            event_index = len(events)
            sub_event_indices.append(event_index)
            events.append(
                EpisodeEvent(
                    event_index=event_index,
                    kind="sub_result",
                    agent_instance_id=result.agent_instance_id,
                    causal_event_indices=(initial_event_index,),
                    status=result.status,
                    phase="sub",
                    detail=result.content,
                    environment=stable_snapshot,
                )
            )

        evidence = [result.model_dump(mode="json") for result in sub_results]
        try:
            final_action, _ = await self._generate_main_action(
                task,
                episode_id,
                phase="final",
                messages=(
                    Message(role="system", content=self._main_system_prompt),
                    Message(role="user", content=task.prompt),
                    Message(
                        role="user",
                        content=(
                            "Sub results follow. Return an ANSWER action only.\n"
                            + json.dumps(evidence, ensure_ascii=False, sort_keys=True)
                        ),
                    ),
                ),
                events=events,
                attempts=attempts,
                causal_event_indices=tuple(sub_event_indices),
                require_answer=True,
            )
        except InvalidActionError:
            raise _episode_run_error(attempts, tuple(sub_results), events) from None
        if not isinstance(final_action, AnswerAction):
            raise AssertionError("final action validation must require ANSWER")
        return self._trace(
            task,
            episode_id,
            final_action.answer,
            attempts,
            tuple(sub_results),
            events,
        )

    async def _generate_main_action(
        self,
        task: ResearchTask,
        episode_id: EpisodeId,
        *,
        phase: str,
        messages: tuple[Message, ...],
        events: list[EpisodeEvent],
        attempts: list[MainAttempt],
        causal_event_indices: tuple[int, ...],
        require_answer: bool,
    ) -> tuple[MainAction, int]:
        current_messages = messages
        attempt_causes = causal_event_indices
        for attempt_index in range(self._repair_attempts + 1):
            result = await self._policy.generate(
                EvaluationGenerationRequest(
                    request_id=f"{episode_id}:main:{phase}:{attempt_index}",
                    task_id=task.task_id,
                    episode_id=episode_id,
                    rollout_id=RolloutId(f"{episode_id}:evaluation"),
                    agent_role="main",
                    agent_instance_id=AgentInstanceId("main-0"),
                    messages=current_messages,
                    sampling_params=self._sampling_params,
                )
            )
            error_code: str | None = None
            try:
                action = parse_main_action(result.content)
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

            attempts.append(
                MainAttempt(
                    phase="initial" if phase == "initial" else "final",
                    attempt_index=attempt_index,
                    content=result.content,
                    raw_response_digest=result.raw_response_digest,
                    usage=result.usage,
                    valid=valid,
                    error_code=error_code,
                )
            )
            event_index = len(events)
            events.append(
                EpisodeEvent(
                    event_index=event_index,
                    kind="main_output",
                    agent_instance_id=AgentInstanceId("main-0"),
                    causal_event_indices=attempt_causes,
                    status="valid" if valid else "invalid",
                    phase="initial" if phase == "initial" else "final",
                    detail=result.content,
                    environment=await self._ledger.snapshot(),
                )
            )
            if action is not None:
                return action, event_index
            attempt_causes = (*causal_event_indices, event_index)
            current_messages = (
                *current_messages,
                Message(role="assistant", content=result.content),
                Message(
                    role="user",
                    content="Repair the invalid action. Return one schema-valid JSON object only.",
                ),
            )

        raise InvalidActionError("Main exhausted action repair attempts") from None

    async def _run_sub(
        self,
        task: ResearchTask,
        episode_id: EpisodeId,
        index: int,
        subtask: str,
    ) -> SubResult:
        agent_instance_id = AgentInstanceId(f"sub-{index}")
        reservation_id = f"{episode_id}:{agent_instance_id}"
        async with self._semaphore:
            await self._ledger.reserve(reservation_id)
            try:
                await self._ledger.commit(reservation_id)
                search_response = await self._search.search(
                    SearchRequest(
                        request_id=f"{episode_id}:search:{index}",
                        query=subtask,
                    )
                )
                sources = [item.model_dump(mode="json") for item in search_response.results]
                policy_response = await self._policy.generate(
                    EvaluationGenerationRequest(
                        request_id=f"{episode_id}:sub:{index}",
                        task_id=task.task_id,
                        episode_id=episode_id,
                        rollout_id=RolloutId(f"{episode_id}:evaluation"),
                        agent_role="sub",
                        agent_instance_id=agent_instance_id,
                        messages=(
                            Message(
                                role="system",
                                content=(
                                    "Synthesize concise evidence for Main from supplied sources."
                                ),
                            ),
                            Message(role="user", content=subtask),
                            Message(
                                role="user",
                                content=json.dumps(sources, ensure_ascii=False, sort_keys=True),
                            ),
                        ),
                        sampling_params=self._sampling_params,
                    )
                )
                return SubResult(
                    agent_instance_id=agent_instance_id,
                    subtask=subtask,
                    status="success",
                    content=policy_response.content,
                    search_provider_revision=search_response.provider_revision,
                    search_provider_request_id=search_response.provider_request_id,
                    policy_provider_request_id=policy_response.provider_request_id,
                    policy_usage=policy_response.usage,
                )
            except Exception as exc:
                return SubResult(
                    agent_instance_id=agent_instance_id,
                    subtask=subtask,
                    status="failed",
                    content="subtask failed",
                    error_code=type(exc).__name__,
                )
            finally:
                await self._ledger.release(reservation_id)

    @property
    def _main_system_prompt(self) -> str:
        return (
            f"{_MAIN_SYSTEM_PROMPT}\n"
            f"At most {self._max_spawn_per_episode} subtasks may be spawned in this episode."
        )

    def _trace(
        self,
        task: ResearchTask,
        episode_id: EpisodeId,
        answer: str,
        attempts: list[MainAttempt],
        sub_results: tuple[SubResult, ...],
        events: list[EpisodeEvent],
    ) -> EpisodeTrace:
        return EpisodeTrace(
            task_id=task.task_id,
            episode_id=episode_id,
            answer=answer,
            spawn_count=len(sub_results),
            main_attempts=tuple(attempts),
            sub_results=sub_results,
            events=tuple(events),
            policy_revisions=((self._policy.policy_id, self._policy.revision),),
            trainable=False,
        )


def _episode_run_error(
    attempts: list[MainAttempt],
    sub_results: tuple[SubResult, ...],
    events: list[EpisodeEvent],
) -> EpisodeRunError:
    usages = [attempt.usage for attempt in attempts]
    usages.extend(result.policy_usage for result in sub_results if result.policy_usage is not None)
    return EpisodeRunError(
        error_code="InvalidActionError",
        spawn_count=len(sub_results),
        successful_subs=sum(result.status == "success" for result in sub_results),
        failed_subs=sum(result.status == "failed" for result in sub_results),
        main_attempts=len(attempts),
        invalid_main_attempts=sum(not attempt.valid for attempt in attempts),
        event_count=len(events),
        prompt_tokens=sum(usage.prompt_tokens for usage in usages),
        completion_tokens=sum(usage.completion_tokens for usage in usages),
        total_tokens=sum(usage.total_tokens for usage in usages),
    )
