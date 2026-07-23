from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from heterospawn.benchmarks.xbench import BenchmarkTask
from heterospawn.domain.ids import EpisodeId, PolicyId, RolloutId, TaskId
from heterospawn.domain.training import (
    GenerationRequest,
    GenerationResult,
    PromptEncoding,
)
from heterospawn.domain.versions import RoleBinding, RolloutRevision
from heterospawn.errors import ConfigurationError, RolloutRevisionMismatch
from heterospawn.orchestration import TrainableEpisodeOrchestrator
from heterospawn.orchestration.trainable_models import TrainableEpisodeTrace
from heterospawn.policies.base import Message
from heterospawn.search.base import SearchRequest, SearchResponse
from heterospawn.search.mock import MockSearchService
from heterospawn.training import (
    FilePhaseTransactionStore,
    MockTrainingBackend,
    PhaseTransactionContext,
    PolicyRegistry,
    RewardComposer,
    RewardConfig,
    TrainableAlternatingCycleRunner,
)

ResponseScript = Callable[[GenerationRequest], str]


class _Utf8Codec:
    def encode(self, messages: tuple[Message, ...]) -> PromptEncoding:
        text = json.dumps(
            [message.model_dump(mode="json") for message in messages],
            ensure_ascii=False,
            sort_keys=True,
        )
        return PromptEncoding(
            prompt_ids=tuple(text.encode()),
            tokenizer_revision="utf8-v1",
            prompt_template_revision="json-messages-v1",
        )

    def decode(self, response_ids: tuple[int, ...]) -> str:
        return bytes(response_ids).decode()


class _ScriptedPolicyService:
    def __init__(
        self,
        backend: MockTrainingBackend,
        policy_id: PolicyId,
        script: ResponseScript,
    ) -> None:
        self._backend = backend
        self._policy_id = policy_id
        self._script = script
        self.results: dict[str, GenerationResult] = {}
        self.expected_revisions: list[tuple[str, RolloutRevision]] = []

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
        current = self._backend.rollout_revision(self._policy_id)
        if current != expected_revision:
            raise RolloutRevisionMismatch("scripted policy rollout revision mismatch")
        content = self._script(request)
        response_ids = tuple(content.encode())
        result = GenerationResult(
            request_id=request.request_id,
            policy_id=self._policy_id,
            rollout_revision=current,
            response_ids=response_ids,
            response_log_probs=tuple(-0.01 * (index + 1) for index in range(len(response_ids))),
            stop_reason="eos",
        )
        self.results[request.request_id] = result
        self.expected_revisions.append((request.request_id, expected_revision))
        return result


class _FailingSearch:
    def __init__(self, failed_query: str) -> None:
        self._failed_query = failed_query
        self._delegate = MockSearchService()

    async def search(self, request: SearchRequest) -> SearchResponse:
        if request.query == self._failed_query:
            raise RuntimeError("synthetic search failure")
        return await self._delegate.search(request)


class _IndexedReward:
    revision = "indexed-contract-reward-v1"

    async def score(self, task: BenchmarkTask, trace: TrainableEpisodeTrace) -> float:
        del task
        return 0.0 if str(trace.episode_id).endswith("episode-0") else 1.0


def _registry(
    backend: MockTrainingBackend,
    main_id: PolicyId,
    sub_id: PolicyId,
    *,
    sub_trainable: bool = True,
) -> PolicyRegistry:
    return PolicyRegistry(
        (
            RoleBinding(role="main", policy_id=main_id, trainable=True),
            RoleBinding(role="sub", policy_id=sub_id, trainable=sub_trainable),
        ),
        tuple(
            (policy_id, backend.rollout_revision(policy_id))
            for policy_id in dict.fromkeys((main_id, sub_id))
        ),
    )


def _orchestrator(
    *,
    registry: PolicyRegistry,
    main: _ScriptedPolicyService,
    sub: _ScriptedPolicyService,
    search: MockSearchService | _FailingSearch | None = None,
    repair_attempts: int = 1,
) -> TrainableEpisodeOrchestrator:
    codec = _Utf8Codec()
    return TrainableEpisodeOrchestrator(
        registry,
        {"main": main, "sub": sub},
        {"main": codec, "sub": codec},
        search or MockSearchService(),
        max_concurrency=2,
        max_spawn_per_episode=4,
        repair_attempts=repair_attempts,
        sampling_params=(("temperature", 0.0),),
    )


@pytest.mark.asyncio
async def test_trainable_episode_retains_invalid_attempts_failures_and_exact_tokens() -> None:
    main_id = PolicyId("main")
    sub_id = PolicyId("sub")
    backend = MockTrainingBackend((main_id, sub_id))
    registry = _registry(backend, main_id, sub_id)

    def main_script(request: GenerationRequest) -> str:
        if ":main:initial:0" in request.request_id:
            return '{"kind":"spawn","subtasks":[]}'
        if ":main:initial:1" in request.request_id:
            return '{"kind":"spawn","subtasks":["q0","q1","q2","q3"]}'
        return '{"kind":"answer","answer":"final"}'

    main = _ScriptedPolicyService(backend, main_id, main_script)
    sub = _ScriptedPolicyService(backend, sub_id, lambda request: f"evidence:{request.request_id}")
    orchestrator = _orchestrator(
        registry=registry,
        main=main,
        sub=sub,
        search=_FailingSearch("q2"),
    )

    trace = await orchestrator.run(
        BenchmarkTask(task_id=TaskId("task-1"), prompt="research"),
        EpisodeId("episode-1"),
        RolloutId("rollout-1"),
        registry.snapshot(),
    )

    assert trace.status == "success"
    assert trace.answer == "final"
    assert trace.spawn_count == 4
    assert trace.invalid_main_attempts == 1
    assert trace.failed_subs == 1
    assert len(trace.evidence) == 3
    assert tuple(event.event_index for event in trace.events) == tuple(range(len(trace.events)))
    assert [event.kind for event in trace.events] == [
        "model",
        "model",
        "search",
        "model",
        "search",
        "model",
        "search",
        "sub_failure",
        "search",
        "model",
        "model",
    ]
    assert trace.events[-1].causal_step_ids == (
        trace.sub_outcomes[0].model_step_id,
        trace.sub_outcomes[1].model_step_id,
        trace.events[7].step_id,
        trace.sub_outcomes[3].model_step_id,
    )
    assert all(
        step.response_ids == main.results[str(step.step_id)].response_ids
        for step in trace.model_steps
        if step.agent_role == "main"
    )
    assert all(
        step.response_ids
        == sub.results[
            str(step.step_id).replace(":sub-", ":sub:").replace(":model", "")
        ].response_ids
        for step in trace.model_steps
        if step.agent_role == "sub"
    )

    reward = await RewardComposer(
        _IndexedReward(),
        RewardConfig(
            invalid_action_penalty=0.2,
            spawn_cost=0.1,
            sub_failure_penalty=0.3,
        ),
    ).score(BenchmarkTask(task_id=TaskId("task-1"), prompt="research"), trace)
    assert reward.outcome_reward == 1.0
    assert reward.invalid_action_component == pytest.approx(-0.2)
    assert reward.spawn_cost_component == pytest.approx(-0.4)
    assert reward.sub_failure_component == pytest.approx(-0.3)
    assert reward.total == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_fresh_alternating_cycle_updates_main_then_sub_with_episode_balancing(
    tmp_path: Path,
) -> None:
    main_id = PolicyId("main")
    sub_id = PolicyId("sub")
    backend = MockTrainingBackend((main_id, sub_id))
    registry = _registry(backend, main_id, sub_id)

    def main_script(request: GenerationRequest) -> str:
        if ":main:initial:" in request.request_id:
            return '{"kind":"spawn","subtasks":["lookup"]}'
        return '{"kind":"answer","answer":"done"}'

    main = _ScriptedPolicyService(backend, main_id, main_script)
    sub = _ScriptedPolicyService(backend, sub_id, lambda request: "evidence")
    reward = RewardComposer(_IndexedReward(), RewardConfig())
    transaction_store = FilePhaseTransactionStore(tmp_path / "transactions")
    runner = TrainableAlternatingCycleRunner(
        registry,
        backend,
        _orchestrator(registry=registry, main=main, sub=sub),
        reward,
        rollouts_per_task=2,
        transaction_store=transaction_store,
    )

    result = await runner.run_cycle(
        cycle_id="cycle-1",
        tasks=(BenchmarkTask(task_id=TaskId("task-1"), prompt="research"),),
        transaction_context=PhaseTransactionContext(
            experiment_id="unit-experiment",
            config_digest="config-digest",
            rng_state="base64-rng-state",
            sampler_state="base64-sampler-state",
            dataset_revision="unspecified",
            environment_snapshot="deterministic-v1",
            reward_revision=reward.revision,
        ),
    )

    assert result.updates.main_update is not None
    assert result.updates.sub_update is not None
    assert result.updates.empty_sub_batch is False
    assert backend.weight_version(main_id).optimizer_step == 1
    assert backend.weight_version(sub_id).optimizer_step == 1
    assert [phase.phase for phase in result.phases] == ["main_update", "sub_update"]
    assert [manifest.phase_completed for manifest in result.phase_commits] == [
        "main_update",
        "sub_update",
    ]
    main_phase, sub_phase = result.phases
    assert main_phase.degenerate_groups == 0
    assert sub_phase.degenerate_groups == 0
    assert len(main_phase.batch.samples) == 4
    assert len(sub_phase.batch.samples) == 2
    assert all(sample.agent_role == "main" for sample in main_phase.batch.samples)
    assert all(sample.agent_role == "sub" for sample in sub_phase.batch.samples)
    assert all(sample.aggregation_weight == 0.5 for sample in main_phase.batch.samples)
    assert all(sample.aggregation_weight == 1.0 for sample in sub_phase.batch.samples)
    assert sorted({sample.advantage for sample in main_phase.batch.samples}) == pytest.approx(
        [-0.99999998, 0.99999998]
    )

    main_rollouts = {trace.rollout_id for group in main_phase.groups for trace in group.traces}
    sub_rollouts = {trace.rollout_id for group in sub_phase.groups for trace in group.traces}
    assert main_rollouts.isdisjoint(sub_rollouts)
    assert main_phase.policy_revisions[0][1].weight_version.optimizer_step == 0
    assert dict(sub_phase.policy_revisions)[main_id].weight_version.optimizer_step == 1
    assert dict(sub_phase.policy_revisions)[sub_id].weight_version.optimizer_step == 0

    generated = {**main.results, **sub.results}
    for phase in result.phases:
        for sample in phase.batch.samples:
            source = generated[
                str(sample.source_step_id)
                .replace(":main:initial:0", ":main:initial:0")
                .replace(":main:final:0", ":main:final:0")
                .replace(":sub-0:model", ":sub:0")
            ]
            assert sample.response_ids == source.response_ids
            assert sample.old_log_probs == source.response_log_probs


@pytest.mark.asyncio
async def test_zero_spawn_sub_phase_contributes_baseline_but_skips_sub_update() -> None:
    main_id = PolicyId("main")
    sub_id = PolicyId("sub")
    backend = MockTrainingBackend((main_id, sub_id))
    registry = _registry(backend, main_id, sub_id)

    def main_script(request: GenerationRequest) -> str:
        if ":main_update:" in request.request_id and ":main:initial:" in request.request_id:
            return '{"kind":"spawn","subtasks":["lookup"]}'
        return '{"kind":"answer","answer":"direct"}'

    main = _ScriptedPolicyService(backend, main_id, main_script)
    sub = _ScriptedPolicyService(backend, sub_id, lambda request: "evidence")
    runner = TrainableAlternatingCycleRunner(
        registry,
        backend,
        _orchestrator(registry=registry, main=main, sub=sub),
        RewardComposer(_IndexedReward(), RewardConfig()),
        rollouts_per_task=2,
    )

    result = await runner.run_cycle(
        cycle_id="cycle-zero",
        tasks=(BenchmarkTask(task_id=TaskId("task-1"), prompt="research"),),
    )

    assert result.updates.main_update is not None
    assert result.updates.sub_update is None
    assert result.updates.empty_sub_batch is True
    assert backend.weight_version(main_id).optimizer_step == 1
    assert backend.weight_version(sub_id).optimizer_step == 0
    assert len(result.phases) == 2
    sub_phase = result.phases[1]
    assert sub_phase.batch.samples == ()
    assert [trace.spawn_count for trace in sub_phase.groups[0].traces] == [0, 0]
    assert len(sub_phase.groups[0].advantages.advantages) == 2


@pytest.mark.asyncio
async def test_shared_policy_joint_cycle_uses_one_physical_update() -> None:
    shared_id = PolicyId("shared")
    backend = MockTrainingBackend((shared_id,))
    registry = _registry(backend, shared_id, shared_id)

    def script(request: GenerationRequest) -> str:
        if request.agent_role == "sub":
            return "evidence"
        if ":main:initial:" in request.request_id:
            return '{"kind":"spawn","subtasks":["lookup"]}'
        return '{"kind":"answer","answer":"done"}'

    shared = _ScriptedPolicyService(backend, shared_id, script)
    runner = TrainableAlternatingCycleRunner(
        registry,
        backend,
        _orchestrator(registry=registry, main=shared, sub=shared),
        RewardComposer(_IndexedReward(), RewardConfig()),
        rollouts_per_task=2,
    )

    result = await runner.run_cycle(
        cycle_id="cycle-shared",
        tasks=(BenchmarkTask(task_id=TaskId("task-1"), prompt="research"),),
    )

    assert result.updates.joint_update is not None
    assert result.updates.main_update is None
    assert result.updates.sub_update is None
    assert backend.weight_version(shared_id).optimizer_step == 1
    assert len(result.phases) == 1
    assert {sample.agent_role for sample in result.phases[0].batch.samples} == {
        "main",
        "sub",
    }


@pytest.mark.asyncio
async def test_transaction_context_rejects_dataset_or_environment_drift(
    tmp_path: Path,
) -> None:
    main_id = PolicyId("main")
    sub_id = PolicyId("sub")
    backend = MockTrainingBackend((main_id, sub_id))
    registry = _registry(backend, main_id, sub_id)

    def main_script(request: GenerationRequest) -> str:
        if ":main:initial:" in request.request_id:
            return '{"kind":"spawn","subtasks":["lookup"]}'
        return '{"kind":"answer","answer":"done"}'

    reward = RewardComposer(_IndexedReward(), RewardConfig())
    runner = TrainableAlternatingCycleRunner(
        registry,
        backend,
        _orchestrator(
            registry=registry,
            main=_ScriptedPolicyService(backend, main_id, main_script),
            sub=_ScriptedPolicyService(backend, sub_id, lambda request: "evidence"),
        ),
        reward,
        rollouts_per_task=2,
        transaction_store=FilePhaseTransactionStore(tmp_path / "transactions"),
    )
    base_context = dict(
        experiment_id="drift-check",
        config_digest="config",
        rng_state="rng",
        sampler_state="sampler",
        corpus_revision="corpus",
        tool_revision="unspecified",
        prompt_revision="unspecified",
        judge_revision="judge",
        reward_revision=reward.revision,
    )
    task = BenchmarkTask(task_id=TaskId("task-1"), prompt="research")

    with pytest.raises(ConfigurationError, match="dataset revision"):
        await runner.run_cycle(
            cycle_id="dataset-drift",
            tasks=(task,),
            transaction_context=PhaseTransactionContext(
                **base_context,
                dataset_revision="another-dataset",
                environment_snapshot="deterministic-v1",
            ),
        )

    with pytest.raises(ConfigurationError, match="environment"):
        await runner.run_cycle(
            cycle_id="environment-drift",
            tasks=(task,),
            transaction_context=PhaseTransactionContext(
                **base_context,
                dataset_revision="unspecified",
                environment_snapshot="stale-environment",
            ),
        )
    assert backend.weight_version(main_id).optimizer_step == 0
    assert backend.weight_version(sub_id).optimizer_step == 0
