from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from heterospawn.benchmarks.wideseek import load_wideseek_dataset
from heterospawn.domain.ids import (
    AgentInstanceId,
    EpisodeId,
    PolicyId,
    RolloutId,
    StepId,
    TaskId,
)
from heterospawn.domain.training import TrajectoryStep, canonical_digest
from heterospawn.evaluation.wideseek import WideSeekEvaluator
from heterospawn.orchestration.models import BudgetSnapshot
from heterospawn.orchestration.trainable_models import (
    ToolOutcomeRecord,
    TrainableEnvironmentSnapshot,
    TrainableEpisodeEvent,
    TrainableEpisodeTrace,
    TrainableMainAttempt,
    TrainableSubOutcome,
)
from heterospawn.training import MockTrainingBackend
from heterospawn.training.wideseek_reward import (
    WideSeekRewardConfig,
    WideSeekRewardService,
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _tool_outcome(
    *,
    step_id: StepId,
    tool_name: str,
    source_search_step_id: StepId | None = None,
) -> ToolOutcomeRecord:
    request = {"name": tool_name}
    result = {"status": "ok", "tool": tool_name}
    return ToolOutcomeRecord(
        step_id=step_id,
        agent_instance_id=AgentInstanceId("sub-0"),
        spawn_round=0,
        sub_turn=0,
        request_index=0 if tool_name == "search" else 1,
        tool_name=tool_name,  # type: ignore[arg-type]
        status="success",
        request_json=_canonical_json(request),
        request_digest=canonical_digest(request),
        result_json=_canonical_json(result),
        result_digest=canonical_digest(result),
        source_search_step_id=source_search_step_id,
        provider_revision="memory@1",
    )


@pytest.mark.asyncio
async def test_reward_breakdown_exposes_shared_main_and_system_sub_totals(
    tmp_path: Path,
) -> None:
    answer = "```markdown\n| Name | City |\n|---|---|\n| A | Paris |\n```"
    record = {"question": "table", "answer": answer, "unique_columns": ["Name"]}
    content = json.dumps(record, separators=(",", ":")) + "\n"
    path = tmp_path / "width_20k.jsonl"
    path.write_bytes(content.encode())
    dataset = load_wideseek_dataset(
        path,
        split="width_20k",
        expected_sha256=hashlib.sha256(content.encode()).hexdigest(),
    )
    task = dataset.tasks[0]

    main_id = PolicyId("main")
    revision = MockTrainingBackend((main_id,)).rollout_revision(main_id)
    rollout_id = RolloutId("rollout")
    episode_id = EpisodeId("episode")
    model_step_id = StepId("rollout:main")
    search_step_id = StepId("rollout:search")
    access_step_id = StepId("rollout:access")
    environment = TrainableEnvironmentSnapshot(
        budget=BudgetSnapshot(capacity=2, reserved=0, committed=0)
    )
    model_step = TrajectoryStep(
        task_id=TaskId(task.task_id),
        episode_id=episode_id,
        rollout_id=rollout_id,
        step_id=model_step_id,
        event_index=0,
        agent_role="main",
        agent_instance_id=AgentInstanceId("main"),
        policy_id=main_id,
        rollout_revision=revision,
        prompt_ids=(1,),
        response_ids=(2, 3, 4, 5),
        response_log_probs=(-0.1, -0.1, -0.1, -0.1),
        tokenizer_revision="tokenizer",
        prompt_template_revision="prompt",
        stop_reason="eos",
    )
    search_outcome = _tool_outcome(step_id=search_step_id, tool_name="search")
    access_outcome = _tool_outcome(
        step_id=access_step_id,
        tool_name="access",
        source_search_step_id=search_step_id,
    )
    trace = TrainableEpisodeTrace(
        task_id=task.task_id,
        episode_id=episode_id,
        rollout_id=rollout_id,
        status="success",
        answer=answer,
        spawn_count=1,
        main_attempts=(
            TrainableMainAttempt(
                phase="final",
                attempt_index=0,
                step_id=model_step_id,
                content=answer,
                valid=True,
                action_kind="answer",
            ),
        ),
        sub_outcomes=(
            TrainableSubOutcome(
                agent_instance_id=AgentInstanceId("sub-0"),
                subtask="lookup",
                status="success",
                content="evidence",
                search_step_id=search_step_id,
                tool_step_ids=(search_step_id, access_step_id),
            ),
        ),
        tool_outcomes=(search_outcome, access_outcome),
        evidence=(),
        model_steps=(model_step,),
        events=(
            TrainableEpisodeEvent(
                event_index=0,
                step_id=model_step_id,
                kind="model",
                agent_role="main",
                agent_instance_id=AgentInstanceId("main"),
                status="valid",
                phase="final",
                payload_digest=canonical_digest({"ids": [2, 3, 4, 5]}),
                environment=environment,
            ),
            TrainableEpisodeEvent(
                event_index=1,
                step_id=search_step_id,
                kind="search",
                agent_role="sub",
                agent_instance_id=AgentInstanceId("sub-0"),
                causal_step_ids=(model_step_id,),
                status="success",
                phase="sub",
                payload_digest=search_outcome.result_digest,
                environment=environment,
            ),
            TrainableEpisodeEvent(
                event_index=2,
                step_id=access_step_id,
                kind="access",
                agent_role="sub",
                agent_instance_id=AgentInstanceId("sub-0"),
                causal_step_ids=(search_step_id,),
                status="success",
                phase="sub",
                payload_digest=access_outcome.result_digest,
                environment=environment,
            ),
        ),
        policy_revisions=((main_id, revision),),
    )
    service = WideSeekRewardService(
        WideSeekEvaluator(dataset),
        WideSeekRewardConfig(
            format_reward=0.1,
            access_credit=0.2,
            length_limit=100,
            max_length_limit=200,
            spawn_cost=0.1,
            search_cost=0.05,
            token_cost=0.01,
        ),
    )
    breakdown = await service.score_breakdown(task, trace)

    assert breakdown.outcome == 1.0
    assert breakdown.search_calls == breakdown.access_calls == 1
    assert breakdown.generated_tokens == 4
    assert breakdown.role_totals.shared == pytest.approx(1.3)
    assert breakdown.role_totals.main == pytest.approx(1.06)
    assert breakdown.role_totals.sub == pytest.approx(1.0)
    assert await service.score(task, trace) == pytest.approx(1.06)
