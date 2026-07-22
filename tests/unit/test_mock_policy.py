import pytest

from heterospawn.domain.ids import AgentInstanceId, EpisodeId, PolicyId, RolloutId, TaskId
from heterospawn.policies.base import EvaluationGenerationRequest, Message
from heterospawn.policies.mock import MockEvaluationPolicy


@pytest.mark.asyncio
async def test_external_api_contract_is_explicitly_not_trainable() -> None:
    policy = MockEvaluationPolicy(PolicyId("main_policy"), "done")
    request = EvaluationGenerationRequest(
        request_id="request-1",
        task_id=TaskId("task-1"),
        episode_id=EpisodeId("episode-1"),
        rollout_id=RolloutId("rollout-1"),
        agent_role="main",
        agent_instance_id=AgentInstanceId("main-0"),
        messages=(Message(role="user", content="question"),),
    )

    result = await policy.generate(request)

    assert result.content == "done"
    assert result.policy_id == PolicyId("main_policy")
    assert result.capabilities.trainable is False
    assert result.capabilities.returns_token_ids is False
    assert result.capabilities.returns_old_log_probs is False
