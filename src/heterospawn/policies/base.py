"""Provider-neutral policy contracts for API-first evaluation."""

from __future__ import annotations

from typing import Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from heterospawn.domain.ids import AgentInstanceId, EpisodeId, PolicyId, RolloutId, TaskId
from heterospawn.domain.versions import AgentRole

JsonScalar: TypeAlias = None | bool | int | float | str


class Message(BaseModel):
    """A provider-neutral chat message."""

    model_config = ConfigDict(frozen=True, strict=True)

    role: Literal["system", "user", "assistant"]
    content: str


class PolicyCapabilities(BaseModel):
    """Declares whether output is eligible for training use."""

    model_config = ConfigDict(frozen=True, strict=True)

    trainable: bool
    returns_token_ids: bool
    returns_old_log_probs: bool


class ExternalModelRevision(BaseModel):
    """Auditable provider/model identity, not a trainable rollout revision."""

    model_config = ConfigDict(frozen=True, strict=True)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_base: str = Field(min_length=1)


class TokenUsage(BaseModel):
    """Provider-reported token accounting."""

    model_config = ConfigDict(frozen=True, strict=True)

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class EvaluationGenerationRequest(BaseModel):
    """Text-level request used for benchmark and orchestration validation."""

    model_config = ConfigDict(frozen=True, strict=True)

    request_id: str = Field(min_length=1)
    task_id: TaskId
    episode_id: EpisodeId
    rollout_id: RolloutId
    agent_role: AgentRole
    agent_instance_id: AgentInstanceId
    messages: tuple[Message, ...] = Field(min_length=1)
    # Generation sampling parameters are deliberately scalar in the shared
    # contract.  The tuple representation keeps a frozen model deeply
    # immutable instead of hiding a mutable dict inside it.
    sampling_params: tuple[tuple[str, JsonScalar], ...] = ()


class EvaluationGenerationResult(BaseModel):
    """Text-level provider response retained for auditing, never implicit training."""

    model_config = ConfigDict(frozen=True, strict=True)

    request_id: str
    policy_id: PolicyId
    revision: ExternalModelRevision
    provider_request_id: str
    content: str
    reasoning_content: str | None = None
    finish_reason: str
    usage: TokenUsage
    raw_response_digest: str
    capabilities: PolicyCapabilities


class EvaluationPolicyService(Protocol):
    """Common inference surface for API benchmark validation."""

    @property
    def policy_id(self) -> PolicyId: ...

    @property
    def revision(self) -> ExternalModelRevision: ...

    async def generate(
        self, request: EvaluationGenerationRequest
    ) -> EvaluationGenerationResult: ...
