"""Deterministic CPU-only evaluation policy."""

from __future__ import annotations

import hashlib

from heterospawn.domain.ids import PolicyId
from heterospawn.policies.base import (
    EvaluationGenerationRequest,
    EvaluationGenerationResult,
    ExternalModelRevision,
    PolicyCapabilities,
    TokenUsage,
)


class MockEvaluationPolicy:
    """Returns a fixed response while preserving request and policy metadata."""

    def __init__(self, policy_id: PolicyId, response: str) -> None:
        self._policy_id = policy_id
        self._response = response
        self._revision = ExternalModelRevision(
            provider="mock",
            model="deterministic-v1",
            api_base="memory://mock",
        )

    @property
    def policy_id(self) -> PolicyId:
        return self._policy_id

    @property
    def revision(self) -> ExternalModelRevision:
        return self._revision

    async def generate(self, request: EvaluationGenerationRequest) -> EvaluationGenerationResult:
        digest = hashlib.sha256(self._response.encode("utf-8")).hexdigest()
        return EvaluationGenerationResult(
            request_id=request.request_id,
            policy_id=self.policy_id,
            revision=self.revision,
            provider_request_id=f"mock:{request.request_id}",
            content=self._response,
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            raw_response_digest=digest,
            capabilities=PolicyCapabilities(
                trainable=False,
                returns_token_ids=False,
                returns_old_log_probs=False,
            ),
        )
