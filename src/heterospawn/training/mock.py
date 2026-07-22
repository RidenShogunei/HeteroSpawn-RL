"""Deterministic in-memory backend used as the executable contract oracle."""

from __future__ import annotations

from dataclasses import dataclass

from heterospawn.domain.ids import CheckpointId, PolicyId
from heterospawn.domain.training import (
    CheckpointRef,
    GenerationRequest,
    GenerationResult,
    PolicyTrainingBatch,
    UpdateResult,
    canonical_digest,
)
from heterospawn.domain.versions import RolloutRevision, WeightVersion
from heterospawn.errors import (
    CheckpointIntegrityError,
    RolloutRevisionMismatch,
    TrainingBatchError,
    WeightVersionMismatch,
)


@dataclass
class _PolicyState:
    weight: WeightVersion
    rollout: RolloutRevision
    parameter_hash: str
    checkpoint: CheckpointRef


class MockPolicyEndpoint:
    def __init__(self, backend: MockTrainingBackend, policy_id: PolicyId) -> None:
        self._backend = backend
        self._policy_id = policy_id

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
            raise RolloutRevisionMismatch("mock rollout revision mismatch")
        token = (sum(request.prompt_ids) + current.replica_set_revision) % 97
        result = GenerationResult(
            request_id=request.request_id,
            policy_id=self._policy_id,
            rollout_revision=current,
            response_ids=(token, 2),
            response_log_probs=(-0.25, -0.5),
            stop_reason="eos",
            usage=(("prompt_tokens", len(request.prompt_ids)), ("completion_tokens", 2)),
        )
        if self._backend.rollout_revision(self._policy_id) != expected_revision:
            raise RolloutRevisionMismatch("mock rollout changed during generation")
        return result


class MockTrainingBackend:
    """Models pending training weights separately from synchronized rollout state."""

    def __init__(self, policy_ids: tuple[PolicyId, ...]) -> None:
        self._states: dict[PolicyId, _PolicyState] = {}
        self._checkpoints: dict[CheckpointId, CheckpointRef] = {}
        self._updates: dict[str, tuple[str, UpdateResult]] = {}
        self._synced: dict[tuple[PolicyId, str], RolloutRevision] = {}
        for policy_id in policy_ids:
            parameter_hash = canonical_digest({"policy_id": policy_id, "step": 0})
            weight = WeightVersion(
                policy_id=policy_id,
                optimizer_step=0,
                checkpoint_digest=parameter_hash,
            )
            checkpoint = self._make_checkpoint(weight, parameter_hash)
            rollout = RolloutRevision(
                policy_id=policy_id,
                weight_version=weight,
                deployment_id=f"mock:{policy_id}",
                replica_set_revision=0,
            )
            self._states[policy_id] = _PolicyState(
                weight=weight,
                rollout=rollout,
                parameter_hash=parameter_hash,
                checkpoint=checkpoint,
            )
            self._checkpoints[checkpoint.checkpoint_id] = checkpoint

    def endpoint(self, policy_id: PolicyId) -> MockPolicyEndpoint:
        self._state(policy_id)
        return MockPolicyEndpoint(self, policy_id)

    def weight_version(self, policy_id: PolicyId) -> WeightVersion:
        return self._state(policy_id).weight

    def rollout_revision(self, policy_id: PolicyId) -> RolloutRevision:
        return self._state(policy_id).rollout

    def parameter_hash(self, policy_id: PolicyId) -> str:
        return self._state(policy_id).parameter_hash

    async def update_policy(
        self,
        policy_id: PolicyId,
        batch: PolicyTrainingBatch,
        expected_base_version: WeightVersion,
    ) -> UpdateResult:
        state = self._state(policy_id)
        previous = self._updates.get(batch.batch_id)
        if previous is not None:
            previous_digest, result = previous
            if previous_digest != batch.batch_digest:
                raise TrainingBatchError("batch_id was already used with another digest")
            return result
        if not batch.samples:
            raise TrainingBatchError("empty batches must be skipped by the coordinator")
        if batch.target_policy_id != policy_id:
            raise TrainingBatchError("batch targets another policy")
        if expected_base_version != batch.expected_base_version:
            raise WeightVersionMismatch("call and batch base versions differ")
        if state.weight != expected_base_version:
            raise WeightVersionMismatch("training backend is not at expected base version")

        parameter_hash = canonical_digest(
            {
                "base": state.parameter_hash,
                "batch_digest": batch.batch_digest,
                "optimizer_step": state.weight.optimizer_step + 1,
            }
        )
        trained = WeightVersion(
            policy_id=policy_id,
            optimizer_step=state.weight.optimizer_step + 1,
            checkpoint_digest=parameter_hash,
        )
        checkpoint = self._make_checkpoint(trained, parameter_hash)
        result = UpdateResult(
            policy_id=policy_id,
            base_version=state.weight,
            trained_version=trained,
            checkpoint=checkpoint,
            metrics=(("sample_count", float(len(batch.samples))),),
        )
        state.weight = trained
        state.parameter_hash = parameter_hash
        state.checkpoint = checkpoint
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        self._updates[batch.batch_id] = (batch.batch_digest, result)
        return result

    async def sync_rollout_weights(
        self,
        policy_id: PolicyId,
        trained_version: WeightVersion,
    ) -> RolloutRevision:
        state = self._state(policy_id)
        if state.weight != trained_version:
            raise WeightVersionMismatch("cannot sync unknown or stale training weights")
        key = (policy_id, trained_version.checkpoint_digest)
        existing = self._synced.get(key)
        if existing is not None:
            return existing
        if state.rollout.weight_version == trained_version:
            return state.rollout
        rollout = RolloutRevision(
            policy_id=policy_id,
            weight_version=trained_version,
            deployment_id=state.rollout.deployment_id,
            replica_set_revision=state.rollout.replica_set_revision + 1,
        )
        state.rollout = rollout
        self._synced[key] = rollout
        return rollout

    async def save_checkpoint(self, policy_id: PolicyId) -> CheckpointRef:
        return self._state(policy_id).checkpoint

    async def restore_checkpoint(self, checkpoint: CheckpointRef) -> WeightVersion:
        known = self._checkpoints.get(checkpoint.checkpoint_id)
        if known != checkpoint:
            raise CheckpointIntegrityError("checkpoint is unknown or modified")
        state = self._state(checkpoint.policy_id)
        state.weight = checkpoint.weight_version
        state.parameter_hash = checkpoint.weight_version.checkpoint_digest
        state.checkpoint = checkpoint
        return state.weight

    def _state(self, policy_id: PolicyId) -> _PolicyState:
        try:
            return self._states[policy_id]
        except KeyError:
            raise TrainingBatchError(f"unknown policy: {policy_id}") from None

    @staticmethod
    def _make_checkpoint(weight: WeightVersion, parameter_hash: str) -> CheckpointRef:
        checkpoint_id = CheckpointId(
            f"{weight.policy_id}:step-{weight.optimizer_step}:{weight.checkpoint_digest[:12]}"
        )
        return CheckpointRef(
            checkpoint_id=checkpoint_id,
            policy_id=weight.policy_id,
            weight_version=weight,
            uri=f"memory://{checkpoint_id}",
            optimizer_state_digest=canonical_digest(
                {"parameter_hash": parameter_hash, "step": weight.optimizer_step}
            ),
        )
