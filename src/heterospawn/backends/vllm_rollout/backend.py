"""Training-backend facade that delegates rollout synchronization to vLLM."""

from __future__ import annotations

from heterospawn.backends.vllm_rollout.models import (
    VllmPolicyDeployment,
    VllmRolloutConfig,
    VllmWorkerFactory,
    VllmWorkerRuntime,
)
from heterospawn.backends.vllm_rollout.service import (
    VllmPolicyEndpoint,
    VllmRolloutService,
)
from heterospawn.domain.ids import PolicyId
from heterospawn.domain.training import (
    CheckpointRef,
    PolicyTrainingBatch,
    UpdateResult,
)
from heterospawn.domain.versions import RolloutRevision, WeightVersion
from heterospawn.training.base import RolloutArtifactProvider, TrainingBackend


class VllmRolloutBackend:
    """Composes any artifact-producing trainer with standalone vLLM rollout workers."""

    def __init__(
        self,
        *,
        training_backend: TrainingBackend,
        artifact_provider: RolloutArtifactProvider,
        rollout_service: VllmRolloutService,
    ) -> None:
        self._training_backend = training_backend
        self._artifact_provider = artifact_provider
        self._rollout_service = rollout_service

    @classmethod
    async def create(
        cls,
        *,
        config: VllmRolloutConfig,
        training_backend: TrainingBackend,
        artifact_provider: RolloutArtifactProvider,
        deployments: tuple[VllmPolicyDeployment, ...],
        worker_factory: VllmWorkerFactory,
    ) -> VllmRolloutBackend:
        initial = []
        for deployment in deployments:
            checkpoint = await training_backend.save_checkpoint(deployment.policy_id)
            artifact = await artifact_provider.export_rollout_artifact(
                deployment.policy_id,
                checkpoint.weight_version,
            )
            initial.append((deployment, artifact))
        service = await VllmRolloutService.start(
            config=config,
            deployments=tuple(initial),
            worker_factory=worker_factory,
        )
        return cls(
            training_backend=training_backend,
            artifact_provider=artifact_provider,
            rollout_service=service,
        )

    def endpoint(self, policy_id: PolicyId) -> VllmPolicyEndpoint:
        return self._rollout_service.endpoint(policy_id)

    def rollout_revision(self, policy_id: PolicyId) -> RolloutRevision:
        return self._rollout_service.rollout_revision(policy_id)

    async def runtime_metrics(self, policy_id: PolicyId) -> VllmWorkerRuntime:
        return await self._rollout_service.runtime_metrics(policy_id)

    async def update_policy(
        self,
        policy_id: PolicyId,
        batch: PolicyTrainingBatch,
        expected_base_version: WeightVersion,
    ) -> UpdateResult:
        return await self._training_backend.update_policy(
            policy_id,
            batch,
            expected_base_version,
        )

    async def sync_rollout_weights(
        self,
        policy_id: PolicyId,
        trained_version: WeightVersion,
    ) -> RolloutRevision:
        artifact = await self._artifact_provider.export_rollout_artifact(
            policy_id,
            trained_version,
        )
        return await self._rollout_service.sync_rollout_weights(
            policy_id,
            trained_version,
            artifact,
        )

    async def save_checkpoint(self, policy_id: PolicyId) -> CheckpointRef:
        return await self._training_backend.save_checkpoint(policy_id)

    async def restore_checkpoint(self, checkpoint: CheckpointRef) -> WeightVersion:
        return await self._training_backend.restore_checkpoint(checkpoint)

    async def close(self) -> None:
        await self._rollout_service.close()
