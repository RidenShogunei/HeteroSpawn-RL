"""Backend-independent domain types."""

from heterospawn.domain.ids import (
    AgentInstanceId,
    EpisodeId,
    PolicyId,
    RolloutId,
    StepId,
    TaskId,
)
from heterospawn.domain.training import (
    CheckpointRef,
    GenerationRequest,
    GenerationResult,
    PolicyTrainingBatch,
    PolicyTrainingSample,
    TrajectoryStep,
    UpdateResult,
)
from heterospawn.domain.versions import RoleBinding, RolloutRevision, WeightVersion

__all__ = [
    "AgentInstanceId",
    "CheckpointRef",
    "EpisodeId",
    "GenerationRequest",
    "GenerationResult",
    "PolicyId",
    "PolicyTrainingBatch",
    "PolicyTrainingSample",
    "RoleBinding",
    "RolloutId",
    "RolloutRevision",
    "StepId",
    "TaskId",
    "TrajectoryStep",
    "UpdateResult",
    "WeightVersion",
]
