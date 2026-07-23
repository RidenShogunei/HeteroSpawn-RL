"""Backend-independent domain types."""

from heterospawn.domain.ids import (
    AgentInstanceId,
    EpisodeId,
    PolicyId,
    RolloutId,
    StepId,
    TaskId,
)
from heterospawn.domain.tasks import ResearchTask
from heterospawn.domain.training import (
    CheckpointRef,
    GenerationRequest,
    GenerationResult,
    PolicyTrainingBatch,
    PolicyTrainingSample,
    PromptEncoding,
    RolloutArtifact,
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
    "PromptEncoding",
    "ResearchTask",
    "RoleBinding",
    "RolloutArtifact",
    "RolloutId",
    "RolloutRevision",
    "StepId",
    "TaskId",
    "TrajectoryStep",
    "UpdateResult",
    "WeightVersion",
]
