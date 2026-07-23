"""Backend-independent policy training state machines."""

from heterospawn.training.base import PolicyService, RolloutArtifactProvider, TrainingBackend
from heterospawn.training.batch import (
    OutcomeAdvantageGroup,
    TrainingBatchBuilder,
    normalize_outcome_advantages,
)
from heterospawn.training.coordinator import AlternatingCoordinator, CycleResult
from heterospawn.training.episode_cycle import (
    EpisodeReward,
    OutcomeRewardService,
    PhaseRolloutResult,
    RewardComposer,
    RewardConfig,
    TaskRolloutGroup,
    TrainableAlternatingCycleRunner,
    TrainableCycleResult,
    TrainableRolloutBatchFactory,
)
from heterospawn.training.mock import MockTrainingBackend
from heterospawn.training.registry import PolicyRegistry

__all__ = [
    "AlternatingCoordinator",
    "CycleResult",
    "EpisodeReward",
    "MockTrainingBackend",
    "OutcomeAdvantageGroup",
    "OutcomeRewardService",
    "PhaseRolloutResult",
    "PolicyRegistry",
    "PolicyService",
    "RewardComposer",
    "RewardConfig",
    "RolloutArtifactProvider",
    "TaskRolloutGroup",
    "TrainableAlternatingCycleRunner",
    "TrainableCycleResult",
    "TrainableRolloutBatchFactory",
    "TrainingBackend",
    "TrainingBatchBuilder",
    "normalize_outcome_advantages",
]
