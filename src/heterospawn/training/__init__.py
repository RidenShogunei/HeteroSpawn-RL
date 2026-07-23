"""Backend-independent policy training state machines."""

from heterospawn.training.base import PolicyService, RolloutArtifactProvider, TrainingBackend
from heterospawn.training.batch import (
    OutcomeAdvantageGroup,
    TrainingBatchBuilder,
    normalize_outcome_advantages,
)
from heterospawn.training.coordinator import AlternatingCoordinator, CycleResult
from heterospawn.training.mock import MockTrainingBackend
from heterospawn.training.registry import PolicyRegistry

__all__ = [
    "AlternatingCoordinator",
    "CycleResult",
    "MockTrainingBackend",
    "OutcomeAdvantageGroup",
    "PolicyRegistry",
    "PolicyService",
    "RolloutArtifactProvider",
    "TrainingBackend",
    "TrainingBatchBuilder",
    "normalize_outcome_advantages",
]
