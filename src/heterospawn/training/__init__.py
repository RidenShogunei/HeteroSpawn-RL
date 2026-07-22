"""Backend-independent policy training state machines."""

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
    "TrainingBatchBuilder",
    "normalize_outcome_advantages",
]
