"""Backend-independent policy training state machines."""

from heterospawn.training.base import PolicyService, RolloutArtifactProvider, TrainingBackend
from heterospawn.training.batch import (
    OutcomeAdvantageGroup,
    TrainingBatchBuilder,
    normalize_outcome_advantages,
)
from heterospawn.training.coordinator import (
    AlternatingCoordinator,
    CycleResult,
    PhaseTransactionHook,
)
from heterospawn.training.episode_cycle import (
    EpisodeReward,
    OutcomeRewardService,
    PhaseOutcomeRewardService,
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
from heterospawn.training.transactions import (
    FilePhaseTransactionStore,
    PhaseCommitManifest,
    PhasePendingUpdate,
    PhaseRecoveryManifest,
    PhaseTransactionContext,
    PhaseTransactionEvidence,
    PhaseTransactionInput,
    PhaseTransactionManager,
)
from heterospawn.training.wideseek_reward import (
    RoleRewardTotals,
    WideSeekRewardBreakdown,
    WideSeekRewardConfig,
    WideSeekRewardService,
)

__all__ = [
    "AlternatingCoordinator",
    "CycleResult",
    "EpisodeReward",
    "FilePhaseTransactionStore",
    "MockTrainingBackend",
    "OutcomeAdvantageGroup",
    "OutcomeRewardService",
    "PhaseCommitManifest",
    "PhaseOutcomeRewardService",
    "PhasePendingUpdate",
    "PhaseRecoveryManifest",
    "PhaseRolloutResult",
    "PhaseTransactionContext",
    "PhaseTransactionEvidence",
    "PhaseTransactionHook",
    "PhaseTransactionInput",
    "PhaseTransactionManager",
    "PolicyRegistry",
    "PolicyService",
    "RewardComposer",
    "RewardConfig",
    "RoleRewardTotals",
    "RolloutArtifactProvider",
    "TaskRolloutGroup",
    "TrainableAlternatingCycleRunner",
    "TrainableCycleResult",
    "TrainableRolloutBatchFactory",
    "TrainingBackend",
    "TrainingBatchBuilder",
    "WideSeekRewardBreakdown",
    "WideSeekRewardConfig",
    "WideSeekRewardService",
    "normalize_outcome_advantages",
]
