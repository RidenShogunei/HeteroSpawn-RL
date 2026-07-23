"""Project-specific error hierarchy."""


class HeteroSpawnError(Exception):
    """Base class for expected project failures."""


class ConfigurationError(HeteroSpawnError):
    """Configuration is missing or internally inconsistent."""


class WeightVersionMismatch(HeteroSpawnError):
    """A training operation received an unexpected immutable weight version."""


class RolloutRevisionMismatch(HeteroSpawnError):
    """A generation operation reached a rollout service at the wrong revision."""


class RolloutServiceError(HeteroSpawnError):
    """A rollout worker failed to start, serve, synchronize, or recover."""


class TrainingBatchError(HeteroSpawnError):
    """A training batch is empty, inconsistent, or has a conflicting digest."""


class PhaseTransactionError(HeteroSpawnError):
    """A phase transaction is missing, conflicting, corrupt, or unrecoverable."""


class CheckpointIntegrityError(HeteroSpawnError):
    """A checkpoint is unknown or fails immutable identity validation."""


class ProviderRequestError(HeteroSpawnError):
    """An external provider request failed after bounded retries."""


class SearchRequestError(HeteroSpawnError):
    """An external search request failed after bounded retries."""


class JudgeRequestError(HeteroSpawnError):
    """A benchmark judge request failed or returned an invalid verdict."""


class BenchmarkDataError(HeteroSpawnError):
    """Benchmark input is missing, malformed, or unsafe to expose."""


class InvalidActionError(HeteroSpawnError):
    """A policy emitted an action that violates the orchestration schema."""


class EpisodeRunError(HeteroSpawnError):
    """Safe partial metrics for an episode that could not produce a final answer."""

    def __init__(
        self,
        *,
        error_code: str,
        spawn_count: int,
        successful_subs: int,
        failed_subs: int,
        main_attempts: int,
        invalid_main_attempts: int,
        event_count: int,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> None:
        super().__init__("episode failed after retaining safe partial metrics")
        self.error_code = error_code
        self.spawn_count = spawn_count
        self.successful_subs = successful_subs
        self.failed_subs = failed_subs
        self.main_attempts = main_attempts
        self.invalid_main_attempts = invalid_main_attempts
        self.event_count = event_count
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
