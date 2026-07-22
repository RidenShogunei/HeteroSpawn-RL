"""Project-specific error hierarchy."""


class HeteroSpawnError(Exception):
    """Base class for expected project failures."""


class ConfigurationError(HeteroSpawnError):
    """Configuration is missing or internally inconsistent."""


class WeightVersionMismatch(HeteroSpawnError):
    """A training operation received an unexpected immutable weight version."""


class RolloutRevisionMismatch(HeteroSpawnError):
    """A generation operation reached a rollout service at the wrong revision."""


class ProviderRequestError(HeteroSpawnError):
    """An external provider request failed after bounded retries."""


class SearchRequestError(HeteroSpawnError):
    """An external search request failed after bounded retries."""


class BenchmarkDataError(HeteroSpawnError):
    """Benchmark input is missing, malformed, or unsafe to expose."""


class InvalidActionError(HeteroSpawnError):
    """A policy emitted an action that violates the orchestration schema."""
