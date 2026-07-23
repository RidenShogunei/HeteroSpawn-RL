"""Crash-safe phase transaction records and recovery."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from heterospawn.domain.ids import EpisodeId, PolicyId, RolloutId, TaskId
from heterospawn.domain.training import (
    CheckpointRef,
    PolicyTrainingBatch,
    TrainingPhase,
    UpdateResult,
    canonical_digest,
)
from heterospawn.domain.versions import RolloutRevision
from heterospawn.errors import PhaseTransactionError
from heterospawn.training.base import TrainingBackend
from heterospawn.training.registry import PolicyRegistry

PHASE_TRANSACTION_SCHEMA_REVISION: Literal["heterospawn-phase-transaction-v1"] = (
    "heterospawn-phase-transaction-v1"
)
ModelT = TypeVar("ModelT", bound=BaseModel)


class PhaseTransactionContext(BaseModel):
    """Versioned experiment state required to reproduce a phase input."""

    model_config = ConfigDict(frozen=True, strict=True)

    experiment_id: str = Field(min_length=1)
    config_digest: str = Field(min_length=1)
    rng_state: str = Field(min_length=1)
    sampler_state: str = Field(min_length=1)
    dataset_revision: str = Field(min_length=1)
    corpus_revision: str = Field(default="unspecified", min_length=1)
    tool_revision: str = Field(default="unspecified", min_length=1)
    prompt_revision: str = Field(default="unspecified", min_length=1)
    judge_revision: str = Field(default="unspecified", min_length=1)
    environment_snapshot: str = Field(min_length=1)
    reward_revision: str = Field(min_length=1)


class PhaseTransactionEvidence(BaseModel):
    """Safe rollout identity supplied after a phase batch has been built."""

    model_config = ConfigDict(frozen=True, strict=True)

    cycle_id: str = Field(min_length=1)
    task_ids: tuple[TaskId, ...] = Field(min_length=1)
    episode_ids: tuple[EpisodeId, ...] = Field(min_length=1)
    rollout_ids: tuple[RolloutId, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def identities_must_be_unique(self) -> PhaseTransactionEvidence:
        for label, values in (
            ("task", self.task_ids),
            ("episode", self.episode_ids),
            ("rollout", self.rollout_ids),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{label} IDs must be unique")
        return self


def _input_payload(
    *,
    transaction_id: str,
    phase: TrainingPhase,
    target_policy_id: PolicyId,
    context: PhaseTransactionContext,
    evidence: PhaseTransactionEvidence,
    base_checkpoint: CheckpointRef,
    base_policy_revisions: tuple[tuple[PolicyId, RolloutRevision], ...],
    batch: PolicyTrainingBatch,
    empty_sub_batch: bool,
) -> dict[str, object]:
    return {
        "schema_revision": PHASE_TRANSACTION_SCHEMA_REVISION,
        "transaction_id": transaction_id,
        "phase": phase,
        "target_policy_id": target_policy_id,
        "context": context.model_dump(mode="json"),
        "evidence": evidence.model_dump(mode="json"),
        "base_checkpoint": base_checkpoint.model_dump(mode="json"),
        "base_policy_revisions": [
            [policy_id, revision.model_dump(mode="json")]
            for policy_id, revision in base_policy_revisions
        ],
        "batch": batch.model_dump(mode="json"),
        "empty_sub_batch": empty_sub_batch,
    }


class PhaseTransactionInput(BaseModel):
    """Durable optimizer input written before the phase can mutate weights."""

    model_config = ConfigDict(frozen=True, strict=True)

    schema_revision: Literal["heterospawn-phase-transaction-v1"]
    transaction_id: str = Field(min_length=1)
    phase: TrainingPhase
    target_policy_id: PolicyId
    context: PhaseTransactionContext
    evidence: PhaseTransactionEvidence
    base_checkpoint: CheckpointRef
    base_policy_revisions: tuple[tuple[PolicyId, RolloutRevision], ...]
    batch: PolicyTrainingBatch
    empty_sub_batch: bool
    input_digest: str = Field(min_length=1)

    @model_validator(mode="after")
    def identity_and_digest_must_match(self) -> PhaseTransactionInput:
        expected_id = f"{self.context.experiment_id}:{self.evidence.cycle_id}:{self.phase}"
        if self.transaction_id != expected_id:
            raise ValueError("phase transaction ID does not match context")
        if self.target_policy_id != self.batch.target_policy_id:
            raise ValueError("phase transaction target does not match batch")
        if self.phase != self.batch.phase:
            raise ValueError("phase transaction phase does not match batch")
        if self.base_checkpoint.policy_id != self.target_policy_id:
            raise ValueError("base checkpoint belongs to another policy")
        if self.base_checkpoint.weight_version != self.batch.expected_base_version:
            raise ValueError("base checkpoint does not match batch base version")
        revision_map = dict(self.base_policy_revisions)
        if len(revision_map) != len(self.base_policy_revisions):
            raise ValueError("base policy revisions contain duplicate policies")
        for policy_id, revision in self.base_policy_revisions:
            if policy_id != revision.policy_id:
                raise ValueError("base policy revision key does not match policy")
        target_revision = revision_map.get(self.target_policy_id)
        if target_revision is None:
            raise ValueError("base policy revisions omit the target policy")
        if target_revision.weight_version != self.base_checkpoint.weight_version:
            raise ValueError("target rollout and base checkpoint weights do not match")
        task_ids = set(self.evidence.task_ids)
        episode_ids = set(self.evidence.episode_ids)
        rollout_ids = set(self.evidence.rollout_ids)
        for sample in self.batch.samples:
            if sample.task_id not in task_ids:
                raise ValueError("training sample task is absent from phase evidence")
            if sample.episode_id not in episode_ids:
                raise ValueError("training sample episode is absent from phase evidence")
            if sample.rollout_id not in rollout_ids:
                raise ValueError("training sample rollout is absent from phase evidence")
        if self.empty_sub_batch != (self.phase == "sub_update" and not self.batch.samples):
            raise ValueError("empty_sub_batch does not match phase batch")
        expected = canonical_digest(
            _input_payload(
                transaction_id=self.transaction_id,
                phase=self.phase,
                target_policy_id=self.target_policy_id,
                context=self.context,
                evidence=self.evidence,
                base_checkpoint=self.base_checkpoint,
                base_policy_revisions=self.base_policy_revisions,
                batch=self.batch,
                empty_sub_batch=self.empty_sub_batch,
            )
        )
        if self.input_digest != expected:
            raise ValueError("phase input digest does not match contents")
        return self

    @classmethod
    def create(
        cls,
        *,
        phase: TrainingPhase,
        target_policy_id: PolicyId,
        context: PhaseTransactionContext,
        evidence: PhaseTransactionEvidence,
        base_checkpoint: CheckpointRef,
        base_policy_revisions: tuple[tuple[PolicyId, RolloutRevision], ...],
        batch: PolicyTrainingBatch,
    ) -> PhaseTransactionInput:
        transaction_id = f"{context.experiment_id}:{evidence.cycle_id}:{phase}"
        empty_sub_batch = phase == "sub_update" and not batch.samples
        payload = _input_payload(
            transaction_id=transaction_id,
            phase=phase,
            target_policy_id=target_policy_id,
            context=context,
            evidence=evidence,
            base_checkpoint=base_checkpoint,
            base_policy_revisions=base_policy_revisions,
            batch=batch,
            empty_sub_batch=empty_sub_batch,
        )
        return cls(
            schema_revision=PHASE_TRANSACTION_SCHEMA_REVISION,
            transaction_id=transaction_id,
            phase=phase,
            target_policy_id=target_policy_id,
            context=context,
            evidence=evidence,
            base_checkpoint=base_checkpoint,
            base_policy_revisions=base_policy_revisions,
            batch=batch,
            empty_sub_batch=empty_sub_batch,
            input_digest=canonical_digest(payload),
        )


def _pending_payload(
    transaction_input: PhaseTransactionInput,
    update: UpdateResult,
) -> dict[str, object]:
    return {
        "schema_revision": PHASE_TRANSACTION_SCHEMA_REVISION,
        "transaction_id": transaction_input.transaction_id,
        "input_digest": transaction_input.input_digest,
        "update": update.model_dump(mode="json"),
    }


class PhasePendingUpdate(BaseModel):
    """Immutable update result persisted before rollout synchronization."""

    model_config = ConfigDict(frozen=True, strict=True)

    schema_revision: Literal["heterospawn-phase-transaction-v1"]
    transaction_id: str = Field(min_length=1)
    input_digest: str = Field(min_length=1)
    update: UpdateResult
    pending_digest: str = Field(min_length=1)

    @model_validator(mode="after")
    def pending_digest_must_match(self) -> PhasePendingUpdate:
        expected = canonical_digest(
            {
                "schema_revision": self.schema_revision,
                "transaction_id": self.transaction_id,
                "input_digest": self.input_digest,
                "update": self.update.model_dump(mode="json"),
            }
        )
        if self.pending_digest != expected:
            raise ValueError("pending update digest does not match contents")
        return self

    @classmethod
    def create(
        cls,
        transaction_input: PhaseTransactionInput,
        update: UpdateResult,
    ) -> PhasePendingUpdate:
        if update.policy_id != transaction_input.target_policy_id:
            raise PhaseTransactionError("pending update belongs to another policy")
        if update.base_version != transaction_input.batch.expected_base_version:
            raise PhaseTransactionError("pending update belongs to another base version")
        payload = _pending_payload(transaction_input, update)
        return cls(
            schema_revision=PHASE_TRANSACTION_SCHEMA_REVISION,
            transaction_id=transaction_input.transaction_id,
            input_digest=transaction_input.input_digest,
            update=update,
            pending_digest=canonical_digest(payload),
        )


def _commit_payload(
    *,
    transaction_input: PhaseTransactionInput,
    update: UpdateResult | None,
    rollout_revision: RolloutRevision,
) -> dict[str, object]:
    return {
        "schema_revision": PHASE_TRANSACTION_SCHEMA_REVISION,
        "transaction_id": transaction_input.transaction_id,
        "phase_completed": transaction_input.phase,
        "input_digest": transaction_input.input_digest,
        "batch_digest": transaction_input.batch.batch_digest,
        "base_checkpoint": transaction_input.base_checkpoint.model_dump(mode="json"),
        "committed_checkpoint": (
            update.checkpoint.model_dump(mode="json") if update is not None else None
        ),
        "rollout_revision": rollout_revision.model_dump(mode="json"),
        "empty_sub_batch": transaction_input.empty_sub_batch,
        "task_ids": list(transaction_input.evidence.task_ids),
        "episode_ids": list(transaction_input.evidence.episode_ids),
        "rollout_ids": list(transaction_input.evidence.rollout_ids),
        "config_digest": transaction_input.context.config_digest,
        "dataset_revision": transaction_input.context.dataset_revision,
        "environment_snapshot": transaction_input.context.environment_snapshot,
        "reward_revision": transaction_input.context.reward_revision,
    }


class PhaseCommitManifest(BaseModel):
    """Atomic public phase boundary; only this record advances coordinator state."""

    model_config = ConfigDict(frozen=True, strict=True)

    schema_revision: Literal["heterospawn-phase-transaction-v1"]
    transaction_id: str = Field(min_length=1)
    phase_completed: TrainingPhase
    input_digest: str = Field(min_length=1)
    batch_digest: str = Field(min_length=1)
    base_checkpoint: CheckpointRef
    committed_checkpoint: CheckpointRef | None
    rollout_revision: RolloutRevision
    empty_sub_batch: bool
    task_ids: tuple[TaskId, ...]
    episode_ids: tuple[EpisodeId, ...]
    rollout_ids: tuple[RolloutId, ...]
    config_digest: str = Field(min_length=1)
    dataset_revision: str = Field(min_length=1)
    environment_snapshot: str = Field(min_length=1)
    reward_revision: str = Field(min_length=1)
    manifest_digest: str = Field(min_length=1)

    @model_validator(mode="after")
    def manifest_must_be_consistent(self) -> PhaseCommitManifest:
        if self.empty_sub_batch != (self.committed_checkpoint is None):
            raise ValueError("empty phase and committed checkpoint must agree")
        expected_weight = (
            self.base_checkpoint.weight_version
            if self.committed_checkpoint is None
            else self.committed_checkpoint.weight_version
        )
        if self.rollout_revision.weight_version != expected_weight:
            raise ValueError("committed rollout does not identify committed weights")
        payload = self.model_dump(mode="json", exclude={"manifest_digest"})
        if self.manifest_digest != canonical_digest(payload):
            raise ValueError("phase manifest digest does not match contents")
        return self

    @classmethod
    def create(
        cls,
        *,
        transaction_input: PhaseTransactionInput,
        update: UpdateResult | None,
        rollout_revision: RolloutRevision,
    ) -> PhaseCommitManifest:
        if transaction_input.empty_sub_batch != (update is None):
            raise PhaseTransactionError("empty phase and update result do not agree")
        if update is not None:
            if update.policy_id != transaction_input.target_policy_id:
                raise PhaseTransactionError("committed update belongs to another policy")
            if update.base_version != transaction_input.batch.expected_base_version:
                raise PhaseTransactionError("committed update belongs to another base version")
        payload = _commit_payload(
            transaction_input=transaction_input,
            update=update,
            rollout_revision=rollout_revision,
        )
        return cls(
            schema_revision=PHASE_TRANSACTION_SCHEMA_REVISION,
            transaction_id=transaction_input.transaction_id,
            phase_completed=transaction_input.phase,
            input_digest=transaction_input.input_digest,
            batch_digest=transaction_input.batch.batch_digest,
            base_checkpoint=transaction_input.base_checkpoint,
            committed_checkpoint=update.checkpoint if update is not None else None,
            rollout_revision=rollout_revision,
            empty_sub_batch=transaction_input.empty_sub_batch,
            task_ids=transaction_input.evidence.task_ids,
            episode_ids=transaction_input.evidence.episode_ids,
            rollout_ids=transaction_input.evidence.rollout_ids,
            config_digest=transaction_input.context.config_digest,
            dataset_revision=transaction_input.context.dataset_revision,
            environment_snapshot=transaction_input.context.environment_snapshot,
            reward_revision=transaction_input.context.reward_revision,
            manifest_digest=canonical_digest(payload),
        )


class PhaseRecoveryManifest(BaseModel):
    """Append-only mapping from a committed weight to a replacement deployment revision."""

    model_config = ConfigDict(frozen=True, strict=True)

    schema_revision: Literal["heterospawn-phase-transaction-v1"]
    transaction_id: str = Field(min_length=1)
    commit_manifest_digest: str = Field(min_length=1)
    recovered_rollout_revision: RolloutRevision
    recovery_digest: str = Field(min_length=1)

    @model_validator(mode="after")
    def recovery_digest_must_match(self) -> PhaseRecoveryManifest:
        payload = self.model_dump(mode="json", exclude={"recovery_digest"})
        if self.recovery_digest != canonical_digest(payload):
            raise ValueError("phase recovery digest does not match contents")
        return self

    @classmethod
    def create(
        cls,
        commit: PhaseCommitManifest,
        revision: RolloutRevision,
    ) -> PhaseRecoveryManifest:
        if revision.weight_version != commit.rollout_revision.weight_version:
            raise PhaseTransactionError("recovered rollout loaded the wrong committed weights")
        payload = {
            "schema_revision": PHASE_TRANSACTION_SCHEMA_REVISION,
            "transaction_id": commit.transaction_id,
            "commit_manifest_digest": commit.manifest_digest,
            "recovered_rollout_revision": revision.model_dump(mode="json"),
        }
        return cls(
            schema_revision=PHASE_TRANSACTION_SCHEMA_REVISION,
            transaction_id=commit.transaction_id,
            commit_manifest_digest=commit.manifest_digest,
            recovered_rollout_revision=revision,
            recovery_digest=canonical_digest(payload),
        )


class FilePhaseTransactionStore:
    """Digest-addressed immutable records with atomic manifest publication."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def persist_input(self, transaction_input: PhaseTransactionInput) -> None:
        self._write_immutable(
            self._path(transaction_input.transaction_id, "input"),
            transaction_input.model_dump_json(indent=2),
        )

    def persist_pending(self, pending: PhasePendingUpdate) -> None:
        self._write_immutable(
            self._path(pending.transaction_id, "pending"),
            pending.model_dump_json(indent=2),
        )

    def publish_commit(self, manifest: PhaseCommitManifest) -> None:
        self._write_immutable(
            self._path(manifest.transaction_id, "commit"),
            manifest.model_dump_json(indent=2),
        )

    def publish_recovery(self, manifest: PhaseRecoveryManifest) -> None:
        revision_digest = canonical_digest(
            manifest.recovered_rollout_revision.model_dump(mode="json")
        )
        self._write_immutable(
            self._path(
                manifest.transaction_id,
                "recovery",
                suffix=revision_digest,
            ),
            manifest.model_dump_json(indent=2),
        )

    def load_input(self, transaction_id: str) -> PhaseTransactionInput | None:
        return self._read(self._path(transaction_id, "input"), PhaseTransactionInput)

    def load_pending(self, transaction_id: str) -> PhasePendingUpdate | None:
        return self._read(self._path(transaction_id, "pending"), PhasePendingUpdate)

    def load_commit(self, transaction_id: str) -> PhaseCommitManifest | None:
        return self._read(self._path(transaction_id, "commit"), PhaseCommitManifest)

    def _path(
        self,
        transaction_id: str,
        kind: Literal["input", "pending", "commit", "recovery"],
        *,
        suffix: str | None = None,
    ) -> Path:
        record_digest = canonical_digest(
            {
                "transaction_id": transaction_id,
                "kind": kind,
                "suffix": suffix,
            }
        )
        return self._root / f"{record_digest}.{kind}.json"

    @staticmethod
    def _read(path: Path, model: type[ModelT]) -> ModelT | None:
        if not path.exists():
            return None
        try:
            return model.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PhaseTransactionError(
                f"phase transaction record is corrupt: {path.name}"
            ) from exc

    @staticmethod
    def _write_immutable(path: Path, content: str) -> None:
        encoded = content.encode("utf-8")
        if path.exists():
            FilePhaseTransactionStore._verify_existing(path, encoded)
            return
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                FilePhaseTransactionStore._verify_existing(path, encoded)
        except OSError as exc:
            raise PhaseTransactionError(
                f"cannot publish phase transaction record: {path.name}"
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _verify_existing(path: Path, expected: bytes) -> None:
        try:
            if path.read_bytes() == expected:
                return
        except OSError as exc:
            raise PhaseTransactionError(
                f"cannot verify phase transaction record: {path.name}"
            ) from exc
        raise PhaseTransactionError(f"conflicting phase transaction record: {path.name}")


EvidenceProvider = Callable[[TrainingPhase], PhaseTransactionEvidence]


class PhaseTransactionManager:
    """Coordinator hook that persists, commits, and recovers one cycle's phases."""

    def __init__(
        self,
        *,
        store: FilePhaseTransactionStore,
        backend: TrainingBackend,
        registry: PolicyRegistry,
        context: PhaseTransactionContext,
        evidence_provider: EvidenceProvider,
    ) -> None:
        self._store = store
        self._backend = backend
        self._registry = registry
        self._context = context
        self._evidence_provider = evidence_provider
        self._prepared: dict[TrainingPhase, PhaseTransactionInput] = {}
        self._commits: list[PhaseCommitManifest] = []
        self._recoveries: list[PhaseRecoveryManifest] = []

    @property
    def commits(self) -> tuple[PhaseCommitManifest, ...]:
        return tuple(self._commits)

    @property
    def recoveries(self) -> tuple[PhaseRecoveryManifest, ...]:
        return tuple(self._recoveries)

    async def prepare(
        self,
        phase: TrainingPhase,
        target: PolicyId,
        batch: PolicyTrainingBatch,
        snapshot: tuple[tuple[PolicyId, RolloutRevision], ...],
    ) -> None:
        base_checkpoint = await self._backend.save_checkpoint(target)
        transaction_input = PhaseTransactionInput.create(
            phase=phase,
            target_policy_id=target,
            context=self._context,
            evidence=self._evidence_provider(phase),
            base_checkpoint=base_checkpoint,
            base_policy_revisions=snapshot,
            batch=batch,
        )
        self._store.persist_input(transaction_input)
        self._prepared[phase] = transaction_input

    async def record_update(
        self,
        phase: TrainingPhase,
        update: UpdateResult,
    ) -> None:
        transaction_input = self._prepared_input(phase)
        pending = PhasePendingUpdate.create(transaction_input, update)
        self._store.persist_pending(pending)

    async def commit(
        self,
        phase: TrainingPhase,
        update: UpdateResult | None,
        rollout_revision: RolloutRevision,
    ) -> None:
        transaction_input = self._prepared_input(phase)
        manifest = PhaseCommitManifest.create(
            transaction_input=transaction_input,
            update=update,
            rollout_revision=rollout_revision,
        )
        self._store.publish_commit(manifest)
        self._commits.append(manifest)

    async def recover(self, transaction_id: str) -> PhaseCommitManifest:
        transaction_input = self._store.load_input(transaction_id)
        if not isinstance(transaction_input, PhaseTransactionInput):
            raise PhaseTransactionError("phase transaction input does not exist")
        existing_commit = self._store.load_commit(transaction_id)
        if isinstance(existing_commit, PhaseCommitManifest):
            await self._restore_committed(existing_commit)
            return existing_commit

        self._prepared[transaction_input.phase] = transaction_input
        pending = self._store.load_pending(transaction_id)
        if isinstance(pending, PhasePendingUpdate):
            if pending.input_digest != transaction_input.input_digest:
                raise PhaseTransactionError("pending update belongs to another input")
            update = pending.update
            if update.policy_id != transaction_input.target_policy_id:
                raise PhaseTransactionError("pending update belongs to another policy")
            if update.base_version != transaction_input.batch.expected_base_version:
                raise PhaseTransactionError("pending update belongs to another base version")
            await self._backend.restore_checkpoint(update.checkpoint)
        elif transaction_input.empty_sub_batch:
            update = None
            await self._backend.restore_checkpoint(transaction_input.base_checkpoint)
        else:
            await self._backend.restore_checkpoint(transaction_input.base_checkpoint)
            update = await self._backend.update_policy(
                transaction_input.target_policy_id,
                transaction_input.batch,
                transaction_input.batch.expected_base_version,
            )
            await self.record_update(transaction_input.phase, update)
            # A backend may idempotently return a previously produced result after
            # its train state was rolled back to the base checkpoint. Re-loading the
            # immutable pending checkpoint makes the subsequent sync unambiguous.
            await self._backend.restore_checkpoint(update.checkpoint)

        target_version = (
            transaction_input.base_checkpoint.weight_version
            if update is None
            else update.trained_version
        )
        revision = await self._backend.sync_rollout_weights(
            transaction_input.target_policy_id,
            target_version,
        )
        await self.commit(transaction_input.phase, update, revision)
        self._registry.replace_revision(revision)
        return self._commits[-1]

    async def _restore_committed(self, manifest: PhaseCommitManifest) -> None:
        checkpoint = manifest.committed_checkpoint or manifest.base_checkpoint
        await self._backend.restore_checkpoint(checkpoint)
        revision = await self._backend.sync_rollout_weights(
            checkpoint.policy_id,
            checkpoint.weight_version,
        )
        if revision != manifest.rollout_revision:
            recovery = PhaseRecoveryManifest.create(manifest, revision)
            self._store.publish_recovery(recovery)
            self._recoveries.append(recovery)
        self._registry.replace_revision(revision)

    def _prepared_input(self, phase: TrainingPhase) -> PhaseTransactionInput:
        try:
            return self._prepared[phase]
        except KeyError:
            raise PhaseTransactionError(f"{phase} was not prepared") from None
