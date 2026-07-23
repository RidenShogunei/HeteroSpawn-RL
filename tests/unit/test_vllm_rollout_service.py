from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path

import pytest

from heterospawn.backends.vllm_rollout import (
    VllmPolicyDeployment,
    VllmRolloutConfig,
    VllmRolloutService,
    VllmSamplingConfig,
    VllmWorkerResult,
    VllmWorkerRuntime,
    VllmWorkerSpec,
)
from heterospawn.backends.vllm_rollout.models import (
    rollout_artifact_digest,
    rollout_artifact_path,
)
from heterospawn.backends.vllm_rollout.process import _worker_environment
from heterospawn.backends.vllm_rollout.service import _sampling_config
from heterospawn.backends.vllm_rollout.worker import _selected_log_probs
from heterospawn.domain.ids import AgentInstanceId, EpisodeId, PolicyId, RolloutId, TaskId
from heterospawn.domain.training import GenerationRequest, RolloutArtifact
from heterospawn.domain.versions import WeightVersion
from heterospawn.errors import (
    CheckpointIntegrityError,
    RolloutRevisionMismatch,
    RolloutServiceError,
    TrainingBatchError,
)


class FakeWorker:
    def __init__(
        self,
        spec: VllmWorkerSpec,
        *,
        started: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
        digest_override: str | None = None,
    ) -> None:
        self.spec = spec
        self.closed = False
        self.generate_count = 0
        self.started = started
        self.release = release
        self._adapter_digest = digest_override or spec.artifact.artifact_digest

    @property
    def adapter_digest(self) -> str:
        return self._adapter_digest

    async def generate(
        self,
        prompt_ids: tuple[int, ...],
        sampling: VllmSamplingConfig,
    ) -> VllmWorkerResult:
        assert not self.closed
        assert sampling.max_new_tokens >= 1
        self.generate_count += 1
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        return VllmWorkerResult(
            prompt_ids=prompt_ids,
            response_ids=(31, 2),
            response_log_probs=(-0.25, -0.5),
            stop_reason="eos",
            adapter_digest=self.adapter_digest,
        )

    async def close(self) -> None:
        self.closed = True

    async def runtime_metrics(self) -> VllmWorkerRuntime:
        return VllmWorkerRuntime(
            gpu_name="fake-gpu",
            total_memory_bytes=1024,
            device_used_bytes_at_query=512,
            peak_allocated_bytes=256,
            peak_reserved_bytes=384,
            versions=(("vllm", "fake"),),
        )


class FakeWorkerFactory:
    def __init__(self) -> None:
        self.created: list[FakeWorker] = []
        self.fail_once: set[str] = set()
        self.wrong_digest_once: set[str] = set()
        self.block_next = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def create(self, spec: VllmWorkerSpec) -> FakeWorker:
        digest = spec.artifact.artifact_digest
        if digest in self.fail_once:
            self.fail_once.remove(digest)
            raise RuntimeError("injected worker launch failure")
        digest_override = None
        if digest in self.wrong_digest_once:
            self.wrong_digest_once.remove(digest)
            digest_override = "wrong-adapter-digest"
        worker = FakeWorker(
            spec,
            started=self.started if self.block_next else None,
            release=self.release if self.block_next else None,
            digest_override=digest_override,
        )
        self.block_next = False
        self.created.append(worker)
        return worker


class _LogProb:
    def __init__(self, value: float) -> None:
        self.logprob = value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path) -> VllmRolloutConfig:
    model_path = tmp_path / "model"
    model_path.mkdir(exist_ok=True)
    model_weight = model_path / "model.safetensors"
    model_weight.write_bytes(b"base-model-fixture")
    return VllmRolloutConfig(
        python_executable=Path(sys.executable),
        model_path=model_path,
        expected_model_weight_sha256=_sha256(model_weight),
        model_revision="model@fixture",
        tokenizer_revision="tokenizer@fixture",
        prompt_template_revision="template@fixture",
        runtime_dir=tmp_path / "runtime",
    )


def _weight(policy_id: PolicyId, step: int) -> WeightVersion:
    return WeightVersion(
        policy_id=policy_id,
        optimizer_step=step,
        checkpoint_digest=f"checkpoint-{step}",
    )


def _artifact(
    tmp_path: Path,
    policy_id: PolicyId,
    step: int,
    *,
    suffix: str = "",
) -> RolloutArtifact:
    path = tmp_path / f"adapter-{step}{suffix}"
    path.mkdir()
    (path / "adapter_config.json").write_text(
        '{"peft_type":"LORA","r":8}',
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(f"adapter-{step}{suffix}".encode())
    return RolloutArtifact(
        policy_id=policy_id,
        weight_version=_weight(policy_id, step),
        uri=path.as_uri(),
        artifact_digest=rollout_artifact_digest(path),
        format_revision="peft-lora-v1",
    )


def _request(index: int = 0) -> GenerationRequest:
    return GenerationRequest(
        task_id=TaskId("task"),
        episode_id=EpisodeId(f"episode-{index}"),
        rollout_id=RolloutId(f"rollout-{index}"),
        request_id=f"request-{index}",
        agent_role="sub",
        agent_instance_id=AgentInstanceId(f"sub-{index}"),
        prompt_ids=(1, 5, 6),
        tokenizer_revision="tokenizer@fixture",
        prompt_template_revision="template@fixture",
        sampling_params=(("max_new_tokens", 2), ("do_sample", False)),
    )


async def _service(
    tmp_path: Path,
    factory: FakeWorkerFactory,
) -> tuple[VllmRolloutService, PolicyId, RolloutArtifact]:
    policy_id = PolicyId("sub")
    artifact = _artifact(tmp_path, policy_id, 0)
    service = await VllmRolloutService.start(
        config=_config(tmp_path),
        deployments=(
            (
                VllmPolicyDeployment(
                    policy_id=policy_id,
                    cuda_device="0",
                    deployment_id="sub-rollout",
                ),
                artifact,
            ),
        ),
        worker_factory=factory,
    )
    return service, policy_id, artifact


@pytest.mark.asyncio
async def test_exact_generation_and_four_requests_share_one_worker(tmp_path: Path) -> None:
    factory = FakeWorkerFactory()
    service, policy_id, _ = await _service(tmp_path, factory)
    revision = service.rollout_revision(policy_id)

    results = await asyncio.gather(
        *(service.endpoint(policy_id).generate(_request(index), revision) for index in range(4))
    )

    assert len(factory.created) == 1
    assert factory.created[0].generate_count == 4
    assert [result.response_ids for result in results] == [(31, 2)] * 4
    assert [result.response_log_probs for result in results] == [(-0.25, -0.5)] * 4
    assert all(result.rollout_revision == revision for result in results)
    assert (await service.runtime_metrics(policy_id)).gpu_name == "fake-gpu"
    await service.close()
    await service.close()
    with pytest.raises(RolloutServiceError, match="closed"):
        await service.endpoint(policy_id).generate(_request(), revision)


@pytest.mark.asyncio
async def test_generation_rejects_stale_revision_and_provenance_mismatch(
    tmp_path: Path,
) -> None:
    factory = FakeWorkerFactory()
    service, policy_id, _ = await _service(tmp_path, factory)
    revision = service.rollout_revision(policy_id)

    with pytest.raises(RolloutRevisionMismatch):
        await service.endpoint(policy_id).generate(
            _request(),
            revision.model_copy(update={"replica_set_revision": 9}),
        )
    with pytest.raises(TrainingBatchError, match="tokenizer"):
        await service.endpoint(policy_id).generate(
            _request().model_copy(update={"tokenizer_revision": "wrong"}),
            revision,
        )
    await service.close()


@pytest.mark.asyncio
async def test_sync_publishes_only_after_verified_worker_and_is_idempotent(
    tmp_path: Path,
) -> None:
    factory = FakeWorkerFactory()
    service, policy_id, _ = await _service(tmp_path, factory)
    old_revision = service.rollout_revision(policy_id)
    replacement = _artifact(tmp_path, policy_id, 1)

    new_revision = await service.sync_rollout_weights(
        policy_id,
        replacement.weight_version,
        replacement,
    )

    assert factory.created[0].closed
    assert new_revision.weight_version == replacement.weight_version
    assert new_revision.replica_set_revision == old_revision.replica_set_revision + 1
    assert (
        await service.sync_rollout_weights(
            policy_id,
            replacement.weight_version,
            replacement,
        )
        == new_revision
    )
    assert len(factory.created) == 2
    with pytest.raises(RolloutRevisionMismatch):
        await service.endpoint(policy_id).generate(_request(), old_revision)
    await service.endpoint(policy_id).generate(_request(), new_revision)
    await service.close()


@pytest.mark.asyncio
async def test_failed_replacement_rolls_back_without_publishing_revision(
    tmp_path: Path,
) -> None:
    factory = FakeWorkerFactory()
    service, policy_id, _ = await _service(tmp_path, factory)
    old_revision = service.rollout_revision(policy_id)
    replacement = _artifact(tmp_path, policy_id, 1)
    factory.fail_once.add(replacement.artifact_digest)

    with pytest.raises(RolloutServiceError, match="prior rollout revision was restored"):
        await service.sync_rollout_weights(
            policy_id,
            replacement.weight_version,
            replacement,
        )

    assert service.rollout_revision(policy_id) == old_revision
    assert len(factory.created) == 2
    assert factory.created[-1].adapter_digest != replacement.artifact_digest
    await service.endpoint(policy_id).generate(_request(), old_revision)
    await service.close()


@pytest.mark.asyncio
async def test_sync_waits_for_active_generation_and_rejects_new_work(
    tmp_path: Path,
) -> None:
    factory = FakeWorkerFactory()
    factory.block_next = True
    service, policy_id, _ = await _service(tmp_path, factory)
    old_revision = service.rollout_revision(policy_id)
    replacement = _artifact(tmp_path, policy_id, 1)
    generation = asyncio.create_task(service.endpoint(policy_id).generate(_request(), old_revision))
    await factory.started.wait()

    synchronization = asyncio.create_task(
        service.sync_rollout_weights(
            policy_id,
            replacement.weight_version,
            replacement,
        )
    )
    await asyncio.sleep(0)
    with pytest.raises(RolloutServiceError, match="transitioning"):
        await service.endpoint(policy_id).generate(_request(1), old_revision)
    assert not factory.created[0].closed

    factory.release.set()
    await generation
    new_revision = await synchronization
    assert factory.created[0].closed
    assert service.rollout_revision(policy_id) == new_revision
    await service.close()


@pytest.mark.asyncio
async def test_same_weight_with_different_artifact_and_corruption_are_rejected(
    tmp_path: Path,
) -> None:
    factory = FakeWorkerFactory()
    service, policy_id, current = await _service(tmp_path, factory)
    conflicting = _artifact(tmp_path, policy_id, 0, suffix="-other")

    with pytest.raises(CheckpointIntegrityError, match="same weight version"):
        await service.sync_rollout_weights(
            policy_id,
            conflicting.weight_version,
            conflicting,
        )

    artifact_path = rollout_artifact_path(current)
    (artifact_path / "adapter_model.safetensors").write_bytes(b"corrupted")
    with pytest.raises(CheckpointIntegrityError, match="digest mismatch"):
        await service.sync_rollout_weights(
            policy_id,
            current.weight_version,
            current,
        )
    await service.close()


def test_worker_protocol_extracts_selected_logprobs_and_filters_credentials() -> None:
    assert _selected_log_probs(
        (7, 8),
        [{7: _LogProb(-0.1)}, {8: _LogProb(-0.2)}],
    ) == (-0.1, -0.2)
    with pytest.raises(RuntimeError, match="absent"):
        _selected_log_probs((7,), [{8: _LogProb(-0.2)}])

    isolated_home = Path("/isolated/worker-home")
    environment = _worker_environment(
        {
            "PATH": "/usr/bin",
            "HOME": "/home/test",
            "HF_TOKEN": "secret",
            "MINIMAX_API_KEY": "secret",
        },
        "3",
        home=isolated_home,
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert environment["VLLM_USE_V1"] == "0"
    assert environment["HOME"] == str(isolated_home)
    assert "HF_TOKEN" not in environment
    assert "MINIMAX_API_KEY" not in environment


def test_sampling_config_accepts_auditable_guided_regex(tmp_path: Path) -> None:
    request = _request().model_copy(
        update={
            "sampling_params": (
                ("max_new_tokens", 8),
                ("do_sample", False),
                ("guided_regex", r'\{"kind":"answer","answer":"Paris"\}'),
            )
        }
    )

    sampling = _sampling_config(request, _config(tmp_path))

    assert sampling.guided_regex == r'\{"kind":"answer","answer":"Paris"\}'
