"""Memory-conscious single-device LoRA backend implementing the common contracts."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from heterospawn.backends.local_hf.config import LocalLoraConfig, LocalPromptEncoder
from heterospawn.domain.ids import CheckpointId, EpisodeId, PolicyId
from heterospawn.domain.training import (
    CheckpointRef,
    GenerationRequest,
    GenerationResult,
    PolicyTrainingBatch,
    UpdateResult,
    canonical_digest,
)
from heterospawn.domain.versions import RolloutRevision, WeightVersion
from heterospawn.errors import (
    CheckpointIntegrityError,
    ConfigurationError,
    RolloutRevisionMismatch,
    TrainingBatchError,
    WeightVersionMismatch,
)


@dataclass
class _LocalPolicyState:
    train_adapter: str
    rollout_adapter: str
    optimizer: Any
    weight: WeightVersion
    rollout: RolloutRevision
    checkpoint: CheckpointRef


class LocalPolicyEndpoint:
    def __init__(self, backend: LocalHfLoraBackend, policy_id: PolicyId) -> None:
        self._backend = backend
        self._policy_id = policy_id

    @property
    def policy_id(self) -> PolicyId:
        return self._policy_id

    async def current_rollout_revision(self) -> RolloutRevision:
        return self._backend.rollout_revision(self._policy_id)

    async def generate(
        self,
        request: GenerationRequest,
        expected_revision: RolloutRevision,
    ) -> GenerationResult:
        return await self._backend.generate(self._policy_id, request, expected_revision)


class LocalHfLoraBackend:
    """Reference implementation; it serializes all work through one device lock."""

    def __init__(
        self,
        *,
        config: LocalLoraConfig,
        model: Any,
        tokenizer: Any,
        policy_ids: tuple[PolicyId, ...],
    ) -> None:
        if not policy_ids or len(set(policy_ids)) != len(policy_ids):
            raise ConfigurationError("policy_ids must be non-empty and unique")
        torch, peft = _local_dependencies()
        self._torch = torch
        self._peft = peft
        self.config = config
        self._lock = asyncio.Lock()
        self._updates: dict[str, tuple[str, UpdateResult]] = {}
        self._synced: dict[tuple[PolicyId, str], RolloutRevision] = {}
        self._states: dict[PolicyId, _LocalPolicyState] = {}
        self._artifact_dir = config.artifact_dir.resolve()
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if str(config.device).startswith("cuda"):
            if not torch.cuda.is_available():
                raise ConfigurationError("CUDA device requested but torch.cuda is unavailable")
            torch.cuda.manual_seed_all(config.seed)

        self.prompt_encoder = LocalPromptEncoder(tokenizer, config)
        self._eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self._pad_token_id = getattr(tokenizer, "pad_token_id", self._eos_token_id)
        lora_config = peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
            bias="none",
        )
        first_train = _adapter_name(policy_ids[0], "train")
        self._model = peft.get_peft_model(model, lora_config, adapter_name=first_train)
        for policy_index, policy_id in enumerate(policy_ids):
            train_adapter = _adapter_name(policy_id, "train")
            rollout_adapter = _adapter_name(policy_id, "rollout")
            if policy_index != 0:
                self._model.add_adapter(train_adapter, lora_config)
            self._model.add_adapter(rollout_adapter, lora_config)
            self._copy_adapter(train_adapter, rollout_adapter)

        self._model.to(config.device)
        for policy_id in policy_ids:
            self._initialize_policy(policy_id)

    @classmethod
    def from_pretrained(
        cls,
        *,
        config: LocalLoraConfig,
        policy_ids: tuple[PolicyId, ...],
    ) -> LocalHfLoraBackend:
        torch, _ = _local_dependencies()
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as exc:  # pragma: no cover - exercised by install diagnostics
            raise ConfigurationError("install the 'local' extra to use LocalHF") from exc
        dtype = torch.float16 if config.dtype == "float16" else torch.float32
        if config.model_path is not None:
            source = config.model_path.resolve()
            weight_path = source / "model.safetensors"
            if not source.is_dir() or not weight_path.is_file():
                raise ConfigurationError("model_path must contain model.safetensors")
            if _file_sha256(weight_path) != config.expected_model_weight_sha256:
                raise CheckpointIntegrityError("local base-model weight digest mismatch")
            tokenizer = transformers.AutoTokenizer.from_pretrained(str(source))
            model = transformers.AutoModelForCausalLM.from_pretrained(
                str(source),
                torch_dtype=dtype,
                attn_implementation="eager",
            )
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                config.model_id,
                revision=config.model_revision,
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                config.model_id,
                revision=config.model_revision,
                torch_dtype=dtype,
                attn_implementation="eager",
            )
        return cls(config=config, model=model, tokenizer=tokenizer, policy_ids=policy_ids)

    def endpoint(self, policy_id: PolicyId) -> LocalPolicyEndpoint:
        self._state(policy_id)
        return LocalPolicyEndpoint(self, policy_id)

    def weight_version(self, policy_id: PolicyId) -> WeightVersion:
        return self._state(policy_id).weight

    def rollout_revision(self, policy_id: PolicyId) -> RolloutRevision:
        return self._state(policy_id).rollout

    def adapter_hash(self, policy_id: PolicyId, *, rollout: bool = False) -> str:
        state = self._state(policy_id)
        adapter = state.rollout_adapter if rollout else state.train_adapter
        return self._adapter_digest(adapter)

    async def generate(
        self,
        policy_id: PolicyId,
        request: GenerationRequest,
        expected_revision: RolloutRevision,
    ) -> GenerationResult:
        async with self._lock:
            state = self._state(policy_id)
            self._validate_generation_request(request, state, expected_revision)
            self._model.set_adapter(state.rollout_adapter)
            self._model.eval()
            prompt = self._torch.tensor(
                [request.prompt_ids], dtype=self._torch.long, device=self.config.device
            )
            if len(request.prompt_ids) >= self.config.max_sequence_length:
                raise TrainingBatchError("prompt exceeds local max_sequence_length")
            params = dict(request.sampling_params)
            max_new_tokens = _as_int(
                params.get("max_new_tokens", self.config.max_new_tokens),
                name="max_new_tokens",
            )
            max_new_tokens = min(
                max_new_tokens,
                self.config.max_sequence_length - len(request.prompt_ids),
            )
            do_sample = _as_bool(params.get("do_sample", False), name="do_sample")
            if max_new_tokens < 1:
                raise TrainingBatchError("max_new_tokens must be positive")
            generate_kwargs: dict[str, object] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "return_dict_in_generate": True,
                "output_scores": True,
                "pad_token_id": self._pad_token_id,
                "eos_token_id": self._eos_token_id,
            }
            if do_sample:
                generate_kwargs["temperature"] = _as_float(
                    params.get("temperature", 1.0), name="temperature"
                )
                generate_kwargs["top_p"] = _as_float(params.get("top_p", 1.0), name="top_p")
            else:
                generate_kwargs.update(temperature=None, top_p=None, top_k=None)
            with self._torch.inference_mode():
                output = self._model.generate(prompt, **generate_kwargs)
            response_tensor = output.sequences[0, prompt.shape[1] :]
            response_ids = tuple(int(item) for item in response_tensor.tolist())
            if not response_ids or len(output.scores) != len(response_ids):
                raise TrainingBatchError("generation did not return aligned token scores")
            log_probs = tuple(
                float(self._torch.log_softmax(score[0].float(), dim=-1)[response_ids[index]].item())
                for index, score in enumerate(output.scores)
            )
            if self._state(policy_id).rollout != expected_revision:
                raise RolloutRevisionMismatch("rollout revision changed during generation")
            stop_reason: Literal["eos", "length"] = (
                "eos"
                if self._eos_token_id is not None and response_ids[-1] == self._eos_token_id
                else "length"
            )
            return GenerationResult(
                request_id=request.request_id,
                policy_id=policy_id,
                rollout_revision=expected_revision,
                response_ids=response_ids,
                response_log_probs=log_probs,
                stop_reason=stop_reason,
                usage=(
                    ("prompt_tokens", len(request.prompt_ids)),
                    ("completion_tokens", len(response_ids)),
                ),
            )

    async def update_policy(
        self,
        policy_id: PolicyId,
        batch: PolicyTrainingBatch,
        expected_base_version: WeightVersion,
    ) -> UpdateResult:
        async with self._lock:
            prior = self._updates.get(batch.batch_id)
            if prior is not None:
                prior_digest, result = prior
                if prior_digest != batch.batch_digest:
                    raise TrainingBatchError("batch_id was already used with another digest")
                return result
            state = self._state(policy_id)
            if not batch.samples:
                raise TrainingBatchError("empty batches must be skipped")
            if batch.target_policy_id != policy_id:
                raise TrainingBatchError("batch targets another policy")
            if batch.expected_base_version != expected_base_version:
                raise WeightVersionMismatch("call and batch base versions differ")
            if state.weight != expected_base_version:
                raise WeightVersionMismatch("local train adapter is not at expected base version")
            self._model.set_adapter(state.train_adapter)
            self._model.train()
            state.optimizer.zero_grad(set_to_none=True)

            episode_losses: dict[EpisodeId, Any] = {}
            episode_weights: dict[EpisodeId, float] = {}
            ratios: list[float] = []
            kls: list[float] = []
            entropies: list[float] = []
            for sample in batch.samples:
                new_log_probs, entropy = self._sample_log_probs(
                    sample.prompt_ids, sample.response_ids
                )
                mask = self._torch.tensor(
                    sample.loss_mask,
                    dtype=new_log_probs.dtype,
                    device=self.config.device,
                )
                active = mask.sum()
                sequence_mean = (new_log_probs * mask).sum() / active
                sequence_loss = -sample.advantage * sequence_mean * sample.aggregation_weight
                episode_losses[sample.episode_id] = (
                    episode_losses.get(sample.episode_id, 0.0) + sequence_loss
                )
                episode_weights[sample.episode_id] = (
                    episode_weights.get(sample.episode_id, 0.0) + sample.aggregation_weight
                )
                old = self._torch.tensor(
                    sample.old_log_probs,
                    dtype=new_log_probs.dtype,
                    device=self.config.device,
                )
                active_ratios = self._torch.exp(new_log_probs.detach() - old)[mask.bool()]
                ratios.extend(float(item) for item in active_ratios.cpu().tolist())
                kls.extend(
                    float(item) for item in (old - new_log_probs.detach())[mask.bool()].cpu()
                )
                entropies.append(float(entropy.detach().cpu()))
            if any(abs(weight - 1.0) > 1e-6 for weight in episode_weights.values()):
                raise TrainingBatchError("aggregation weights must sum to one per episode")
            loss = self._torch.stack(tuple(episode_losses.values())).mean()
            loss.backward()
            grad_norm = self._gradient_norm(state.train_adapter)
            state.optimizer.step()
            base_version = state.weight
            checkpoint = self._save_checkpoint(policy_id, base_version.optimizer_step + 1)
            state.weight = checkpoint.weight_version
            state.checkpoint = checkpoint
            result = UpdateResult(
                policy_id=policy_id,
                base_version=base_version,
                trained_version=state.weight,
                checkpoint=checkpoint,
                metrics=(
                    ("loss", float(loss.detach().cpu())),
                    ("gradient_norm", grad_norm),
                    ("old_new_ratio_mean", sum(ratios) / len(ratios)),
                    ("approx_kl_mean", sum(kls) / len(kls)),
                    ("entropy_mean", sum(entropies) / len(entropies)),
                    ("episode_count", float(len(episode_losses))),
                ),
            )
            self._updates[batch.batch_id] = (batch.batch_digest, result)
            return result

    async def sync_rollout_weights(
        self,
        policy_id: PolicyId,
        trained_version: WeightVersion,
    ) -> RolloutRevision:
        async with self._lock:
            state = self._state(policy_id)
            if state.weight != trained_version:
                raise WeightVersionMismatch("cannot sync unknown or stale training weights")
            key = (policy_id, trained_version.checkpoint_digest)
            existing = self._synced.get(key)
            if existing is not None:
                return existing
            if state.rollout.weight_version == trained_version:
                return state.rollout
            self._copy_adapter(state.train_adapter, state.rollout_adapter)
            if self._adapter_digest(state.train_adapter) != self._adapter_digest(
                state.rollout_adapter
            ):
                raise CheckpointIntegrityError("rollout adapter differs after sync")
            revision = RolloutRevision(
                policy_id=policy_id,
                weight_version=trained_version,
                deployment_id=state.rollout.deployment_id,
                replica_set_revision=state.rollout.replica_set_revision + 1,
            )
            state.rollout = revision
            self._synced[key] = revision
            return revision

    async def save_checkpoint(self, policy_id: PolicyId) -> CheckpointRef:
        async with self._lock:
            return self._state(policy_id).checkpoint

    async def restore_checkpoint(self, checkpoint: CheckpointRef) -> WeightVersion:
        async with self._lock:
            path = _path_from_uri(checkpoint.uri)
            manifest = self._verified_manifest(path, checkpoint)
            try:
                safetensors_torch = importlib.import_module("safetensors.torch")
            except ImportError as exc:  # pragma: no cover
                raise ConfigurationError("safetensors is required for restore") from exc
            state = self._state(checkpoint.policy_id)
            adapter_state = safetensors_torch.load_file(
                str(path / "adapter.safetensors"), device="cpu"
            )
            self._peft.set_peft_model_state_dict(
                self._model,
                adapter_state,
                adapter_name=state.train_adapter,
            )
            optimizer_state = self._torch.load(
                path / "optimizer.pt", map_location=self.config.device, weights_only=True
            )
            state.optimizer.load_state_dict(optimizer_state)
            rng_state = self._torch.load(path / "rng.pt", map_location="cpu", weights_only=True)
            self._torch.set_rng_state(rng_state["cpu"])
            if str(self.config.device).startswith("cuda") and rng_state["cuda"]:
                self._torch.cuda.set_rng_state_all(rng_state["cuda"])
            random.setstate(
                _nested_tuple(json.loads((path / "python_random.json").read_text(encoding="utf-8")))
            )
            state.weight = checkpoint.weight_version
            state.checkpoint = checkpoint
            if manifest["adapter_digest"] != self._adapter_digest(state.train_adapter):
                raise CheckpointIntegrityError("restored adapter hash differs from manifest")
            return state.weight

    def _initialize_policy(self, policy_id: PolicyId) -> None:
        train_adapter = _adapter_name(policy_id, "train")
        rollout_adapter = _adapter_name(policy_id, "rollout")
        parameters = [
            parameter
            for name, parameter in self._model.named_parameters()
            if f".{train_adapter}." in name
        ]
        if not parameters:
            raise ConfigurationError(f"LoRA adapter has no parameters: {train_adapter}")
        optimizer = self._torch.optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=0.0,
        )
        placeholder = WeightVersion(
            policy_id=policy_id,
            optimizer_step=0,
            checkpoint_digest="pending",
        )
        placeholder_checkpoint = CheckpointRef(
            checkpoint_id=CheckpointId(f"{policy_id}:pending"),
            policy_id=policy_id,
            weight_version=placeholder,
            uri="memory://pending",
            optimizer_state_digest="pending",
        )
        placeholder_rollout = RolloutRevision(
            policy_id=policy_id,
            weight_version=placeholder,
            deployment_id=f"local-hf:{self.config.device}:{policy_id}",
            replica_set_revision=0,
        )
        self._states[policy_id] = _LocalPolicyState(
            train_adapter=train_adapter,
            rollout_adapter=rollout_adapter,
            optimizer=optimizer,
            weight=placeholder,
            rollout=placeholder_rollout,
            checkpoint=placeholder_checkpoint,
        )
        checkpoint = self._save_checkpoint(policy_id, 0)
        state = self._states[policy_id]
        state.weight = checkpoint.weight_version
        state.checkpoint = checkpoint
        state.rollout = state.rollout.model_copy(update={"weight_version": state.weight})

    def _validate_generation_request(
        self,
        request: GenerationRequest,
        state: _LocalPolicyState,
        expected_revision: RolloutRevision,
    ) -> None:
        if state.rollout != expected_revision:
            raise RolloutRevisionMismatch("local rollout revision mismatch")
        if request.tokenizer_revision != self.prompt_encoder.tokenizer_revision:
            raise TrainingBatchError("tokenizer revision mismatch")
        if request.prompt_template_revision != self.prompt_encoder.prompt_template_revision:
            raise TrainingBatchError("prompt-template revision mismatch")

    def _sample_log_probs(
        self,
        prompt_ids: tuple[int, ...],
        response_ids: tuple[int, ...],
    ) -> tuple[Any, Any]:
        combined = prompt_ids + response_ids
        if len(combined) > self.config.max_sequence_length:
            raise TrainingBatchError("training sample exceeds local max_sequence_length")
        tokens = self._torch.tensor([combined], dtype=self._torch.long, device=self.config.device)
        output = self._model(input_ids=tokens)
        start = len(prompt_ids) - 1
        response_logits = output.logits[0, start : start + len(response_ids), :].float()
        distributions = self._torch.log_softmax(response_logits, dim=-1)
        targets = self._torch.tensor(
            response_ids, dtype=self._torch.long, device=self.config.device
        )
        log_probs = distributions.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        entropy = -(distributions.exp() * distributions).sum(dim=-1).mean()
        return log_probs, entropy

    def _gradient_norm(self, adapter: str) -> float:
        squared = 0.0
        for name, parameter in self._model.named_parameters():
            if f".{adapter}." in name and parameter.grad is not None:
                squared += float(parameter.grad.detach().float().norm().cpu()) ** 2
        return float(squared**0.5)

    def _copy_adapter(self, source: str, target: str) -> None:
        state = self._peft.get_peft_model_state_dict(self._model, adapter_name=source)
        self._peft.set_peft_model_state_dict(self._model, state, adapter_name=target)

    def _adapter_state(self, adapter: str) -> dict[str, Any]:
        state = self._peft.get_peft_model_state_dict(self._model, adapter_name=adapter)
        return {key: value.detach().cpu().contiguous() for key, value in state.items()}

    def _adapter_digest(self, adapter: str) -> str:
        digest = hashlib.sha256()
        for key, tensor in sorted(self._adapter_state(adapter).items()):
            digest.update(key.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(bytes(tensor.view(self._torch.uint8).flatten().tolist()))
        return digest.hexdigest()

    def _save_checkpoint(self, policy_id: PolicyId, optimizer_step: int) -> CheckpointRef:
        try:
            safetensors_torch = importlib.import_module("safetensors.torch")
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError("safetensors is required for checkpoints") from exc
        state = self._state(policy_id)
        temporary = Path(tempfile.mkdtemp(prefix="pending-", dir=self._artifact_dir))
        try:
            adapter_state = self._adapter_state(state.train_adapter)
            safetensors_torch.save_file(adapter_state, str(temporary / "adapter.safetensors"))
            self._torch.save(state.optimizer.state_dict(), temporary / "optimizer.pt")
            rng_state = {
                "cpu": self._torch.get_rng_state(),
                "cuda": self._torch.cuda.get_rng_state_all()
                if str(self.config.device).startswith("cuda")
                else [],
            }
            self._torch.save(rng_state, temporary / "rng.pt")
            (temporary / "python_random.json").write_text(
                json.dumps(random.getstate(), separators=(",", ":")),
                encoding="utf-8",
            )
            file_digests = {
                name: _file_sha256(temporary / name)
                for name in (
                    "adapter.safetensors",
                    "optimizer.pt",
                    "rng.pt",
                    "python_random.json",
                )
            }
            adapter_digest = self._adapter_digest(state.train_adapter)
            manifest_payload = {
                "schema_version": 1,
                "policy_id": policy_id,
                "optimizer_step": optimizer_step,
                "base_model_id": self.config.model_id,
                "base_model_revision": self.config.model_revision,
                "base_model_weight_sha256": self.config.expected_model_weight_sha256,
                "tokenizer_revision": self.prompt_encoder.tokenizer_revision,
                "prompt_template_revision": self.prompt_encoder.prompt_template_revision,
                "config": self.config.model_dump(
                    mode="json", exclude={"artifact_dir", "model_path"}
                ),
                "adapter_digest": adapter_digest,
                "file_digests": file_digests,
            }
            checkpoint_digest = canonical_digest(manifest_payload)
            weight = WeightVersion(
                policy_id=policy_id,
                optimizer_step=optimizer_step,
                checkpoint_digest=checkpoint_digest,
            )
            checkpoint_id = CheckpointId(
                f"{policy_id}:step-{optimizer_step}:{checkpoint_digest[:12]}"
            )
            manifest = {**manifest_payload, "checkpoint_digest": checkpoint_digest}
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            destination = self._artifact_dir / str(checkpoint_id).replace(":", "_")
            if destination.exists():
                shutil.rmtree(temporary)
            else:
                temporary.replace(destination)
            return CheckpointRef(
                checkpoint_id=checkpoint_id,
                policy_id=policy_id,
                weight_version=weight,
                uri=destination.as_uri(),
                optimizer_state_digest=file_digests["optimizer.pt"],
            )
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    def _verified_manifest(self, path: Path, checkpoint: CheckpointRef) -> dict[str, Any]:
        try:
            manifest: dict[str, Any] = json.loads(
                (path / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointIntegrityError("checkpoint manifest is unreadable") from exc
        payload = {key: value for key, value in manifest.items() if key != "checkpoint_digest"}
        digest = canonical_digest(payload)
        if digest != checkpoint.weight_version.checkpoint_digest:
            raise CheckpointIntegrityError("checkpoint manifest digest mismatch")
        if manifest.get("policy_id") != checkpoint.policy_id:
            raise CheckpointIntegrityError("checkpoint policy does not match manifest")
        if manifest.get("optimizer_step") != checkpoint.weight_version.optimizer_step:
            raise CheckpointIntegrityError("checkpoint optimizer step does not match manifest")
        if (
            manifest.get("base_model_id") != self.config.model_id
            or manifest.get("base_model_revision") != self.config.model_revision
        ):
            raise CheckpointIntegrityError("checkpoint base model does not match runtime")
        if manifest["file_digests"].get("optimizer.pt") != checkpoint.optimizer_state_digest:
            raise CheckpointIntegrityError("checkpoint optimizer digest does not match reference")
        for name, expected in manifest["file_digests"].items():
            if _file_sha256(path / name) != expected:
                raise CheckpointIntegrityError(f"checkpoint file digest mismatch: {name}")
        return manifest

    def _state(self, policy_id: PolicyId) -> _LocalPolicyState:
        try:
            return self._states[policy_id]
        except KeyError:
            raise TrainingBatchError(f"unknown local policy: {policy_id}") from None


def _local_dependencies() -> tuple[Any, Any]:
    try:
        peft = importlib.import_module("peft")
        torch = importlib.import_module("torch")
    except (ImportError, RuntimeError) as exc:
        raise ConfigurationError(
            "LocalHF dependencies are unavailable; use an isolated environment with the local extra"
        ) from exc
    return torch, peft


def _adapter_name(policy_id: PolicyId, kind: str) -> str:
    safe_policy = "".join(character if character.isalnum() else "_" for character in policy_id)
    return f"{safe_policy}_{kind}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_from_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise CheckpointIntegrityError("local checkpoint URI must use file scheme")
    return Path(url2pathname(unquote(parsed.path))).resolve()


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _as_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingBatchError(f"{name} must be an integer")
    return value


def _as_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingBatchError(f"{name} must be numeric")
    return float(value)


def _as_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TrainingBatchError(f"{name} must be a boolean")
    return value
