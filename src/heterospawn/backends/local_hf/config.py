"""Configuration and prompt identity for the local LoRA contract backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from heterospawn.domain.training import PromptEncoding, canonical_digest
from heterospawn.policies.base import Message
from heterospawn.policies.trainable import ToolDefinition

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEFAULT_MODEL_WEIGHT_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"


class LocalLoraConfig(BaseModel):
    """Small deterministic defaults suitable for an 8 GB development GPU."""

    model_config = ConfigDict(frozen=True, strict=True)

    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = DEFAULT_MODEL_REVISION
    model_path: Path | None = None
    expected_model_weight_sha256: str = DEFAULT_MODEL_WEIGHT_SHA256
    device: str = "cuda:0"
    dtype: Literal["float16", "float32"] = "float16"
    max_sequence_length: int = Field(default=1024, ge=16)
    max_new_tokens: int = Field(default=32, ge=1)
    lora_rank: int = Field(default=8, ge=1)
    lora_alpha: int = Field(default=16, ge=1)
    lora_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    learning_rate: float = Field(default=1e-4, gt=0)
    artifact_dir: Path = Path("artifacts/local-contract/checkpoints")
    seed: int = 20260722


class LocalPromptEncoder:
    """Applies the pinned chat template once; training never calls this encoder."""

    def __init__(self, tokenizer: Any, config: LocalLoraConfig) -> None:
        self._tokenizer = tokenizer
        tokenizer_identity = {
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "tokenizer_class": type(tokenizer).__name__,
            "special_tokens": getattr(tokenizer, "special_tokens_map", {}),
            "vocab": sorted(tokenizer.get_vocab().items()),
        }
        self.tokenizer_revision = canonical_digest(tokenizer_identity)
        self.prompt_template_revision = canonical_digest(
            {
                "model_revision": config.model_revision,
                "chat_template": getattr(tokenizer, "chat_template", None),
            }
        )
        self._issued_prompt_template_revisions = {self.prompt_template_revision}

    def encode(
        self,
        messages: tuple[Message, ...],
        tools: tuple[ToolDefinition, ...] = (),
    ) -> PromptEncoding:
        payload = [message.model_dump(mode="json") for message in messages]
        tool_payload = [tool.as_chat_template_tool() for tool in tools]
        template_args: dict[str, object] = {
            "tokenize": True,
            "add_generation_prompt": True,
        }
        if tool_payload:
            template_args["tools"] = tool_payload
        prompt_ids = self._tokenizer.apply_chat_template(payload, **template_args)
        prompt_template_revision = (
            canonical_digest(
                {
                    "base_revision": self.prompt_template_revision,
                    "tools": tool_payload,
                }
            )
            if tool_payload
            else self.prompt_template_revision
        )
        self._issued_prompt_template_revisions.add(prompt_template_revision)
        return PromptEncoding(
            prompt_ids=tuple(int(token_id) for token_id in prompt_ids),
            tokenizer_revision=self.tokenizer_revision,
            prompt_template_revision=prompt_template_revision,
        )

    def accepts_prompt_template_revision(self, revision: str) -> bool:
        """Return whether this encoder issued the revision for an encoded prompt."""

        return revision in self._issued_prompt_template_revisions

    def decode(self, response_ids: tuple[int, ...]) -> str:
        """Decode only for environment/action interpretation, never for training reconstruction."""

        return str(
            self._tokenizer.decode(
                list(response_ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
