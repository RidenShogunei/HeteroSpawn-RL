"""Backend-neutral codec contract for trainable token-level policy services."""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from heterospawn.domain.training import PromptEncoding
from heterospawn.policies.base import Message


class ToolDefinition(BaseModel):
    """Immutable function-tool schema accepted by a trainable chat template."""

    model_config = ConfigDict(frozen=True, strict=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters_json: str = Field(min_length=2)

    def as_chat_template_tool(self) -> dict[str, object]:
        parameters = json.loads(self.parameters_json)
        if not isinstance(parameters, dict):
            raise ValueError("tool parameters must decode to a JSON object")
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


class TrainablePolicyCodec(Protocol):
    """Converts environment messages to model input and output to environment text."""

    def encode(
        self,
        messages: tuple[Message, ...],
        tools: tuple[ToolDefinition, ...] = (),
    ) -> PromptEncoding: ...

    def decode(self, response_ids: tuple[int, ...]) -> str: ...
