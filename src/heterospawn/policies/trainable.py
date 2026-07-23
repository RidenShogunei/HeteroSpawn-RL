"""Backend-neutral codec contract for trainable token-level policy services."""

from __future__ import annotations

from typing import Protocol

from heterospawn.domain.training import PromptEncoding
from heterospawn.policies.base import Message


class TrainablePolicyCodec(Protocol):
    """Converts environment messages to model input and output to environment text."""

    def encode(self, messages: tuple[Message, ...]) -> PromptEncoding: ...

    def decode(self, response_ids: tuple[int, ...]) -> str: ...
