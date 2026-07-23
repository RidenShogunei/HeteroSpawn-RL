"""Provider-neutral research task contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from heterospawn.domain.ids import TaskId
from heterospawn.domain.training import JsonScalar


class ResearchTask(BaseModel):
    """Policy-visible task data with no evaluator reference answer."""

    model_config = ConfigDict(frozen=True, strict=True)

    task_id: TaskId
    prompt: str = Field(min_length=1)
    dataset_revision: str = "unspecified"
    answer_format: Literal["plain", "boxed", "markdown_table"] = "plain"
    language: Literal["en", "zh"] = "en"
    metadata: tuple[tuple[str, JsonScalar], ...] = ()
