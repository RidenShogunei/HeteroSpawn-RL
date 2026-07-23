from __future__ import annotations

import pytest
from pydantic import ValidationError

from heterospawn.benchmarks.xbench import BenchmarkTask
from heterospawn.domain.ids import TaskId
from heterospawn.domain.tasks import ResearchTask
from heterospawn.search.base import AccessRequest, AccessResponse


def test_xbench_task_is_provider_neutral_research_task() -> None:
    task = BenchmarkTask(task_id=TaskId("task-1"), prompt="question")

    assert isinstance(task, ResearchTask)
    assert task.dataset_revision == "unspecified"
    assert "answer" not in task.model_dump()


def test_research_task_retains_format_and_revision_without_reference_answer() -> None:
    task = ResearchTask(
        task_id=TaskId("wide-1"),
        prompt="Return a table.",
        dataset_revision="dataset-revision",
        answer_format="markdown_table",
        language="en",
        metadata=(("source", "width"),),
    )

    assert task.answer_format == "markdown_table"
    assert dict(task.metadata) == {"source": "width"}


def test_access_contract_is_strict_and_revision_complete() -> None:
    request = AccessRequest(
        request_id="access-1",
        url="https://example.invalid/page",
        info_to_extract="capital",
    )
    response = AccessResponse(
        request_id=request.request_id,
        provider="fixture",
        provider_revision="fixture-v1",
        provider_request_id="provider-1",
        url=request.url,
        content="page",
        truncated=False,
        raw_response_digest="digest",
    )

    assert response.url == request.url
    with pytest.raises(ValidationError):
        AccessRequest(
            request_id="access-2",
            url=request.url,
            info_to_extract="capital",
            max_characters=0,
        )
