from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from heterospawn.benchmarks.xbench import BenchmarkTask, load_xbench, parse_final_answer
from heterospawn.domain.ids import TaskId
from heterospawn.errors import BenchmarkDataError

FixtureFactory = Callable[[tuple[tuple[str, str, str], ...]], Path]


def _one_task_fixture(factory: FixtureFactory) -> Path:
    return factory((("synthetic-1", "policy-visible question", "42"),))


def test_encrypted_fixture_is_decrypted_in_memory_and_answer_stays_evaluator_only(
    xbench_fixture_factory: FixtureFactory,
) -> None:
    fixture = _one_task_fixture(xbench_fixture_factory)

    dataset = load_xbench(fixture, verify_official_digest=False)

    assert dataset.tasks[0].prompt == "policy-visible question"
    assert "answer" not in BenchmarkTask.model_fields
    score = dataset.evaluate_exact({TaskId("synthetic-1"): "reasoning\n最终答案:42"})
    assert score.correct == 1
    assert score.accuracy == 1.0
    assert score.comparable_to_official is False


def test_official_digest_is_required_by_default(
    xbench_fixture_factory: FixtureFactory,
) -> None:
    fixture = _one_task_fixture(xbench_fixture_factory)

    with pytest.raises(BenchmarkDataError, match="digest"):
        load_xbench(fixture)


def test_score_scope_rejects_unknown_task_ids(
    xbench_fixture_factory: FixtureFactory,
) -> None:
    fixture = _one_task_fixture(xbench_fixture_factory)
    dataset = load_xbench(fixture, verify_official_digest=False)

    with pytest.raises(BenchmarkDataError, match="unknown task id"):
        dataset.evaluate_exact({}, task_ids=(TaskId("missing"),))


def test_repeat_selection_and_exact_metrics_are_deterministic(
    xbench_fixture_factory: FixtureFactory,
) -> None:
    fixture = xbench_fixture_factory(
        (
            ("synthetic-1", "first private prompt", "42"),
            ("synthetic-2", "second private prompt", "alpha"),
        )
    )
    dataset = load_xbench(fixture, verify_official_digest=False)

    selected = dataset.select_tasks((TaskId("synthetic-2"), TaskId("synthetic-1")))
    report = dataset.evaluate_repeat_exact(
        {
            TaskId("synthetic-1"): ("最终答案:42", None),
            TaskId("synthetic-2"): ("最终答案:wrong", "no marker"),
        },
        task_ids=(TaskId("synthetic-1"), TaskId("synthetic-2")),
        repeats_per_task=2,
    )

    assert [task.task_id for task in selected] == ["synthetic-2", "synthetic-1"]
    assert report.completed_episodes == 3
    assert report.exact_correct_episodes == 1
    assert report.average_exact_accuracy == 0.25
    assert report.best_of_n_correct_tasks == 1
    assert report.best_of_n_exact_accuracy == 0.5
    assert report.comparable_to_official is False

    with pytest.raises(BenchmarkDataError, match="duplicate"):
        dataset.select_tasks((TaskId("synthetic-1"), TaskId("synthetic-1")))


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("最终答案:alpha", "alpha"),
        ("text\n最终答案::: beta", ""),
        ("no marker", None),
    ],
)
def test_final_answer_parser_matches_pinned_deterministic_shortcut(
    response: str, expected: str | None
) -> None:
    assert parse_final_answer(response) == expected
