import pytest

from heterospawn.errors import InvalidActionError
from heterospawn.orchestration.models import AnswerAction, SpawnAction, parse_main_action


def test_answer_means_no_spawn() -> None:
    assert parse_main_action('{"kind":"answer","answer":"done"}') == AnswerAction(
        kind="answer", answer="done"
    )


def test_spawn_requires_one_or_more_subtasks() -> None:
    assert parse_main_action('{"kind":"spawn","subtasks":["a"]}') == SpawnAction(
        kind="spawn", subtasks=("a",)
    )
    with pytest.raises(InvalidActionError):
        parse_main_action('{"kind":"spawn","subtasks":[]}')
