from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from heterospawn.evaluation.wideseek_comparison import compare_wideseek_reports


def _report(*, outcome_offset: float = 0.0) -> dict[str, Any]:
    tasks = (("width_20k", 0), ("depth_20k", 1))
    episodes = []
    for split, task_index in tasks:
        for rollout_index in range(2):
            outcome = min(1.0, 0.1 * (task_index + rollout_index + 1) + outcome_offset)
            episodes.append(
                {
                    "split": split,
                    "task_index": task_index,
                    "rollout_index": rollout_index,
                    "status": "success",
                    "format_ok": True,
                    "outcome_score": outcome,
                    "spawn_count": task_index,
                    "invalid_main_attempts": 0,
                    "failed_subs": 0,
                    "length_truncated": False,
                    "tool_counts": {"search:success": task_index + 1},
                }
            )
    return {
        "schema_revision": "heterospawn-wideseek-compliance-v3",
        "status": "passed",
        "optimizer_updates": 0,
        "checks": {"exact_token_logprob_alignment": True},
        "selection_profile": "profile",
        "task_selection": [
            {"split": split, "task_index": task_index} for split, task_index in tasks
        ],
        "rollouts_per_task": 2,
        "model_id": "Qwen/Qwen3-4B",
        "model_revision": "revision",
        "model_identity_kind": "hf-asset-manifest",
        "model_identity": "identity",
        "sampling_params": {"do_sample": True},
        "tool_message_budgets": {
            "search_results": 3,
            "search_content_characters": 600,
            "access_characters": 800,
        },
        "environment_revision": "environment",
        "dataset_revision": "dataset",
        "evaluator_revisions": {"width_20k": "evaluator", "depth_20k": "evaluator"},
        "judge_mode": "none",
        "max_sequence_length": 4096,
        "max_new_tokens": 1024,
        "episodes": episodes,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_comparison_is_paired_by_task_and_deterministic(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    _write(baseline_path, _report())
    _write(candidate_path, _report(outcome_offset=0.2))

    first = compare_wideseek_reports(
        baseline_name="sft",
        baseline_report=baseline_path,
        candidate_reports=(("rl", candidate_path),),
        output_path=tmp_path / "first.json",
        bootstrap_samples=200,
        seed=7,
    )
    second = compare_wideseek_reports(
        baseline_name="sft",
        baseline_report=baseline_path,
        candidate_reports=(("rl", candidate_path),),
        output_path=tmp_path / "second.json",
        bootstrap_samples=200,
        seed=7,
    )

    assert first["comparison_digest"] == second["comparison_digest"]
    assert first["comparisons"]["rl"]["outcome_mean"]["estimate"] == pytest.approx(0.2)
    assert first["task_count"] == 2
    assert first["bootstrap"]["method"] == "paired-task-cluster-percentile"


def test_comparison_rejects_contract_drift(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline = _report()
    candidate = _report()
    candidate["max_new_tokens"] = 512
    _write(baseline_path, baseline)
    _write(candidate_path, candidate)

    with pytest.raises(ValueError, match="max_new_tokens"):
        compare_wideseek_reports(
            baseline_name="sft",
            baseline_report=baseline_path,
            candidate_reports=(("rl", candidate_path),),
            output_path=tmp_path / "comparison.json",
            bootstrap_samples=100,
        )


def test_comparison_rejects_optimizer_or_duplicate_episode(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    invalid_path = tmp_path / "invalid.json"
    baseline = _report()
    invalid = _report()
    invalid["optimizer_updates"] = 1
    _write(baseline_path, baseline)
    _write(invalid_path, invalid)

    with pytest.raises(ValueError, match="optimizer-free"):
        compare_wideseek_reports(
            baseline_name="sft",
            baseline_report=baseline_path,
            candidate_reports=(("rl", invalid_path),),
            output_path=tmp_path / "comparison.json",
            bootstrap_samples=100,
        )

    invalid["optimizer_updates"] = 0
    invalid["episodes"].append(dict(invalid["episodes"][0]))
    _write(invalid_path, invalid)
    with pytest.raises(ValueError, match="duplicate episode"):
        compare_wideseek_reports(
            baseline_name="sft",
            baseline_report=baseline_path,
            candidate_reports=(("rl", invalid_path),),
            output_path=tmp_path / "comparison.json",
            bootstrap_samples=100,
        )
