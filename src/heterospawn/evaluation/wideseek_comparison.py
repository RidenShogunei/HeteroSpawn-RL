"""Paired task-cluster comparison for safe WideSeek compliance reports."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

from heterospawn.domain.training import canonical_digest

COMPARISON_SCHEMA_REVISION = "heterospawn-wideseek-comparison-v1"
_CONTRACT_KEYS = (
    "selection_profile",
    "task_selection",
    "rollouts_per_task",
    "model_id",
    "model_revision",
    "model_identity_kind",
    "model_identity",
    "sampling_params",
    "tool_message_budgets",
    "environment_revision",
    "dataset_revision",
    "evaluator_revisions",
    "judge_mode",
    "max_sequence_length",
    "max_new_tokens",
)


def _tool_calls(record: dict[str, Any]) -> float:
    counts = record.get("tool_counts")
    if not isinstance(counts, dict):
        raise ValueError("compliance episode tool_counts must be an object")
    return float(sum(_number(value, "tool count") for value in counts.values()))


_METRICS: dict[str, tuple[str, Callable[[dict[str, Any]], float]]] = {
    "outcome_mean": ("score", lambda record: _number(record.get("outcome_score"), "outcome")),
    "format_ok_rate": ("rate", lambda record: float(record.get("format_ok") is True)),
    "nonzero_outcome_rate": (
        "rate",
        lambda record: float(_number(record.get("outcome_score"), "outcome") > 0.0),
    ),
    "spawn_rate": (
        "rate",
        lambda record: float(_integer(record.get("spawn_count"), "spawn_count") > 0),
    ),
    "success_rate": ("rate", lambda record: float(record.get("status") == "success")),
    "invalid_main_episode_rate": (
        "rate",
        lambda record: float(
            _integer(record.get("invalid_main_attempts"), "invalid_main_attempts") > 0
        ),
    ),
    "failed_sub_episode_rate": (
        "rate",
        lambda record: float(_integer(record.get("failed_subs"), "failed_subs") > 0),
    ),
    "length_truncation_rate": (
        "rate",
        lambda record: float(record.get("length_truncated") is True),
    ),
    "tool_calls_per_episode": ("count", _tool_calls),
}


def compare_wideseek_reports(
    *,
    baseline_name: str,
    baseline_report: Path,
    candidate_reports: tuple[tuple[str, Path], ...],
    output_path: Path,
    bootstrap_samples: int = 10_000,
    seed: int = 20_260_727,
) -> dict[str, Any]:
    """Compare optimizer-free reports with paired task-cluster resampling."""

    if not baseline_name.strip():
        raise ValueError("baseline_name cannot be empty")
    if not candidate_reports:
        raise ValueError("at least one candidate report is required")
    candidate_names = tuple(name for name, _ in candidate_reports)
    if any(not name.strip() for name in candidate_names):
        raise ValueError("candidate names cannot be empty")
    if len(set(candidate_names)) != len(candidate_names) or baseline_name in candidate_names:
        raise ValueError("comparison condition names must be unique")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")

    reports = {
        baseline_name: _load_compliance_report(baseline_report),
        **{name: _load_compliance_report(path) for name, path in candidate_reports},
    }
    paths = {baseline_name: baseline_report, **dict(candidate_reports)}
    baseline = reports[baseline_name]
    contract = {key: baseline.get(key) for key in _CONTRACT_KEYS}
    baseline_records = _records_by_identity(baseline)
    task_keys = tuple(
        sorted(
            {(identity[0], identity[1]) for identity in baseline_records},
            key=lambda item: (item[0], item[1]),
        )
    )
    if not task_keys:
        raise ValueError("baseline report contains no compliance tasks")

    for name, report in reports.items():
        observed_contract = {key: report.get(key) for key in _CONTRACT_KEYS}
        if observed_contract != contract:
            differing = sorted(
                key for key in _CONTRACT_KEYS if observed_contract[key] != contract[key]
            )
            raise ValueError(f"comparison contract differs for {name}: {', '.join(differing)}")
        if set(_records_by_identity(report)) != set(baseline_records):
            raise ValueError(f"comparison episode identities differ for {name}")

    task_metrics = {
        name: _task_metric_means(_records_by_identity(report)) for name, report in reports.items()
    }
    bootstrap_indices = _bootstrap_indices(
        task_count=len(task_keys),
        samples=bootstrap_samples,
        seed=seed,
    )
    condition_summaries = {
        name: {
            metric: statistics.fmean(task_metrics[name][task][metric] for task in task_keys)
            for metric in _METRICS
        }
        for name in reports
    }
    comparisons: dict[str, dict[str, Any]] = {}
    for candidate_name in candidate_names:
        metric_deltas: dict[str, Any] = {}
        for metric, (unit, _) in _METRICS.items():
            per_task_deltas = tuple(
                task_metrics[candidate_name][task][metric]
                - task_metrics[baseline_name][task][metric]
                for task in task_keys
            )
            bootstrap_deltas = tuple(
                statistics.fmean(per_task_deltas[index] for index in indices)
                for indices in bootstrap_indices
            )
            ordered = sorted(bootstrap_deltas)
            metric_deltas[metric] = {
                "estimate": statistics.fmean(per_task_deltas),
                "lower_95": _percentile(ordered, 0.025),
                "upper_95": _percentile(ordered, 0.975),
                "unit": unit,
            }
        comparisons[candidate_name] = metric_deltas

    report_digests = {name: _file_sha256(path) for name, path in paths.items()}
    payload: dict[str, Any] = {
        "schema_revision": COMPARISON_SCHEMA_REVISION,
        "status": "passed",
        "comparable_to_official": False,
        "baseline": baseline_name,
        "candidates": list(candidate_names),
        "contract_digest": canonical_digest(contract),
        "selection_profile": baseline["selection_profile"],
        "task_count": len(task_keys),
        "rollouts_per_task": baseline["rollouts_per_task"],
        "bootstrap": {
            "method": "paired-task-cluster-percentile",
            "samples": bootstrap_samples,
            "seed": seed,
            "confidence": 0.95,
        },
        "condition_summaries": condition_summaries,
        "comparisons": comparisons,
        "source_report_digests": report_digests,
    }
    payload["comparison_digest"] = canonical_digest(payload)
    _write_json(output_path, payload)
    return payload


def _load_compliance_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load compliance report: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("compliance report root must be an object")
    if not str(payload.get("schema_revision", "")).startswith("heterospawn-wideseek-compliance-v"):
        raise ValueError("unsupported compliance report schema")
    if payload.get("status") != "passed" or payload.get("optimizer_updates") != 0:
        raise ValueError("comparison accepts only passed optimizer-free reports")
    checks = payload.get("checks")
    if (
        not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise ValueError("compliance report contract checks did not all pass")
    if not isinstance(payload.get("episodes"), list):
        raise ValueError("compliance report episodes must be a list")
    return payload


def _records_by_identity(report: dict[str, Any]) -> dict[tuple[str, int, int], dict[str, Any]]:
    records: dict[tuple[str, int, int], dict[str, Any]] = {}
    for raw in report["episodes"]:
        if not isinstance(raw, dict):
            raise ValueError("compliance episode must be an object")
        split = raw.get("split")
        if not isinstance(split, str) or not split:
            raise ValueError("compliance episode split is invalid")
        identity = (
            split,
            _integer(raw.get("task_index"), "task_index"),
            _integer(raw.get("rollout_index"), "rollout_index"),
        )
        if identity in records:
            raise ValueError("compliance report contains duplicate episode identity")
        records[identity] = raw
    return records


def _task_metric_means(
    records: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[tuple[str, int], dict[str, float]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for (split, task_index, _), record in records.items():
        grouped.setdefault((split, task_index), []).append(record)
    return {
        task: {
            metric: statistics.fmean(getter(record) for record in task_records)
            for metric, (_, getter) in _METRICS.items()
        }
        for task, task_records in grouped.items()
    }


def _bootstrap_indices(
    *,
    task_count: int,
    samples: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(seed)
    return tuple(
        tuple(rng.randrange(task_count) for _ in range(task_count)) for _ in range(samples)
    )


def _percentile(ordered: list[float], probability: float) -> float:
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
