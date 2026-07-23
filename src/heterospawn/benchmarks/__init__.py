"""Benchmark adapters and evaluator-only scoring surfaces."""

from heterospawn.benchmarks.wideseek import (
    WIDESEEK_TRAIN_REPO,
    WIDESEEK_TRAIN_REVISION,
    WideSeekDataset,
    WideSeekDatasetSummary,
    load_wideseek_dataset,
)
from heterospawn.benchmarks.xbench import (
    XBENCH_DATASET_SHA256,
    XBENCH_UPSTREAM_REVISION,
    BenchmarkTask,
    ExactScoreReport,
    JudgedRepeatScoreReport,
    RepeatExactScoreReport,
    XBenchDataset,
    load_xbench,
    parse_final_answer,
)

__all__ = [
    "WIDESEEK_TRAIN_REPO",
    "WIDESEEK_TRAIN_REVISION",
    "XBENCH_DATASET_SHA256",
    "XBENCH_UPSTREAM_REVISION",
    "BenchmarkTask",
    "ExactScoreReport",
    "JudgedRepeatScoreReport",
    "RepeatExactScoreReport",
    "WideSeekDataset",
    "WideSeekDatasetSummary",
    "XBenchDataset",
    "load_wideseek_dataset",
    "load_xbench",
    "parse_final_answer",
]
