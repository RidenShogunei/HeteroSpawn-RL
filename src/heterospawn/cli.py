"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from heterospawn import __version__
from heterospawn.benchmarks.xbench import load_xbench
from heterospawn.domain.ids import EpisodeId, PolicyId, TaskId
from heterospawn.orchestration.api_episode import ApiEpisodeOrchestrator
from heterospawn.policies.minimax import MiniMaxConfig, MiniMaxEvaluationPolicy
from heterospawn.search.tavily import TavilyConfig, TavilySearchService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="heterospawn")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser(
        "inspect-xbench", help="verify encrypted xbench data without printing plaintext"
    )
    inspect_parser.add_argument("dataset", type=Path)

    run_parser = subparsers.add_parser(
        "run-api-task", help="run one xbench task through MiniMax and Tavily"
    )
    run_parser.add_argument("dataset", type=Path)
    run_parser.add_argument("task_id")
    run_parser.add_argument(
        "--allow-network",
        action="store_true",
        help="required acknowledgement that this command spends external API credits",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect-xbench":
        dataset = load_xbench(args.dataset)
        print(
            json.dumps(
                {
                    "benchmark": "xbench-DeepSearch-2510",
                    "encrypted_source_sha256": dataset.source_digest,
                    "task_count": len(dataset.tasks),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run-api-task":
        if not args.allow_network:
            raise SystemExit("--allow-network is required for external API calls")
        return asyncio.run(_run_api_task(args.dataset, args.task_id))
    return 0


async def _run_api_task(dataset_path: Path, task_id: str) -> int:
    dataset = load_xbench(dataset_path)
    task = next((item for item in dataset.tasks if item.task_id == TaskId(task_id)), None)
    if task is None:
        raise SystemExit(f"task id not found: {task_id}")

    policy = MiniMaxEvaluationPolicy(PolicyId("shared-minimax"), MiniMaxConfig.from_environment())
    search = TavilySearchService(TavilyConfig.from_environment())
    trace = await ApiEpisodeOrchestrator(policy, search).run(
        task,
        EpisodeId(f"api-{task_id}"),
    )
    score = dataset.evaluate_exact({task.task_id: trace.answer})
    print(
        json.dumps(
            {
                "trace": trace.model_dump(mode="json"),
                "score": score.model_dump(mode="json"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
