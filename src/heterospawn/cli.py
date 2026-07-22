"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from heterospawn import __version__
from heterospawn.benchmarks.xbench import BenchmarkTask, load_xbench
from heterospawn.domain.ids import EpisodeId, PolicyId, TaskId
from heterospawn.evaluation.api_pilot import (
    ApiPilotConfig,
    ApiPilotRunner,
    PilotEpisodeSummary,
)
from heterospawn.orchestration.api_episode import ApiEpisodeOrchestrator
from heterospawn.policies.minimax import MiniMaxConfig, MiniMaxEvaluationPolicy
from heterospawn.search.base import SearchService
from heterospawn.search.minimax_mcp import (
    MINIMAX_MCP_REVISION,
    MiniMaxMcpConfig,
    MiniMaxMcpSearchService,
)
from heterospawn.search.mock import MOCK_SEARCH_REVISION, MockSearchService
from heterospawn.search.tavily import (
    TAVILY_SEARCH_REVISION,
    TavilyConfig,
    TavilySearchService,
)


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
    run_parser.add_argument(
        "--search-backend",
        choices=("mock", "tavily", "minimax-mcp"),
        default="mock",
        help="use deterministic local evidence or the live Tavily API",
    )
    run_parser.add_argument(
        "--summary-only",
        action="store_true",
        help="omit model text and emit a credential-safe validation summary",
    )

    conformance_parser = subparsers.add_parser(
        "run-api-conformance",
        help="run a synthetic real-model episode that requires one Sub",
    )
    conformance_parser.add_argument(
        "--allow-network",
        action="store_true",
        help="required acknowledgement that this command spends external API credits",
    )
    conformance_parser.add_argument(
        "--search-backend",
        choices=("mock", "tavily", "minimax-mcp"),
        default="mock",
        help="search backend used by the required Sub",
    )
    pilot_parser = subparsers.add_parser(
        "run-api-pilot",
        help="run a fixed credential-safe xbench task set through fresh API episodes",
    )
    pilot_parser.add_argument("dataset", type=Path)
    pilot_parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="task ID to include; repeat for multiple tasks (default: 101, 102, 103)",
    )
    pilot_parser.add_argument("--repeats", type=int, default=1)
    pilot_parser.add_argument("--run-id", default="xbench-api-pilot-v1")
    pilot_parser.add_argument(
        "--search-backend",
        choices=("mock", "tavily", "minimax-mcp"),
        default="minimax-mcp",
    )
    pilot_parser.add_argument(
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
        return asyncio.run(
            _run_api_task(
                args.dataset,
                args.task_id,
                search_backend=args.search_backend,
                summary_only=args.summary_only,
            )
        )
    if args.command == "run-api-conformance":
        if not args.allow_network:
            raise SystemExit("--allow-network is required for external API calls")
        return asyncio.run(_run_api_conformance(args.search_backend))
    if args.command == "run-api-pilot":
        if not args.allow_network:
            raise SystemExit("--allow-network is required for external API calls")
        task_ids = tuple(TaskId(item) for item in (args.task_ids or ("101", "102", "103")))
        return asyncio.run(
            _run_api_pilot(
                args.dataset,
                task_ids=task_ids,
                repeats=args.repeats,
                run_id=args.run_id,
                search_backend=args.search_backend,
            )
        )
    return 0


async def _run_api_task(
    dataset_path: Path,
    task_id: str,
    *,
    search_backend: str,
    summary_only: bool,
) -> int:
    dataset = load_xbench(dataset_path)
    task = next((item for item in dataset.tasks if item.task_id == TaskId(task_id)), None)
    if task is None:
        raise SystemExit(f"task id not found: {task_id}")

    policy = MiniMaxEvaluationPolicy(PolicyId("shared-minimax"), MiniMaxConfig.from_environment())
    search = _build_search(search_backend)
    trace = await ApiEpisodeOrchestrator(policy, search).run(
        task,
        EpisodeId(f"api-{task_id}"),
    )
    score = dataset.evaluate_exact(
        {task.task_id: trace.answer},
        task_ids=(task.task_id,),
    )
    if summary_only:
        output = {
            "task_id": trace.task_id,
            "episode_id": trace.episode_id,
            "model_provider": policy.revision.provider,
            "model": policy.revision.model,
            "search_backend": search_backend,
            "spawn_count": trace.spawn_count,
            "sub_statuses": [result.status for result in trace.sub_results],
            "main_attempts": len(trace.main_attempts),
            "invalid_main_attempts": sum(not attempt.valid for attempt in trace.main_attempts),
            "event_count": len(trace.events),
            "trainable": trace.trainable,
            "score": score.model_dump(mode="json"),
        }
    else:
        output = {
            "trace": trace.model_dump(mode="json"),
            "score": score.model_dump(mode="json"),
        }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


async def _run_api_conformance(search_backend: str) -> int:
    policy = MiniMaxEvaluationPolicy(PolicyId("shared-minimax"), MiniMaxConfig.from_environment())
    trace = await ApiEpisodeOrchestrator(policy, _build_search(search_backend)).run(
        BenchmarkTask(
            task_id=TaskId("api-conformance-1"),
            prompt=(
                "This is an orchestration conformance check. For the initial action, "
                "you must SPAWN exactly one subtask asking for the capital of France. "
                "After the Sub result arrives, return an ANSWER action."
            ),
        ),
        EpisodeId("api-conformance-1"),
    )
    output = {
        "task_id": trace.task_id,
        "episode_id": trace.episode_id,
        "model_provider": policy.revision.provider,
        "model": policy.revision.model,
        "search_backend": search_backend,
        "spawn_count": trace.spawn_count,
        "sub_statuses": [result.status for result in trace.sub_results],
        "main_attempts": len(trace.main_attempts),
        "invalid_main_attempts": sum(not attempt.valid for attempt in trace.main_attempts),
        "event_count": len(trace.events),
        "trainable": trace.trainable,
    }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    if trace.spawn_count != 1 or [result.status for result in trace.sub_results] != ["success"]:
        raise SystemExit("live conformance episode did not complete the required one-Sub path")
    return 0


async def _run_api_pilot(
    dataset_path: Path,
    *,
    task_ids: tuple[TaskId, ...],
    repeats: int,
    run_id: str,
    search_backend: str,
) -> int:
    dataset = load_xbench(dataset_path)
    policy = MiniMaxEvaluationPolicy(PolicyId("shared-minimax"), MiniMaxConfig.from_environment())
    report = await ApiPilotRunner(
        dataset,
        policy,
        _build_search(search_backend),
        ApiPilotConfig(
            run_id=run_id,
            task_ids=task_ids,
            repeats_per_task=repeats,
            search_backend=search_backend,
            search_revision=_search_revision(search_backend),
        ),
        progress_callback=_write_pilot_progress,
    ).run()
    print(report.model_dump_json())
    if report.failed_episodes:
        raise SystemExit("one or more pilot episodes failed")
    return 0


def _build_search(search_backend: str) -> SearchService:
    if search_backend == "tavily":
        return TavilySearchService(TavilyConfig.from_environment())
    if search_backend == "minimax-mcp":
        return MiniMaxMcpSearchService(MiniMaxMcpConfig.from_environment())
    return MockSearchService()


def _search_revision(search_backend: str) -> str:
    if search_backend == "tavily":
        return TAVILY_SEARCH_REVISION
    if search_backend == "minimax-mcp":
        return MINIMAX_MCP_REVISION
    return MOCK_SEARCH_REVISION


def _write_pilot_progress(summary: PilotEpisodeSummary) -> None:
    print(f"pilot-progress {summary.model_dump_json()}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
