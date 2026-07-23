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
from heterospawn.evaluation.judges import MiniMaxDevelopmentJudge
from heterospawn.orchestration.api_episode import ApiEpisodeOrchestrator
from heterospawn.policies.minimax import (
    MiniMaxChatClient,
    MiniMaxConfig,
    MiniMaxEvaluationPolicy,
)
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
        "--judge-backend",
        choices=("none", "minimax-development"),
        default="none",
        help="optional non-official judge for answers that miss direct exact match",
    )
    pilot_parser.add_argument(
        "--allow-network",
        action="store_true",
        help="required acknowledgement that this command spends external API credits",
    )
    local_parser = subparsers.add_parser(
        "local-contract-smoke",
        help="run the opt-in single-GPU exact-token LoRA contract",
    )
    local_parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="required acknowledgement that the pinned model may be downloaded",
    )
    local_parser.add_argument("--device", default="cuda:0")
    local_parser.add_argument(
        "--model-path",
        type=Path,
        help="optional local directory whose pinned model weight digest is verified",
    )
    local_parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/local-contract/checkpoints"),
    )
    local_parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/local-contract/report.json"),
    )
    vllm_parser = subparsers.add_parser(
        "vllm-rollout-contract-smoke",
        help="run LocalHF training with restart-synchronized standalone vLLM rollout",
    )
    vllm_parser.add_argument("--model-path", type=Path, required=True)
    vllm_parser.add_argument("--vllm-python", type=Path, default=Path(sys.executable))
    vllm_parser.add_argument("--training-device", default="cuda:0")
    vllm_parser.add_argument("--main-rollout-device", default="1")
    vllm_parser.add_argument("--sub-rollout-device", default="2")
    vllm_parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/vllm-rollout/checkpoints"),
    )
    vllm_parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path("artifacts/vllm-rollout/runtime"),
    )
    vllm_parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/vllm-rollout/report.json"),
    )
    fetch_parser = subparsers.add_parser(
        "wideseek-fetch-assets",
        help="download and verify pinned WideSeek runtime assets",
    )
    fetch_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/wideseek-train-data.json"),
    )
    fetch_parser.add_argument(
        "--destination",
        type=Path,
        default=Path("artifacts/wideseek-assets/train-data"),
    )
    fetch_parser.add_argument(
        "--endpoint",
        choices=("auto", "official", "mirror"),
        default="auto",
    )
    fetch_parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify files copied from another machine without network access",
    )
    wideseek_inspect = subparsers.add_parser(
        "wideseek-inspect-data",
        help="verify and inspect one WideSeek split without printing references",
    )
    wideseek_inspect.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/wideseek-train-data.json"),
    )
    wideseek_inspect.add_argument(
        "--data-dir",
        type=Path,
        default=Path("artifacts/wideseek-assets/train-data"),
    )
    wideseek_inspect.add_argument(
        "--split",
        choices=("width_20k", "depth_20k", "hybrid_20k"),
        default="hybrid_20k",
    )
    environment_check = subparsers.add_parser(
        "wideseek-check-environment",
        help="verify pinned corpus/retriever assets and probe the offline Search/Access service",
    )
    environment_check.add_argument(
        "--corpus-manifest",
        type=Path,
        default=Path("manifests/wideseek-wiki-2018-corpus.json"),
    )
    environment_check.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("artifacts/wideseek-assets/wiki-2018-corpus"),
    )
    environment_check.add_argument(
        "--retriever-manifest",
        type=Path,
        default=Path("manifests/wideseek-e5-base-v2.json"),
    )
    environment_check.add_argument(
        "--retriever-dir",
        type=Path,
        default=Path("artifacts/wideseek-assets/e5-base-v2"),
    )
    environment_check.add_argument(
        "--service-url",
        default="http://127.0.0.1:8000",
    )
    environment_check.add_argument(
        "--qdrant-url",
        default="http://127.0.0.1:6333",
    )
    environment_check.add_argument(
        "--probe-query",
        default="Red Bull",
        help="readiness query; text is never included in the report",
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
                judge_backend=args.judge_backend,
            )
        )
    if args.command == "local-contract-smoke":
        if args.model_path is None and not args.allow_model_download:
            raise SystemExit("--allow-model-download is required for the local model smoke")
        from heterospawn.backends.local_hf.config import LocalLoraConfig
        from heterospawn.backends.local_hf.smoke import run_local_contract_smoke

        report = asyncio.run(
            run_local_contract_smoke(
                config=LocalLoraConfig(
                    device=args.device,
                    model_path=args.model_path,
                    artifact_dir=args.artifact_dir,
                ),
                report_path=args.report,
            )
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "vllm-rollout-contract-smoke":
        from heterospawn.backends.local_hf.config import LocalLoraConfig
        from heterospawn.backends.vllm_rollout.smoke import (
            run_vllm_rollout_contract_smoke,
        )

        report = asyncio.run(
            run_vllm_rollout_contract_smoke(
                local_config=LocalLoraConfig(
                    device=args.training_device,
                    model_path=args.model_path,
                    artifact_dir=args.artifact_dir,
                    max_sequence_length=512,
                    max_new_tokens=32,
                ),
                vllm_python=args.vllm_python,
                main_rollout_device=args.main_rollout_device,
                sub_rollout_device=args.sub_rollout_device,
                runtime_dir=args.runtime_dir,
                report_path=args.report,
            )
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "wideseek-fetch-assets":
        from heterospawn.assets import AssetPreparer, load_asset_manifest

        manifest = load_asset_manifest(args.manifest)
        preparer = AssetPreparer()
        asset_report = (
            preparer.verify_copy(manifest, args.destination)
            if args.verify_only
            else preparer.prepare(manifest, args.destination, endpoint=args.endpoint)
        )
        print(asset_report.model_dump_json())
        return 0
    if args.command == "wideseek-inspect-data":
        from heterospawn.assets import load_asset_manifest
        from heterospawn.benchmarks.wideseek import load_wideseek_dataset

        manifest = load_asset_manifest(args.manifest)
        filename = f"{args.split}.jsonl"
        expected = next((file for file in manifest.files if file.path == filename), None)
        if expected is None or expected.sha256 is None:
            raise SystemExit(f"split is absent from manifest: {args.split}")
        wideseek_dataset = load_wideseek_dataset(
            args.data_dir / filename,
            split=args.split,
            expected_sha256=expected.sha256,
            revision=manifest.revision,
        )
        print(wideseek_dataset.summary().model_dump_json())
        return 0
    if args.command == "wideseek-check-environment":
        from heterospawn.assets import AssetPreparer, load_asset_manifest
        from heterospawn.search.wideseek_local import (
            WideSeekLocalConfig,
            WideSeekLocalToolService,
        )

        corpus_manifest = load_asset_manifest(args.corpus_manifest)
        retriever_manifest = load_asset_manifest(args.retriever_manifest)
        config = WideSeekLocalConfig(
            service_url=args.service_url,
            qdrant_url=args.qdrant_url,
        )
        if (
            corpus_manifest.revision != config.identity.corpus_revision
            or corpus_manifest.manifest_digest != config.identity.corpus_manifest_digest
            or retriever_manifest.revision != config.identity.retriever_revision
            or retriever_manifest.manifest_digest != config.identity.retriever_manifest_digest
        ):
            raise SystemExit("asset manifests differ from the pinned environment identity")
        preparer = AssetPreparer()
        corpus = preparer.verify_copy(corpus_manifest, args.corpus_dir)
        retriever = preparer.verify_copy(retriever_manifest, args.retriever_dir)
        environment = asyncio.run(
            WideSeekLocalToolService(config).check_environment(
                probe_query=args.probe_query,
            )
        )
        print(
            json.dumps(
                {
                    "environment": environment.model_dump(mode="json"),
                    "verified_assets": {
                        "corpus_files": corpus.verified_files,
                        "corpus_bytes": corpus.verified_bytes,
                        "retriever_files": retriever.verified_files,
                        "retriever_bytes": retriever.verified_bytes,
                    },
                },
                sort_keys=True,
            )
        )
        return 0
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
    judge_backend: str,
) -> int:
    dataset = load_xbench(dataset_path)
    minimax_config = MiniMaxConfig.from_environment()
    policy = MiniMaxEvaluationPolicy(PolicyId("shared-minimax"), minimax_config)
    judge = (
        MiniMaxDevelopmentJudge(MiniMaxChatClient(minimax_config))
        if judge_backend == "minimax-development"
        else None
    )
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
            judge_mode="minimax-development" if judge is not None else "none",
        ),
        judge=judge,
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
