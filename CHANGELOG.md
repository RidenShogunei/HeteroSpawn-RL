# Changelog

All notable project changes are recorded here. Architecture-changing decisions also require an ADR.

## Unreleased

### Added

- Architecture Baseline v0.2.
- Initial Python project, governance, CI, and CPU-only policy contracts.
- Pinned encrypted xbench-DeepSearch-2510 loader and non-comparable exact-only scorer.
- MiniMax evaluation policy and Tavily search adapters with bounded retries.
- API-first 0/1/4-Sub orchestration with structured repair and failure isolation.
- Live MiniMax conformance mode, inline-thinking normalization, scoped smoke scoring, and bounded invalid-schema retry.
- MiniMax Token Plan MCP search adapter, pinned stdio transport, and live Main/Sub search validation.
- Reproducible xbench API pilot manifests, repeat-aware exact-only metrics, bounded spawn, safe progress, and aggregate token/latency reporting.
- Provider-neutral xbench Judge contracts, MiniMax development judging, versioned verdict-format repair, and safe failed-episode/Judge cost accounting.
- Deterministic per-task pilot summaries, failure counts, Sub/Main totals, and nearest-rank latency percentiles after a recorded 15-episode MiniMax multi-task validation.
- Exact-token training contracts, episode-balanced batch derivation, policy topology registry, deterministic mock training backend, and fresh-rollout alternating coordinator.
- Optional single-GPU Hugging Face LoRA reference backend with independent Main/Sub adapters, exact generation log-probabilities, immutable checkpoints, explicit rollout sync, and a credential-safe smoke report.
