# Development log

This file records milestone-level events. Fine-grained work remains in GitHub issues, commits, CI runs, and pull requests.

## 2026-07-22

- Accepted Architecture Baseline v0.2.
- Selected xbench-DeepSearch-2510 as the first conformance benchmark.
- Selected an API-first evaluation slice using MiniMax-M2.7 and an independent search adapter.
- Established the rule that external API responses are not trainable trajectories without exact token/log-probability provenance.
- Pinned xbench-evals commit `17c562192cc7e62215bfb98b65e9f8806fb95504` and encrypted dataset digest.
- Implemented provider-neutral MiniMax policy and Tavily search boundaries using only environment credentials.
- Kept development exact-only scoring explicitly non-comparable to the official Gemini-judge protocol.
- Verified the pinned encrypted file in place: 100 tasks and SHA-256 `a9378e56b05ec8f007b8ecc8f6ac74900abafd558267acd5839d0d05fbc6977a`; no decrypted field was persisted.
- Validated strict actions, 0/1/4 Sub episodes, repair traces, partial Sub failure, adapters, and version contracts with 18 offline tests.
- Completed real MiniMax-M2.7 smoke validation: a synthetic 1-Sub Main → Sub → Main episode and xbench task 101 both reached terminal answers.
- Fixed live-only findings for inline thinking normalization, scoped single-task scoring, sanitized schema diagnostics, and bounded retry of HTTP-200 schema failures.
- Validated MiniMax Token Plan MCP `web_search` with the existing credential and completed a real M2.7 Main → MCP search → Sub → Main episode.
- Pinned `minimax-coding-plan-mcp==0.0.4` and restricted its subprocess environment to avoid inheriting unrelated credentials.
- Added a fixed-task, repeat-aware xbench API pilot with revision-complete manifests, fresh episode IDs, bounded spawn, token/latency accounting, failure isolation, and credential-safe progress/report output.
- Completed the first three-task MiniMax-M2.7 + MiniMax MCP pilot: 3/3 episodes and 6/6 spawned Subs completed; the exact-only result remained explicitly non-official.
- Removed the one-off `probe_minimax_mcp.py` after its live capability coverage was replaced by the formal pilot, MCP adapter contract tests, and retained synthetic conformance command. Shared encrypted xbench fixture construction moved to `tests/conftest.py`; no distinct contract tests were deleted.
- Added a provider-neutral xbench Judge contract and role-neutral MiniMax chat transport; MiniMax development verdicts reuse the pinned upstream prompt/parser but are structurally unable to claim official comparability.
- Ran two real five-repeat task-101 validations. The first exposed 2/5 malformed Judge verdicts; a single versioned format repair reduced Judge failures to 0/4 scoreable predictions in the second run. One rollout failed after exhausting Main action repair and remained an incorrect sample.
- Live findings also led to safe partial cost accounting for failed episodes and explicit Judge latency accounting; no failed repeat was replaced or selectively resampled.
- Ran a fixed 15-episode MiniMax multi-task pilot across tasks 101/104/108: 7 completed, 8 failed, 14/14 Subs succeeded, and the development Judge completed 7/7 calls; added safe per-task and latency aggregates and opened #14 for episode deadlines and phase progress.
- Implemented the Milestone 2 exact-token training contracts, topology registry, mock backend, and fresh-rollout coordinator without changing the non-trainable status of API traces.
- Added executable CPU contracts for stale-version rejection, immutable update/sync/checkpoint transitions, shared and frozen policies, 0/1/4-Sub advantage groups, episode-balanced weights, and empty Sub batches.
- Added the optional LocalHF LoRA reference backend and validated the pinned Qwen2.5-0.5B model on one RTX 4060 Laptop GPU: independent Main/Sub optimizer steps, explicit sync barriers, stale-revision rejection, four shared Sub calls, and checkpoint restore all passed.
