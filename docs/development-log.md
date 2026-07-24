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
- Audited the remote RTX 2080 Ti host without mutation and added a credential-safe preflight plus a remote-agent runbook for isolated verl/RLinf capability spikes.

## 2026-07-23

- Ran sequential native capability spikes for RLinf then verl on one idle RTX 2080 Ti with shared pinned `Qwen2.5-0.5B-Instruct` and the five-item contract matrix.
- Kept candidates in isolated `$HOME/heterospawn-runtime/{rlinf,verl}` environments; never co-installed or co-ran them.
- Recorded RLinf as BLOCKED: vLLM worker hardcodes V1 (needs CC>=8.0), FSDP hardcodes FlashAttention2, and Transformer Engine failed against host CUDA toolkit 11.5; SGLang rollout init was the only partial success.
- Recorded verl as BLOCKED after RLinf teardown: official docs require >=24 GB HBM and CUDA>=12.8; Docker pull failed; UV install reached Ray init but rollout backends failed (`vllm` API mismatch / `sgl_kernel` sm_75 load failure).
- Wrote credential-safe validation notes and ignored `artifacts/backend-spikes/{rlinf,verl}/report.json`; deferred backend-selection ADR while both candidates remain blocked on this host profile.
- Ran an additional isolated OpenRLHF native spike on the same host/model/GPU matrix.
- Recorded OpenRLHF as BLOCKED: its CLI hard-imported unavailable FlashAttention before SDPA configuration, it forced vLLM V1, and its pinned vLLM/FlashInfer path failed on sm_75.
- Ran a standalone vLLM Turing spike with the V0/XFormers path: exact token/log-prob trajectories, 0/1/4 requests, two LoRA revisions, and worker-restart adapter refresh passed at about 5.7 GB GPU memory.
- Accepted ADR-0002: proceed with the project-owned PyTorch/Transformers/PEFT training path and keep standalone vLLM as an isolated optional rollout engine; modern distributed-framework selection remains deferred.
- Audited repository history and GitHub metadata for credentials/private data, then changed the repository visibility to public. Anonymous GitHub access is allowed; Git bundle transfer remains a fallback for unstable host connectivity.
- Implemented the ADR-0002 product adapter: immutable LocalHF-to-PEFT rollout artifacts, isolated batched vLLM workers, exact token/log-probability transport, strict revision checks, restart synchronization, rollback, and credential-safe worker environments.
- Passed the real three-GPU product contract on RTX 2080 Ti: independent Main/Sub updates and worker restarts, pre-sync old-revision service, post-sync stale rejection, four shared Sub requests, partner isolation, checkpoint restore, and resource reporting all succeeded.
- Implemented the first complete trainable Main → Sub → Main cycle: immutable causal event traces, strict invalid-action repair retention, structured sibling-isolated Sub failures, caller-versioned reward composition, per-task system advantage normalization, episode-balanced batches, and fresh Main-first alternating updates.
- A real three-GPU validation exposed and fixed FP16 state in later-created PEFT adapters; every LoRA adapter now stays FP32 over the frozen FP16 base, and non-finite losses, gradients, or post-step weights are rejected before checkpoint publication.
- Added opt-in phase-specific regex-constrained vLLM decoding for deterministic contract tests while leaving normal rollout sampling unconstrained.
- Passed the complete real-model cycle with two full rollouts in each phase, exact raw-token/log-probability round-trip, Main update/sync, fresh Sub-phase rollout, Sub update/sync, independent revisions, and no stale trajectory reuse.
- Accepted ADR-0003 for binary xbench development training reward. Exact-only and non-official
  development-Judge modes keep ground truth evaluator-only, reject official-comparable Judges,
  and bind idempotent verdict caches to episode/task/response identity.
- Added crash-safe phase transactions to the trainable cycle: immutable input and pending records,
  conditional atomic commit publication, empty-Sub commits, and manifest-driven recovery.
- Passed CPU fault injection before/after pending-checkpoint persistence and before/after manifest
  publication; recovery preserved one logical optimizer step and did not publish stale registry
  state. Replacement rollout deployments are captured in append-only recovery manifests. The full
  suite reached 70 passed with the optional local-model test skipped by default.
- Accepted ADR-0004 and Architecture Baseline v0.3: WideSeek-R1 is the primary RL environment,
  while HeteroSpawn continues to own exact-token rollout, LoRA optimization, synchronization, and
  phase recovery instead of depending on the blocked RLinf runtime.
- Introduced the provider-neutral `ResearchTask` boundary and Search/Access service contracts.
  xbench keeps a compatibility alias and remains a held-out evaluation path; policy-visible tasks
  no longer carry reference answers.
- Added the bounded WideSeek multi-round agent loop: Main tool-call spawning, Sub Search/Access,
  per-Sub URL provenance, episode-local concurrency accounting, deterministic event insertion,
  sibling-isolated failures, and exact-token retention across every model turn.
- Versioned the WideSeek tool schema, parser, and prompts against the pinned upstream revision.
  Canonical tool request/result payloads and their digests are retained as auditable trace facts.
- Added the pinned WideSeek training-data trust manifest, bounded official-to-mirror fallback,
  resumable standard Hub downloads, corrupt-file quarantine, and answer-safe split inspection.
- Verified all 60,000 real width/depth/hybrid records and 129,978,280 bytes through the mirror
  after the official endpoint was unreachable; the committed manifest remains source-independent.
- Ported boxed and Markdown-table parsing, required/unique-column checks, item-level F1, a
  provider-neutral semantic Judge with digest-only cache, and auditable role-specific reward
  totals. Judge failure is phase-fatal rather than silently scored as zero.
- Pinned the complete 155,895,995,164-byte Wiki-2018/Qdrant inventory and the exact E5-base-v2
  runtime subset with source-independent content digests.
- Added the official-shaped offline `/retrieve` and `/access` client, collection/configuration
  checks, a safe environment report, and a crash-clean Linux launcher for the pinned upstream
  retrieval server. Full corpus execution remains an opt-in remote acceptance step.
- Completed the WideSeek training closeout path: shared-policy joint updates and independent
  Main-first fresh alternating updates now consume the same multi-round environment traces,
  role-aware reward totals, exact token/log-probability samples, version synchronization, and
  crash-safe phase transactions.
- Added credential-safe rollout/train smoke commands, a hard 128-call MiniMax development-Judge
  budget, and environment-drift rejection across dataset, corpus, tool, prompt, Judge, and reward
  revisions.
- Removed the superseded xbench training reward and duplicate fixtures. xbench task loading,
  API pilots, exact evaluation, repeat aggregation, and held-out Judge paths remain intact.
- A real RTX 4060 controlled-fixture run passed shared and independent WideSeek-shaped cycles at
  about 3.41 GB peak allocated VRAM. The selected greedy 0.5B rollout produced 0-spawn and
  zero-variance reward groups, so Main recorded a zero-gradient optimizer transaction and the
  empty Sub batch was correctly skipped; no reward-improvement claim is made.
- Fixed LocalHF tool-aware prompt revision validation and cross-platform LF-stable PEFT rollout
  artifact digests after the real WideSeek smoke exposed both integration boundaries.
- Made each LocalHF backend process publish a unique deployment identity. A clean replacement
  process can restore the same immutable weight/optimizer/RNG checkpoint, but it rejects the
  terminated process's rollout revision and publishes a new revision after synchronization.
- Verified all 3,400 files and 155,895,995,164 bytes of the pinned Wiki-2018 corpus after automatic
  mirror fallback, then launched the complete Qdrant/E5/Access environment. Qdrant reported
  26,134,257 indexed vectors and the Access service loaded 5,903,530 pages.
- Passed a real offline Search-to-Access smoke plus shared `width_20k` and independent `depth_20k`
  LocalHF cycles on RTX 2080 Ti hardware, including MiniMax development judging, atomic phase
  commits, and replacement-process recovery. The sampled 0.5B policy used 0-spawn and produced
  degenerate advantages, so this validates the closed loop but does not claim learning progress.
- Fixed two full-environment findings: the Linux launcher now passes absolute trusted-manifest
  paths after changing working directories, and identical concurrent semantic-Judge requests use
  per-key single-flight instead of racing digest-only cache publication.
- Made the offline launcher own Qdrant and retrieval as dedicated process groups after full-scale
  teardown exposed orphaned E5 workers. Bounded group termination passed a parent/child probe, and
  the acceptance host returned all GPUs to their idle state.

## 2026-07-24

- Accepted ADR-0005: keep Qwen2.5-0.5B as the inexpensive contract model and use pinned
  Qwen3-4B QLoRA as the research-scale policy profile on 11 GB Turing GPUs.
- Added full multi-shard model-manifest verification, NF4 QLoRA loading, gradient checkpointing,
  and manifest-based checkpoint identity. The real independent Main/Sub contract passed at
  4.41 GB peak allocated VRAM.
- Added raw-policy categorical sampling for non-degenerate same-task rollouts and verified that
  rollout old log-probs match update-time recomputation. Warped sampling remains explicitly out of
  contract.
- Rejected length-truncated Main answers and Sub summaries, disabled Qwen3 thinking for the
  bounded 4096-token profile, and versioned deterministic model-visible Search/Access budgets.
- Decoupled the retrieval-service Python runtime from the QLoRA environment in the crash-clean
  offline launcher.
- Reworked LocalHF episode-balanced optimization to backward one sequence at a time, releasing
  each graph while preserving the exact batch gradient.
- Passed a real Qwen3-4B shared WideSeek cycle on one width task with two rollouts: 3 spawned Subs,
  10 real tool outcomes, 18 trainable sequences, non-degenerate advantages, non-zero gradient,
  adapter update, checkpoint/sync, atomic phase commit, and restore. Peak allocated VRAM was
  7.29 GB and the longest complete training sequence was 1,985 tokens.
