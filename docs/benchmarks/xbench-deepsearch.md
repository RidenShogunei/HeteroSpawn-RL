# xbench-DeepSearch-2510 integration

## Pinned provenance

- Upstream: <https://github.com/xbench-ai/xbench-evals>
- Commit: `17c562192cc7e62215bfb98b65e9f8806fb95504`
- Encrypted dataset: `data/DeepSearch-2510.csv`
- SHA-256: `a9378e56b05ec8f007b8ecc8f6ac74900abafd558267acd5839d0d05fbc6977a`

The fetch script downloads only the pinned encrypted CSV. The loader verifies its digest and decrypts prompt and answer cells in memory using the upstream canary/XOR format. Decrypted answers remain evaluator-only and must not be logged, traced, or written to disk.

```bash
python scripts/fetch_xbench.py
heterospawn inspect-xbench data/private/xbench/DeepSearch-2510.csv
```

## Scoring modes

`development-exact-only` mirrors the upstream evaluator's deterministic `最终答案` extraction and direct equality shortcut. Upstream sends mismatches to a Gemini 2.0 judge, so exact-only reports are always marked `comparable_to_official: false`.

An official-comparable report requires the pinned upstream evaluator, its judge protocol, repeat count, and aggregation behavior. A MiniMax-based judge or exact-only run must never be presented as an official xbench score.

The pinned upstream default is five repeats per task. It records average score, best-of-N, and a randomly tie-broken majority vote after sending non-exact answers to the Gemini 2.0 judge. `development-repeat-exact-only` preserves the repeat denominators and deterministic direct-match shortcut, but intentionally omits the judge and majority vote; it is therefore always non-comparable.

### Development training reward

Benchmark-driven development training uses the separate `XBenchOutcomeReward` contract defined by
[ADR-0003](../adr/0003-xbench-development-training-reward.md). It returns binary correctness and
supports only `development-exact-only` or a pinned non-official development Judge. An
official-comparable Judge is rejected at construction and evaluation time.

The adapter binds each cached verdict to episode ID, task ID, and a digest of the terminal answer.
Concurrent retries share one Judge call, while reuse of an episode ID for different content is
rejected. Reward audit records contain only IDs, booleans, revisions, usage, latency, and digests;
questions, answers, Judge text, and model text do not cross the evaluator boundary.

Training phases may opt into `FilePhaseTransactionStore`. Before an optimizer step it atomically
persists the exact digest-protected training input, base checkpoint/revisions, task/episode/rollout
IDs, configuration, RNG/sampler state, dataset/environment identity, and reward revision. A
pending checkpoint is recorded before rollout synchronization; only the atomic commit manifest
publishes the new `RolloutRevision`. Runtime transaction files belong under ignored `artifacts/`
and must never be committed.

## API smoke run

The runnable first architecture uses MiniMax for Main/Sub policy calls and a provider-neutral search service. Credentials are environment-only. Network use and external credit spending require an explicit flag:

```bash
set MINIMAX_API_KEY=<rotated-local-secret>
set TAVILY_API_KEY=<local-secret>
heterospawn run-api-task data/private/xbench/DeepSearch-2510.csv 101 --allow-network
```

For a model-only live conformance run, keep retrieval deterministic and suppress generated text:

```bash
heterospawn run-api-task data/private/xbench/DeepSearch-2510.csv 101 --allow-network --search-backend mock --summary-only
```

The synthetic conformance command forces one real Main → Sub → Main path without exposing benchmark text:

```bash
heterospawn run-api-conformance --allow-network
```

To use MiniMax's pinned Token Plan MCP `web_search` instead of a separate Tavily key:

```bash
heterospawn run-api-conformance --allow-network --search-backend minimax-mcp
```

This backend requires `uvx` on `PATH` and a MiniMax Token Plan credential in `MINIMAX_API_KEY`. The subprocess package is pinned by the adapter; no Tavily credential is used.

## Reproducible API pilot

The pilot runner executes every task/repeat as a fresh sequential episode and emits only revision-complete manifests, per-episode operational summaries, and aggregate development scores:

```bash
heterospawn run-api-pilot data/private/xbench/DeepSearch-2510.csv \
  --allow-network \
  --search-backend minimax-mcp \
  --task-id 101 --task-id 102 --task-id 103 \
  --repeats 1 \
  --run-id xbench-minimax-mcp-pilot-20260722
```

The default task set is `101, 102, 103`; explicit IDs are preferred for recorded runs. The manifest pins dataset/evaluator/model/search revisions, resolved sampling parameters, repeat count, execution mode, repair count, concurrency, and a maximum of four Sub instances per episode. Safe progress lines are written after each episode. Prompts, answers, evidence, search result bodies, provider error bodies, and reasoning are never included.

### MiniMax development Judge

MiniMax can exercise the complete judge path without another credential:

```bash
heterospawn run-api-pilot data/private/xbench/DeepSearch-2510.csv \
  --allow-network \
  --search-backend minimax-mcp \
  --judge-backend minimax-development \
  --task-id 101 \
  --repeats 5 \
  --run-id xbench-minimax-judge-5repeat
```

The adapter uses the byte-pinned upstream judge prompt and direct-match shortcut. A MiniMax verdict is always reported as `development-minimax-judge` with `comparable_to_official: false`. One versioned format-repair attempt may ask MiniMax to restate an invalid verdict in the required three-line schema; this retry changes neither the reference answer nor the correctness criteria. Judge token usage, latency, failures, prompt revision, repair revision, and response digests are reported without persisting judge text.

Pilot reports also include deterministic task-level aggregates, sorted failure counts, Sub and invalid-Main totals, and nearest-rank p50/p95/max episode latency. This makes task-shape and long-tail differences auditable without persisting benchmark or model text.

`gemini-official` remains a reserved compatibility mode. It will require a separate Google credential and an adapter that reproduces the pinned upstream API behavior before any report may claim official comparability.

The command emits no ground truth and labels its exact-only score non-comparable. API traces are evaluation-only because provider responses do not supply the exact rollout token IDs, old log-probabilities, or `RolloutRevision` required for RL training.

References: [pinned xbench evaluator](https://github.com/xbench-ai/xbench-evals/tree/17c562192cc7e62215bfb98b65e9f8806fb95504), [MiniMax OpenAI-compatible API](https://platform.minimaxi.com/docs/api-reference/api-overview), [MiniMax Token Plan MCP](https://platform.minimaxi.com/docs/guides/token-plan-mcp-guide), [Tavily Search API](https://docs.tavily.com/documentation/api-reference/endpoint/search).
