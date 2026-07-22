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

## API smoke run

The runnable first architecture uses MiniMax for Main/Sub policy calls and Tavily as an independently observable search service. Both credentials are environment-only. Network use and external credit spending require an explicit flag:

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

The command emits no ground truth and labels its exact-only score non-comparable. API traces are evaluation-only because provider responses do not supply the exact rollout token IDs, old log-probabilities, or `RolloutRevision` required for RL training.

References: [MiniMax OpenAI-compatible API](https://platform.minimaxi.com/docs/api-reference/api-overview), [Tavily Search API](https://docs.tavily.com/documentation/api-reference/endpoint/search).
