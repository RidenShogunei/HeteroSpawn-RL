# xbench MiniMax + MCP API pilot — 2026-07-22

## Scope

- Benchmark: pinned `xbench-DeepSearch-2510` at commit `17c562192cc7e62215bfb98b65e9f8806fb95504`.
- Encrypted dataset digest: `a9378e56b05ec8f007b8ecc8f6ac74900abafd558267acd5839d0d05fbc6977a`.
- Tasks: `101`, `102`, `103`; one fresh episode per task.
- Policy: real `MiniMax-M2.7` through the evaluation-only API adapter.
- Search: real MiniMax Token Plan MCP `web_search`, package revision `minimax-coding-plan-mcp@0.0.4`.
- Manifest digest: `5049d53265d423f5a0ef06e3ec272310e9f02a22683218c7dc197403183e608d`.
- Output discipline: no decrypted prompt, reference answer, model answer, reasoning, evidence body, URL, provider error body, or credential was persisted.

## Results

| Metric | Result |
|---|---:|
| Completed episodes | 3 / 3 |
| Failed episodes | 0 |
| Spawn counts by task | 2, 1, 3 |
| Successful spawned Subs | 6 / 6 |
| Invalid Main attempts | 0 |
| Total prompt tokens | 6,472 |
| Total completion tokens | 11,268 |
| Total tokens | 17,740 |
| Total wall time | 348,656 ms |
| Development direct exact matches | 0 / 3 |
| Official comparability | false |
| Trainable trajectory | false |

The exact-only value is not an xbench leaderboard score. The pinned upstream evaluator sends non-exact answers to Gemini 2.0 and defaults to five repeats; neither the judge nor the official aggregation protocol ran in this pilot.

## Contract observations

- All task/repeat combinations received distinct deterministic episode and request IDs; model actions were not reused.
- The observed search revision matched the manifest for every successful Sub.
- The configured maximum of four spawned Subs bounded external cost and was enforced as a repairable Main action violation.
- Every model call contributed provider-reported token usage, including Main repair attempts when present.
- Episode failures are reduced to an exception class and do not expose provider response text or cancel later tasks.

## Cleanup audit

- Removed `scripts/probe_minimax_mcp.py`: the formal pilot now exercises the same real MCP initialization/search path, while `test_minimax_mcp_search.py` retains offline schema, error-redaction, version, and environment-isolation coverage.
- Retained `run-api-conformance`: it deterministically forces the one-Sub topology independently of benchmark model behavior.
- Retained `run-api-task`: it remains the smallest single-task diagnostic entry point.
- Consolidated synthetic encrypted xbench fixture generation in `tests/conftest.py` to avoid duplicate encryption/file helpers as pilot tests grow.
