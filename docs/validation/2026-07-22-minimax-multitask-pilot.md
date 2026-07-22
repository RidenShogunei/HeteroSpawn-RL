# MiniMax multi-task development pilot — 2026-07-22

## Fixed protocol

- GitHub tracking: [issue #13](https://github.com/RidenShogunei/HeteroSpawn-RL/issues/13).
- Run ID: `xbench-minimax-multitask-15-20260722`.
- Manifest digest: `aee22518c2f4847ae7591b310ff1d5f8e86c1c799d077a3bd8d9022b4988be89`.
- Tasks: `101`, `104`, and `108`, selected by structural features without publishing decrypted text.
- Repeats: five fresh sequential episodes per task, 15 attempts total.
- Policy and development Judge: `MiniMax-M2.7`.
- Search: `minimax-coding-plan-mcp@0.0.4`.
- Benchmark/evaluator: `xbench-evals@17c562192cc7e62215bfb98b65e9f8806fb95504`.
- Failed episodes were not replaced. The run remained non-trainable and `comparable_to_official: false`.

## Outcome

| Scope | Completed | Failed | Failure counts | Spawn total | Successful / failed Subs | Invalid Main attempts | Policy tokens | Latency p50 / p95 / max |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| All | 7 | 8 | InvalidAction 3; Provider 5 | 14 | 14 / 0 | 10 | 85,541 | 245.219s / 487.547s / 487.547s |
| Task 101 | 4 | 1 | InvalidAction 1 | 4 | 4 / 0 | 3 | 27,674 | 52.000s / 130.609s / 130.609s |
| Task 104 | 2 | 3 | InvalidAction 1; Provider 2 | 8 | 8 / 0 | 5 | 47,160 | 366.953s / 449.187s / 449.187s |
| Task 108 | 1 | 4 | InvalidAction 1; Provider 3 | 2 | 2 / 0 | 2 | 10,707 | 350.812s / 487.547s / 487.547s |

Percentiles use the nearest-rank definition. The sum of rollout latencies was 3,561.000 seconds. Three completed episodes were true 0-spawn episodes. Across all attempts, the spawn-count distribution was `0: 9`, `1: 3`, `3: 1`, and `4: 2`.

All 14 spawned Sub instances completed successfully. The observed instability was instead concentrated in MiniMax chat availability and Main action formatting:

- Five episodes failed before the first Main response with `ProviderRequestError`.
- Three episodes failed after exhausting action-format repair with `InvalidActionError`; two of these had already completed one successful Sub.
- Successful task-104 episodes both used the four-Sub limit, while three of four successful task-101 episodes used no Sub. A single-task smoke run therefore does not represent orchestration depth or cost.

## Development Judge

- Direct exact matches: 0.
- Judge calls: 7; Judge failures: 0.
- Judge tokens: 7,780.
- Judge latency: 98.031 seconds.
- Development result: 0 / 15; best-of-five tasks: 0 / 3.

This is not an official xbench score. MiniMax exercised the pinned prompt/parser path only; the official-comparable Gemini boundary remains unimplemented.

## Engineering decisions

The live run motivated two changes without altering its completed manifest:

1. Safe reports now expose deterministic per-task aggregates, sorted failure counts, successful/failed Sub totals, invalid Main totals, and nearest-rank p50/p95/max latency.
2. Whole-episode deadlines and credential-safe phase progress are tracked separately in [issue #14](https://github.com/RidenShogunei/HeteroSpawn-RL/issues/14). Timeout cancellation must first preserve safe partial usage and event counts, so it is not retrofitted into this run.

No decrypted prompt, reference answer, prediction, evidence, URL, provider body, model/Judge reasoning, verdict text, or credential was persisted.
