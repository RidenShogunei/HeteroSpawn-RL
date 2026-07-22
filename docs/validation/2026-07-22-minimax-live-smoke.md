# MiniMax live API smoke validation — 2026-07-22

## Scope

- Real policy backend: MiniMax OpenAI-compatible API, model `MiniMax-M2.7`.
- Benchmark: pinned xbench-DeepSearch-2510, task `101` only.
- Retrieval: deterministic local mock, so no claim is made about live search quality.
- Output: summary-only; credentials, benchmark prompt, ground truth, model answer, and reasoning were not persisted.

## Results

### Real Main → Sub → Main conformance

- Spawn count: `1`.
- Sub statuses: `success`.
- Stable events: `3`.
- Main attempts: `2` (initial spawn and final answer).
- Invalid Main attempts: `0`.
- Trace trainable: `false`, as required for an external API policy.

### Real xbench task path

- Task scope: `1` task, not the full 100-task benchmark.
- Spawn count: `0`.
- Main attempts: `2`; the first illegal action was retained and the repair succeeded.
- Stable events: `2`.
- Development exact-only score: `0/1`.
- Official comparability: `false` because the pinned Gemini judge/repeat protocol was not run.

The score is not an architecture acceptance criterion. This run validates encrypted task loading, real policy inference, strict action parsing and repair, episode tracing, scoped evaluation, and safe summary output.

## Findings fixed before merge

1. MiniMax may return a leading `<think>` block inside `content`. The adapter now requests `reasoning_split` and normalizes a remaining inline block while retaining the raw-response digest.
2. A single-task smoke run initially reported against all 100 tasks. Evaluator scope is now explicit and rejects unknown task IDs.
3. One response returned HTTP 200 with an unusable schema. Schema-invalid responses now receive bounded retries and safe field/type-only diagnostics; response bodies and credentials are excluded.

## Offline regression gates

- Ruff lint and format: passed.
- Strict mypy: passed.
- Pytest: `20 passed`.
