# MiniMax development Judge validation — 2026-07-22

## Protocol boundary

- Benchmark/evaluator revision: `xbench-evals@17c562192cc7e62215bfb98b65e9f8806fb95504`.
- Policy and development Judge: `MiniMax-M2.7`.
- Search: `minimax-coding-plan-mcp@0.0.4`.
- Judge prompt revision: `e3422231ab04e701dc551d28983572407970ed4ddd83c373cae7bb5a51cd2559`.
- Judge mode: `development-minimax-judge`.
- Official comparability: `false` by type and manifest contract.
- Output discipline: no decrypted prompt, reference answer, prediction, evidence, URL, model/Judge reasoning, verdict text, provider error body, or credential was persisted.

## First live five-repeat run

- Run ID: `xbench-minimax-judge-5repeat-20260722`.
- Manifest digest: `3195fdbeb48bfdd4c5bfe12d6e6d2a23d05605353c4d4fbf4a948e58fa6116b9`.
- Rollouts completed: 5 / 5.
- Spawn counts: `0, 2, 1, 0, 2`; successful/failed Subs: `4 / 1`.
- Main invalid attempts: 1.
- Policy tokens reported: 23,020.
- Judge calls: 5; valid verdicts: 3; format failures: 2.
- Tokens from valid Judge results: 4,585. Failed-verdict token usage was not recoverable in this pre-fix report.
- Development Judge result: 0 / 5; this is not an official xbench score.

The live-only failure showed that MiniMax does not reliably emit the upstream three-field verdict shape on the first attempt. The parser was not relaxed. Instead, the development adapter gained one bounded, versioned request to restate the same verdict in the required format.

## Format-repair validation run

- Run ID: `xbench-minimax-judge-5repeat-repair1-20260722`.
- Manifest digest: `583a02c000f91f3dbdeff8a2ff3d4c18f44e5e06d1c95e8503311bd45a73633e`.
- Fresh rollout attempts: 5; completed: 4; failed: 1 (`InvalidActionError`).
- Completed spawn counts: `0, 2, 0, 0`; successful Subs: 2.
- Main invalid attempts visible before the failed repeat: 1.
- Policy tokens from completed episodes: 13,309. The failed repeat occurred before safe partial-usage accounting was added, so this historical total is intentionally not retroactively estimated.
- Judge calls: 4; Judge failures: 0.
- Judge tokens: 5,867.
- Development Judge result: 0 / 5, with the failed rollout counted as incorrect.
- Rollout wall time: 398,188 ms; this historical report predates separate Judge latency accounting.

The failed repeat was not replaced. This preserves the fixed five-repeat denominator and avoids selectively resampling failures.

## Post-run hardening

- Failed episodes now raise a structured `EpisodeRunError` carrying only safe attempt, event, spawn, and provider-reported usage totals.
- Aggregate policy cost includes completed and failed episodes; a failed episode is never classified as a 0-spawn success.
- Judge results now include total latency across the initial verdict and optional format repair.
- Offline tests cover exact-match short-circuiting, prompt revision, verdict parsing, one-shot repair, token/latency aggregation, judge failure isolation, secret exclusion, and failed-episode partial metrics.
