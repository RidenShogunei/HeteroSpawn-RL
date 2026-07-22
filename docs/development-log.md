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
