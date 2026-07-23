# ADR-0003: Use a versioned xbench development correctness reward for initial training

- Status: Superseded by ADR-0004
- Date: 2026-07-23
- Decision owners: repository maintainers

The implementation and dedicated fixtures were removed after the WideSeek training environment
replaced them; this file remains only as historical rationale.

## Context

The trainable Main/Sub cycle now runs end to end, but its GPU conformance command deliberately
uses a synthetic reward. Starting a benchmark-driven training experiment requires a primary
outcome reward without exposing decrypted xbench answers to policies, traces, logs, or tracked
artifacts.

xbench first applies a deterministic final-answer exact-match shortcut and otherwise invokes its
pinned Gemini judge. The repository also has a MiniMax implementation of the pinned judge prompt,
but ADR-0001 correctly marks it development-only and non-comparable to the official benchmark.
Using an evaluator during optimization also makes the selected task IDs training data, so the
result cannot be presented as an official evaluation even if the official judge is later used.

## Decision

- The initial benchmark-driven training reward is binary xbench answer correctness: `1.0` for a
  correct terminal answer and `0.0` for an incorrect terminal answer.
- Always apply the pinned deterministic exact-match shortcut first.
- Treat the terminal payload of a validated trainable `ANSWER` action as the final answer even when
  it does not repeat xbench's presentation-layer `最终答案` marker.
- Support two explicitly named training modes:
  - `development-exact-only`: an exact miss receives `0.0` without a Judge call.
  - `development-judge`: an exact miss is scored by a pinned, versioned development Judge.
- Keep decrypted questions and reference answers inside `XBenchDataset`. The reward adapter may
  retain only task/episode IDs, booleans, revisions, usage counts, latency, and response digests.
- Reject official-comparable Judges in the training adapter. Official Gemini evaluation remains a
  separate held-out workflow and is never implied by a training reward.
- A Judge failure fails the phase by default. It must not silently become zero reward.
- Bind the reward cache to episode ID, task ID, and a response digest. Concurrent retries share one
  in-flight Judge call; reuse of an episode ID for different content is rejected.
- Include dataset revision, encrypted source digest, selected training task IDs, evaluator mode,
  and complete Judge revision in the reward revision digest.
- Keep invalid-action, spawn-cost, and Sub-failure components separate from correctness.
- Record and accept degenerate all-equal reward groups; they produce zero advantage and no policy
  gradient rather than being selectively resampled.

## Consequences

- The first training experiment is reproducible and cannot leak ground truth through policy-facing
  contracts.
- Exact-only training is deterministic and offline, but likely sparse and frequently degenerate.
- Development-Judge training costs API credits and is non-official. It may be useful for short
  optimization experiments but cannot establish benchmark quality.
- Task IDs used for optimization must be listed in the experiment manifest and excluded from any
  claimed held-out comparison.
- Citation/evidence quality rewards remain future components and are not mixed into the initial
  outcome reward.

## Validation

- Synthetic encrypted fixtures prove exact hit/miss behavior without persisting plaintext answers.
- Judge fixtures prove exact hits bypass the Judge and exact misses retain only safe audit fields.
- Unknown or mismatched tasks, official-comparable Judges, and Judge failures are rejected.
- Reward revision digests change with dataset source, selected task IDs, mode, or Judge revision.
