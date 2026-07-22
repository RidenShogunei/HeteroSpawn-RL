# ADR-0001: Validate the first benchmark with external APIs before local-model rollout

- Status: Accepted
- Date: 2026-07-22
- Decision owners: repository maintainers

## Context

The first implementation must distinguish benchmark misuse from model-rollout failures. xbench-DeepSearch has a small encrypted dataset and an official evaluator, while local trainable policies add tokenizer, inference-engine, weight-sync, and GPU variables before benchmark semantics are known to be correct.

## Decision

- Use xbench-DeepSearch-2510 as the first adapter and conformance benchmark.
- Use MiniMax's current OpenAI-compatible API with `MiniMax-M2.7` for the first API policy.
- Keep model inference and web search behind separate provider-neutral interfaces.
- Treat external-API episodes as evaluation-only. They cannot enter a training batch unless exact rollout tokens, old log-probabilities, and a rollout revision are available.
- Keep ground truth inside the evaluator. Policies receive only task input and budget.
- Preserve an official-comparability mode matching the pinned xbench evaluator. Development-only judges must be labeled non-comparable.

## Consequences

This vertical slice validates task loading, orchestration, search, output parsing, scoring, cost accounting, and traces without GPUs. A later local-model adapter can replace the policy service without changing benchmark or environment contracts. API text and provider request metadata remain auditable but are not misrepresented as on-policy RL trajectories.

## Validation

- Adapter scores fixed predictions identically to the pinned upstream evaluator.
- Decrypted answers never appear in policy inputs, traces, logs, or tracked files.
- Mock 0/1/4-Sub episodes pass without network access.
- Live tests are opt-in and skipped when credentials are absent.
