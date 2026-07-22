# Milestone 2 CPU training-contract validation

Date: 2026-07-22

## Purpose

Validate the backend-independent exact-token and policy-version semantics before loading a local model or evaluating verl/RLinf. This validation does not make API-backed trajectories trainable.

## Covered contracts

- Exact response token and old-log-probability alignment.
- Training batches derived from immutable MODEL steps without decode/re-encode.
- System-rollout advantage normalization, including zero-spawn baseline episodes and degenerate reward groups.
- Episode-balanced sequence, agent-instance, and episode weights.
- Single, shared, heterogeneous, and frozen role-to-policy topology.
- `WeightVersion` update followed by an explicit `RolloutRevision` sync barrier.
- Idempotent batch replay, conflicting-digest rejection, stale-version rejection, and checkpoint identity.
- Fresh Main update/sync before the Sub rollout and an explicit empty-Sub skip.

## Result

The offline suite passed on the local development machine. The LocalHF LoRA backend and its opt-in GPU report are intentionally tracked as the next, separate change.
