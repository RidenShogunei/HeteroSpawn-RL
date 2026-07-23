# xbench training reward and phase transaction validation

- Date: 2026-07-23
- Scope: CPU contract and pinned encrypted xbench provenance
- Result: PASS
- Official benchmark result: no

## Validated boundary

- The pinned encrypted xbench file contains 100 tasks and retains SHA-256
  `a9378e56b05ec8f007b8ecc8f6ac74900abafd558267acd5839d0d05fbc6977a`.
- Binary training correctness uses exact match first and optionally a non-official, versioned
  development Judge. Official-comparable Judges are structurally rejected.
- Safe reward audits retain no question, reference answer, Judge text, or model response.
- Concurrent retries of the same episode/task/response share one Judge call. Conflicting response
  reuse is rejected.
- The reward revision covers the adapter, upstream dataset revision, encrypted source digest,
  selected training task IDs, evaluator mode, and complete Judge revision.

## Phase transaction checks

- The exact training input and base checkpoint/revisions are persisted before update.
- Training samples must belong to the recorded task, episode, and rollout identity sets.
- The pending immutable checkpoint is persisted before rollout synchronization.
- The new rollout revision becomes public only after conditional atomic commit publication.
- Empty Sub batches commit without changing optimizer or rollout revisions.
- Repeated identical writes are idempotent; conflicting records are rejected.
- Faults injected before and after pending persistence and before and after manifest publication
  recover to one logical optimizer step without exposing stale registry state.
- A restarted rollout deployment may receive a new deployment revision while loading the same
  committed weight; this mapping is retained in an append-only recovery manifest.

## Commands

```text
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src/heterospawn
heterospawn inspect-xbench data/private/xbench/DeepSearch-2510.csv
```

## Results

- pytest: 70 passed, 1 optional local-backend test skipped
- Ruff lint: pass
- Ruff format: pass
- mypy strict: pass
- xbench encrypted source verification: pass

This validation does not claim a real xbench optimization result. Runtime phase records, exact
token arrays, checkpoints, model outputs, and decrypted benchmark fields remain ignored artifacts.
