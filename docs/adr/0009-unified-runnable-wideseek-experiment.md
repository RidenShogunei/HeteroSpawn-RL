# ADR-0009: Make one resumable WideSeek experiment the operational version

- Status: Accepted
- Date: 2026-07-27
- Decision owners: repository maintainers

## Context

The repository has separately validated Qwen3-4B SFT, shared-policy RL, independent Main/Sub
fresh-alternating RL, checkpoint recovery, and held-out evaluation. Those validations were run
from clean, commit-pinned checkouts, but the operational guide still required a human to copy
checkpoint paths between several diagnostic commands. That made the repository look like a
collection of demos and made it too easy for a remote agent to combine incompatible revisions or
evaluate different conditions with different budgets.

The backend and training semantics are already viable on RTX 2080 Ti. Replacing them would discard
validated behavior. The missing layer is one experiment identity that owns stage order, immutable
inputs, checkpoint lineage, restart state, and the final comparison.

## Decision

- `configs/wideseek-qwen3-4b-2080ti.json` is the canonical supported experiment profile.
- `heterospawn wideseek-run` is the canonical operational entry point. It runs:
  1. role-targeted shared SFT;
  2. shared-policy RL;
  3. independent Main/Sub RL forked from the same SFT checkpoint;
  4. update-free SFT, shared-RL, and independent-RL evaluation on one fixed task contract;
  5. paired task-cluster bootstrap comparison.
- One ignored run directory owns reports, checkpoints, phase transactions, and an atomic
  `state.json`. A stage is complete only after its report and checkpoint identities are verified
  and recorded in that state.
- `--resume` verifies every committed stage digest and continues from the first uncommitted stage.
  Configuration drift, missing reports, changed reports, or changed checkpoint identity are hard
  failures.
- Independent training accepts a verified Main/Sub checkpoint pair after cycle zero. Main and Sub
  optimizer steps and checkpoint lineages therefore continue across cycles without reconstructing
  either role from prompt text or a shared adapter.
- Training-phase transactions remain the inner crash boundary. If a process dies inside an
  optimizer phase, `wideseek-recover-phase` completes or restores that transaction without
  replaying its rollout. The outer v1 runner resumes complete stages automatically but fails
  closed rather than silently adopting a partially completed cycle.
- The comparison implementation accepts only passed, optimizer-free compliance reports with
  identical task, model, sampling, environment, evaluator, and tool-budget contracts. It averages
  rollouts within task and bootstraps paired task clusters.
- `wideseek-sft-train`, `wideseek-train-cycle`, `wideseek-compliance-baseline`, and
  `wideseek-recover-phase` remain supported stage-level diagnostics. The old
  `wideseek-train-smoke` spelling is a compatibility alias, not an operational workflow.
- Historical validation reports and ADRs remain immutable evidence. Per-experiment runtime clones
  are not alternative product versions and must not be used as the current source tree.

This decision does not change reward composition, fresh-rollout ordering, policy sharing,
episode-balanced loss, or benchmark fairness. It makes their already accepted implementations
reachable through one auditable experiment.

## Consequences

- A clean clone contains the complete SFT → RL → evaluation path and enough configuration for a
  remote agent to run it without editing source files.
- Shared and independent RL are branches from the same audited SFT checkpoint, so their held-out
  comparison has an explicit common origin.
- Completed cycle boundaries resume automatically. A crash inside a phase still needs the explicit
  recovery command and operator review because replaying or guessing around a pending optimizer
  update would weaken the transaction contract.
- Large assets, raw traces, model outputs, retrieved text, checkpoints, and credentials remain
  outside Git.
- The canonical profile is a feasible 2080 Ti research run, not a claim of WideSeek-R1 paper-scale
  reproduction or distributed training.

## Acceptance

- CPU tests prove strict config loading, deterministic stage order, atomic state persistence,
  configuration-drift rejection, digest verification, and resume from the first unfinished stage.
- Independent cycle two can restore both role checkpoints, preserve their optimizer steps, and
  produce a complete new checkpoint pair even when an empty Sub batch carries forward the prior
  Sub checkpoint.
- Comparison tests reject optimizer updates, contract drift, duplicate episode identities, and
  unpaired task sets.
- A clean remote clone runs the canonical command against the pinned assets and records the final
  comparison under one run directory.
