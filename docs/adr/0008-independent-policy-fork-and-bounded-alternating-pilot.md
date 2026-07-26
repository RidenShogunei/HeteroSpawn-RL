# ADR-0008: Fork the audited shared SFT checkpoint into independent Main/Sub policies

- Status: Accepted
- Date: 2026-07-27
- Decision owners: repository maintainers

## Context

ADR-0007 proved that the audited 48-step shared Qwen3-4B SFT checkpoint can produce a real
WideSeek rollout, non-degenerate system advantages, one non-zero LoRA update, explicit rollout
synchronization, and a crash-safe phase commit. The shared policy remains a diagnostic baseline:
Main answer synthesis and Sub evidence summarization update the same adapter and optimizer.

The primary HeteroSpawn topology instead gives Main and Sub separate train adapters, rollout
adapters, optimizer states, `WeightVersion` lineages, and synchronization barriers. Starting those
policies from unrelated random LoRA adapters would discard the verified warm start. Treating one
`shared` checkpoint as both policies without changing its identity would violate the checkpoint
contract and make later restore and audit records ambiguous.

## Decision

- Add one explicit LocalHF checkpoint-fork operation. Its only supported use in this experiment is
  a verified `shared` checkpoint forked into the configured `main` and `sub` policies.
- Verify the source canonical manifest, base-model identity, and every checkpoint file before any
  target checkpoint is published.
- Copy the source train-adapter parameters and full AdamW optimizer state into each target. Do not
  reset optimizer moments and do not perform an optimizer step during the fork.
- Restore the source RNG state once. Each target checkpoint records the resulting RNG state, but
  the fork itself does not consume rollout samples or claim a training update.
- Materialize a new immutable checkpoint per target policy. Each manifest records:
  - `kind: policy_fork`;
  - source policy ID, optimizer step, checkpoint digest, and optimizer-file digest;
  - target policy ID through the normal checkpoint identity.
- Target `WeightVersion` values retain the source optimizer-step number but have their own policy
  IDs and canonical checkpoint digests. Main and Sub checkpoint digests must therefore differ even
  when their initial adapter and optimizer contents are equal.
- Keep each rollout adapter on its old revision until that target checkpoint is explicitly
  synchronized. Construct the independent `PolicyRegistry` only after both sync operations
  succeed.
- Extend `wideseek-train-smoke --topology independent --checkpoint-dir ...` to use this operation.
  The safe report records only checkpoint/revision identities and equality checks, never model
  text, token arrays, prompts, retrieved content, or answers.
- Preserve the established fresh-alternating order:
  `M48/S48 rollout -> Main update/sync/commit -> fresh M49/S48 rollout ->
  Sub update/sync/commit`.
- An empty Sub batch remains a valid skip: it creates no optimizer step, checkpoint, or rollout
  revision. Runs intended to validate both updates use `--require-sub-update`.
- The first real pilot remains bounded and diagnostic: Qwen3-4B, 4,096-token context,
  1,024-token generation allowance, raw-policy sampling, pinned offline WideSeek environment,
  no semantic Judge, and no distributed optimizer. Task selection and rollout count must be
  declared in its validation record before execution.

## Consequences

- Both roles inherit the same audited SFT capability while all subsequent learning state is
  independently attributable.
- Equal initial adapters do not claim architectural model heterogeneity; this validates
  role/policy/optimizer/version separation for the current same-base-model experiment.
- Copying optimizer moments is more faithful to the audited SFT lineage than resetting AdamW, but
  it also carries shared-SFT optimization history into both roles. A reset-optimizer ablation
  requires a separate declared experiment.
- The fork operation remains LocalHF-specific checkpoint materialization. Core training and
  orchestration contracts continue to depend only on policy IDs, versions, checkpoints, and
  rollout revisions.

## Acceptance

- A tiny Qwen contract test proves:
  - source manifest and files are verified;
  - Main and Sub adapters equal the source immediately after the fork;
  - both optimizer states equal the source semantically but are held by independent optimizers;
  - target policy IDs and checkpoint digests are distinct;
  - both target manifests contain exact source-lineage fields;
  - sync is independent and stale revisions are rejected.
- The bounded real cycle must prove:
  - initial Main/Sub adapter hashes are equal to the source SFT adapter hash;
  - Main advances exactly once while Sub remains unchanged during the Main phase;
  - the Sub phase uses a fresh rollout snapshot containing updated Main and unchanged Sub;
  - a non-empty Sub batch advances Sub exactly once without changing Main;
  - both phase commits restore under their independent policy identities;
  - exact token/log-probability round-trip and environment revision checks remain true.
- The result is a systems validation, not a benchmark-quality or reward-improvement claim.
