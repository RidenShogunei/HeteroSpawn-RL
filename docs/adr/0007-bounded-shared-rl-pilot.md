# ADR-0007: Run a bounded shared-policy RL pilot from the audited SFT lineage

- Status: Accepted
- Date: 2026-07-26
- Decision owners: repository maintainers

## Context

ADR-0006 blocks direct-RL scaling when the fixed compliance profile fails to retain legal spawn
in at least 12 of 16 episodes. The 192-task shared-policy SFT run improved output format but its
final checkpoint retained spawn in only 6 of 16 episodes. A controlled output-budget ablation
then showed that the 512-token generation cap was independently too short: at 4K/1024, all
length stops and invalid Main attempts disappeared, but spawn reached only 7 of 16.

An intermediate-checkpoint sweep could diagnose the exact format-versus-spawn frontier. It would
not answer the more immediate systems question: whether the current WideSeek reward, normalized
system-rollout advantage, shared joint update, synchronization, and phase transaction produce a
real non-zero RL update from the SFT lineage. A bounded pilot can answer that question without
claiming that the readiness gate or the research result is complete.

## Decision

- Permit one bounded diagnostic shared-policy RL cycle despite the failed ADR-0006 spawn gate.
  This is an exception for learning-signal and transaction validation, not authorization for
  direct-RL scaling.
- Recreate the deterministic 192-task, 48-step shared SFT run from the pinned base model and use
  its newly produced immutable checkpoint as the RL initialization. Record both report digests
  and the exact checkpoint identity.
- Add an explicit shared-checkpoint initialization path to `wideseek-train-smoke`. It must verify
  the canonical checkpoint manifest and every file, restore optimizer/RNG state, explicitly
  synchronize rollout weights, and bind the loaded `WeightVersion` into the RL config and phase
  transaction identities.
- Keep the total sequence limit at 4,096 and use the measured experimental generation allowance
  of 1,024. The repository default remains 512; this ADR authorizes only the named pilot profile.
- Use PyTorch SDPA for the Qwen3-4B Turing profile instead of eager attention. This is an
  explicit backend setting which preserves the causal attention and loss contract while keeping
  4K backward passes within the 2080 Ti memory envelope. Use explicit non-reentrant gradient
  checkpointing so PyTorch can stop recomputation once all required activations are available.
  For Qwen3 training forwards, materialize only the final response-length-plus-one logits needed
  for the exact shifted response loss; prompt logits that cannot contribute to the loss are not
  allocated.
- Use raw-policy sampling and eight complete rollouts for each of eight answer-independent,
  SFT-disjoint `width_20k` positions:
  `1, 2500, 5000, 7500, 10000, 12500, 15000, 17500`.
- Run exactly one shared-policy `joint_update` cycle. Use the pinned WideSeek exact/format/tool
  reward and existing spawn/search/token costs. Do not call a semantic Judge.
- Preserve the 3-result, 600-character Search-content, and 800-character Access display budgets
  used by the Qwen3 bounded profile.
- Publish safe task-group reward means, standard deviations, degenerate flags, non-zero
  advantage counts, and structural rollout counts. Do not publish questions, answers, model
  text, retrieved content, or token arrays.
- After the committed RL update, restore its immutable checkpoint in a fresh backend and run the
  fixed 16-task 4K/1024 compliance profile as a fresh rollout diagnostic.

## Acceptance

The pilot closes its learning-signal objective only if:

- exact token/log-probability round-trip, checkpoint restore, explicit sync, stable phase commit,
  and unchanged environment identity all pass;
- at least one of the eight task groups has non-zero reward variance and the joint batch contains
  non-zero advantages;
- the joint update has a finite non-zero gradient and changes the shared train adapter;
- the post-update compliance run uses the committed checkpoint and a fresh rollout deployment.

Reward or compliance improvement after one cycle is not required and must not be claimed from
one stochastic run. Failure of the non-degenerate advantage or non-zero-gradient checks is a
diagnostic result and does not authorize forced spawn, reward changes, or task selection based
on answers.

## Consequences

- The project can validate the real SFT-to-RL lineage without weakening the ADR-0006 gate for
  larger experiments.
- Shared-policy interference remains expected. A successful pilot validates the learning path,
  not the final HeteroSpawn topology.
- Independent Main/Sub initialization, fresh alternating RL, repeated budget comparison, and
  larger task counts remain separate decisions after this pilot.
- If the pilot cannot produce a useful signal on the predeclared width tasks, the next change
  must address policy initialization or reward/exploration semantics through another ADR rather
  than silently increasing compute.
