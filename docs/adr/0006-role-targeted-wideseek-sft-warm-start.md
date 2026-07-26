# ADR-0006: Use a role-targeted WideSeek SFT warm start before further RL scaling

- Status: Accepted
- Date: 2026-07-26
- Decision owners: repository maintainers

## Context

The fixed 16-task Qwen3-4B compliance profile produced legal Main spawn actions in every episode
and executed 55 real Search/Access calls. Only one final answer passed its required format and
received a non-zero exact outcome. Increasing direct-RL scale at this point would spend most
rollouts on a sparse output-format and evidence-synthesis failure rather than the dynamic-spawn
research question.

The pinned `RLinf/WideSeek-R1-train-data` revision contains questions, reference answers, and
format metadata. It does not contain supervised Main/Sub trajectories or Search/Access traces.
Plain question-to-reference-answer SFT would therefore teach Main to answer on its first turn,
bypassing the very spawn behavior the project is intended to study. Distilling the released
WideSeek-R1-4B policy would transfer behavior from an already RL-trained result and weaken
attribution of later HeteroSpawn improvements.

## Decision

- Add an optional, bounded SFT warm start before the existing RL cycle. It is a readiness aid, not
  a replacement reward or a benchmark result.
- Construct only two supervised behaviors:
  - `main_final`: Main receives a legal earlier spawn turn plus ordered successful worker results,
    then emits the required Markdown table or boxed final answer.
  - `sub_summary`: Sub receives legal Search and Access history with evidence, then emits a concise
    evidence summary with no further tool call.
- Do not supervise first-turn Main spawn decisions, spawn counts, subtask decomposition, Search
  queries, Access choices, or tool-call envelopes in the initial warm start. The compliance run
  already established legal spawn/tool usage, and these decisions remain RL-controlled.
- Use only the pinned WideSeek training revision. Never use xbench, a held-out WideSeek selection,
  development-Judge outputs, or reference answers from an evaluation split to construct SFT.
- The v1 constructor is deterministic. It partitions Markdown table rows into at most four
  ordered evidence blocks and uses one evidence block for boxed facts. It reuses the exact
  HeteroSpawn rollout prompts and tool schemas.
- Reference plaintext exists only inside the verified dataset/evaluator boundary and the
  in-memory constructor until tokenization. Reports contain counts and revisions, not prompts,
  targets, task answers, token arrays, or per-example target digests.
- Use a separate `SupervisedTrainingBatch` contract and a causal SFT objective whose mask is zero
  over prompt tokens and one over assistant target tokens. Do not manufacture rollout revisions,
  old log-probabilities, rewards, or advantages for SFT data.
- Aggregate the SFT loss by active target token within each behavior, then average the active
  behaviors. A shared Main/Sub batch therefore gives `main_final` and `sub_summary` equal loss
  mass; a role-filtered independent-policy batch reduces to that role's target-token mean.
- A supervised optimizer update must still create a new immutable `WeightVersion`; rollout
  services see it only after explicit synchronization creates a new `RolloutRevision`.
  Checkpoint, idempotency, restore, and phase-transaction rules remain unchanged.
- The first real run uses the shared Qwen3-4B QLoRA policy. Independent-policy experiments may
  later initialize both policies from the same audited warm-start checkpoint or use separately
  declared role-filtered batches; they may not silently change topology.
- Do not use the released `RLinf/WideSeek-R1-4b` as a teacher in the default experiment. Any later
  teacher-distillation study requires a separate ADR and must label the attribution change.

## Consequences

- The warm start directly targets the observed termination, evidence-synthesis, and answer-format
  failures while leaving dynamic spawn and tool selection available for RL.
- Synthetic reference-derived evidence does not prove retrieval quality or factual
  generalization. Those properties remain measured by fresh rollouts against the pinned offline
  environment and held-out evaluation.
- The supervised examples can teach copying from clean evidence. The dataset must therefore stay
  small, be followed immediately by the unchanged fixed compliance profile, and must not be
  reported as benchmark performance.
- SFT and RL batch types cannot be accidentally interchanged. Backend support can be added without
  weakening exact-token rollout contracts.

## Validation

- CPU fixtures verify deterministic construction, legal post-tool histories, absence of
  `main_spawn` and Sub tool-call targets, reference-safe summaries, exact prompt/target
  tokenization, target-only masks, and digest rejection.
- A real RTX 2080 Ti Qwen3-4B run must verify a non-zero LoRA update, immutable checkpoint,
  explicit sync, stale-revision rejection, and restore before the warm-start checkpoint is used.
- Rerun the exact fixed 16-task profile with no optimizer update during evaluation. The minimum
  engineering gate is:
  - all exact trajectory/event/revision checks pass;
  - at least 4 of 16 answers pass required format;
  - at least 2 of 16 receive non-zero exact outcome;
  - at least 12 of 16 episodes retain a legal spawn action.
- These thresholds are a go/no-go gate on the unchanged small profile, not a statistically
  meaningful model-quality claim. Failure triggers constructor/objective diagnosis rather than
  immediate larger-scale RL.

## References

- [Pinned WideSeek-R1 training dataset card](https://huggingface.co/datasets/RLinf/WideSeek-R1-train-data/blob/47832ea20581f78d32cd6b32b4b37b985cbbc9df/README.md)
- [WideSeek-R1 project page](https://wideseek-r1.github.io/)
- [Released WideSeek-R1-4B model card](https://huggingface.co/RLinf/WideSeek-R1-4b)
