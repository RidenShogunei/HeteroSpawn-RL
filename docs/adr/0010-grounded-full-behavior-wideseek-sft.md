# ADR-0010: Grounded full-behavior WideSeek SFT before transfer eval

- Status: Accepted
- Date: 2026-07-28
- Decision owners: repository maintainers

## Context

ADR-0006 introduced a role-targeted warm start that supervises only `main_final` and
`sub_summary`, using synthetic evidence partitioned from reference answers. That design fixed
format/termination failures on the fixed compliance profile, but deliberately left Main spawn and
Sub Search/Access decisions to RL.

BrowseComp-Plus transfer eval of the format-SFT + short shared RL checkpoint showed near-zero
accuracy with very low Search/Access usage. The pinned `RLinf/WideSeek-R1-train-data` revision
contains questions, reference answers, and format metadata only — not Main/Sub tool trajectories.
A “full SFT from the official dataset” therefore requires constructing complete trajectories from
that Q/A seed, not loading ready-made rollout JSONL.

## Decision

- Add constructor schema `heterospawn-wideseek-role-sft-v2` that supervises five behaviors per
  successfully grounded task:
  - `main_spawn`
  - `sub_search`
  - `sub_access`
  - `sub_summary`
  - `main_final`
- Ground every supervised Search/Access turn against the pinned offline WideSeek
  `ResearchToolService` (Wiki-2018 + Qdrant/E5). Drop tasks whose partitions cannot be supported by
  retrieved evidence under a deterministic overlap check.
- Keep using the pinned `hybrid_20k` revision as the only SFT seed. Do not distill
  `RLinf/WideSeek-R1-4b`, and do not train on BrowseComp-Plus queries or corpus.
- Reuse exact HeteroSpawn Main/Sub prompts and tool schemas. Reference plaintext remains confined to
  the dataset/evaluator boundary and in-memory construction until tokenization.
- Aggregate causal SFT loss by active target token within each behavior present in the batch, then
  average those behaviors (same equal-behavior mass rule as ADR-0006, extended to five labels).
- Primary transfer claim after this SFT is formal BrowseComp-Plus 830 with the existing official
  BM25 + Qwen3-32B judge path. No RL is required between this SFT and that eval for attribution.
- ADR-0006 constructor v1 remains available for format-only warm starts and historical reruns.

## Consequences

- SFT can teach legal spawn and tool-use envelopes grounded in the training retrieval stack, which
  is a prerequisite for meaningful BC+ transfer comparison.
- Keep-rate will be less than 20k; ungroundable hybrid tasks are excluded and reported only as
  counts, never with answer plaintext.
- Synthetic-reference SFT (v1) remains useful for format gates but must not be described as
  full-behavior SFT.
- Longer train schedules and higher peak VRAM/IO cost than ADR-0006 warm starts.

## Validation

- Unit tests with a mock `ResearchToolService` verify deterministic construction, legal tool-call
  targets for all five behaviors, grounding rejection, and role/mask digest contracts.
- A real CUDA SFT run must report keep-rate, per-behavior example counts, non-zero LoRA update,
  immutable checkpoint/sync/restore checks, and constructor revision for schema v2.
- Formal BrowseComp-Plus 830 on the resulting shared checkpoint must publish Accuracy, Recall, and
  Search/Access usage beside the prior `rl-shared-c4` transfer numbers.

## References

- [ADR-0006](0006-role-targeted-wideseek-sft-warm-start.md)
- [Pinned WideSeek-R1 training dataset](https://huggingface.co/datasets/RLinf/WideSeek-R1-train-data)
