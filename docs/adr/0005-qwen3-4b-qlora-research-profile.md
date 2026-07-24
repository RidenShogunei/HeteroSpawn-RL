# ADR-0005: Use Qwen3-4B QLoRA as the research-scale policy profile

- Status: Accepted
- Date: 2026-07-24
- Decision owners: repository maintainers

## Context

The pinned Qwen2.5-0.5B LocalHF backend proves token, optimizer, synchronization, checkpoint, and
recovery contracts cheaply. Complete WideSeek acceptance also showed that its greedy base policy
does not reliably choose the structured `subtask` action. It is therefore a useful conformance
model but too small to be the primary model for behavior experiments.

WideSeek-R1's pinned upstream configuration uses Qwen3-4B. The available RTX 2080 Ti hosts have
11 GB per GPU and cannot run that model's full-precision training stack or the blocked
FlashAttention-dependent framework backends. The project-owned LocalHF backend can retain the
same training contracts while using parameter-efficient quantized training.

## Decision

- Keep `Qwen/Qwen2.5-0.5B-Instruct` FP16 LoRA as the default CI, CPU-fixture, and inexpensive GPU
  contract profile.
- Add `Qwen/Qwen3-4B` at revision
  `1cfa9a7208912126459214e8b04321603b3df60c` as the research-scale WideSeek profile.
- Verify every local Qwen3 file against the committed Hugging Face asset manifest before loading
  the model. The manifest, not a cache path or one shard digest, is the base-model identity stored
  in checkpoints and validation reports.
- Use bitsandbytes NF4 4-bit base weights with double quantization, FP16 compute, gradient
  checkpointing, and the existing rank-8 `q_proj`/`v_proj` LoRA adapters.
- Disable Qwen3 thinking in the bounded 4096-token CLI profile. This keeps the complete tool call
  and final answer inside the training window; longer-context thinking experiments require a
  separately versioned profile.
- Preserve separate train and rollout adapters and all existing `WeightVersion`,
  `RolloutRevision`, exact-token trajectory, synchronization, checkpoint, and phase-transaction
  semantics. Quantization does not weaken those contracts.
- Expose the profile through explicit CLI flags and a separate `qlora` optional dependency. Do
  not silently change the existing 0.5B defaults.
- Start WideSeek validation at a 4096-token context on one training GPU. Longer contexts,
  independent-policy throughput, and alternate quantization settings are measured extensions,
  not assumed capabilities.
- Bound the model-visible Search result count and content, plus Access content, independently.
  The 11 GB Turing validation uses at most three displayed results, 600 Search-content characters,
  and 800 Access characters. Preserve complete provider digests and provenance in the audit
  record, and include these deterministic display budgets in the prompt revision.

## Consequences

- Real behavior experiments use a model at the same parameter scale as the pinned upstream
  WideSeek configuration while remaining viable on Turing GPUs.
- The QLoRA backend depends on a recent Transformers version with Qwen3 support and on a
  platform-compatible bitsandbytes build. Those dependencies remain isolated from the core and
  from the retrieval-service environment.
- QLoRA validation proves the project architecture and short optimization loop. It does not claim
  exact reproduction of upstream large-scale RL, 32K-context throughput, or comparable benchmark
  scores.
- A length-truncated generation is never accepted as a direct Main answer or Sub evidence
  summary. It remains an invalid, trainable repair attempt.
- Full-model optimizer state, distributed training, and framework-specific FlashAttention kernels
  remain out of scope.

## Validation

- Run the shared LocalHF contract fixture against a randomly initialized tiny Qwen3 configuration
  without downloading model weights.
- On an RTX 2080 Ti, verify the pinned real model through its full asset manifest and run
  `generate -> update -> checkpoint -> sync -> restore` for independent Main/Sub adapters.
- Run at least one complete WideSeek shared-policy cycle with two rollouts, the pinned offline
  Search/Access environment, exact trajectory checks, phase commit, and checkpoint restore.
