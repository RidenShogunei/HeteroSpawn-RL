# ADR-0002: Continue with a project-owned Turing backend and optional standalone vLLM rollout

- Status: Accepted
- Date: 2026-07-23
- Decision owners: repository maintainers

## Context

Native capability spikes for RLinf, verl, and OpenRLHF all stopped before a valid training loop
on the available RTX 2080 Ti host. Their pinned stacks require or hard-import combinations of
FlashAttention-2, vLLM V1, FlashInfer, SGLang kernels, or newer CUDA/toolkit behavior that is not
available on sm_75. Docker cannot change the GPU compute capability.

The existing LocalHF reference backend already proves exact-token generation, independent
Main/Sub LoRA updates, immutable checkpoints, explicit rollout synchronization, and restore on a
smaller local GPU. A separate standalone spike also proved that vLLM 0.7.0 can run on sm_75
through its V0/XFormers path, retain selected-token log-probabilities, batch four requests, and
load distinct LoRA adapters.

## Decision

- Do not select RLinf, verl, or OpenRLHF as the current training backend on the Turing host.
- Evolve the project-owned PyTorch/Transformers/PEFT backend from a reference implementation into
  the first end-to-end experimental training path.
- Keep Transformers generation as the correctness fallback.
- Permit an optional standalone vLLM rollout engine behind the existing `PolicyService`
  contract, initially pinned to vLLM 0.7.0, V0, FP16, eager execution, and XFormers.
- Treat vLLM as rollout-only. It does not own optimizer state, advantage semantics, phase
  transactions, or HeteroSpawn version publication.
- Refresh vLLM rollout weights initially by stopping the old worker, loading and hashing the new
  LoRA checkpoint in a new worker, validating it, and atomically publishing a new
  `RolloutRevision`.
- Keep modern distributed-framework selection deferred. Repeat the common backend matrix on
  Ampere-or-newer hardware with sufficient per-GPU memory before selecting one.

## Consequences

- HeteroSpawn can proceed on the available hardware without patching upstream framework cores or
  implementing CUDA kernels.
- The first training path will have lower throughput and more project-owned orchestration than a
  modern distributed framework.
- The standalone vLLM environment is an isolated compatibility island with pinned older
  dependencies. It cannot become a transitive dependency of the core or training environment.
- Worker restart adds synchronization latency but gives a simple, auditable revision barrier.
- Passing the standalone rollout spike does not complete Milestone 2.5 or validate distributed
  optimizer/checkpoint semantics.

## Validation

- `docs/validation/2026-07-23-rlinf-native-spike.md`
- `docs/validation/2026-07-23-verl-native-spike.md`
- `docs/validation/2026-07-23-openrlhf-native-spike.md`
- `docs/validation/2026-07-23-vllm-turing-rollout-spike.md`
- The aggregate standalone vLLM report records successful startup, exact-token trajectory
  alignment, 0/1/4 requests, two distinct LoRA revisions, resource use, and report digests.
