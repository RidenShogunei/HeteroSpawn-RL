# ADR-0004: Adopt WideSeek-R1 as the primary RL environment

- Status: Accepted
- Date: 2026-07-23
- Decision owners: repository maintainers

## Context

HeteroSpawn-RL has validated exact-token rollout, independent LoRA updates, synchronization, and
crash-safe phase recovery. Its first benchmark integration, xbench-DeepSearch, was intentionally
API-first and useful for evaluation conformance, but it is not a complete offline RL environment.
Using xbench answers during optimization would also turn selected benchmark tasks into training
data and invalidate an official held-out claim.

WideSeek-R1 provides a better environment fit: decomposition-heavy width, depth, and hybrid
research tasks; a multi-turn planner/worker protocol; Search and Access tools backed by a pinned
offline corpus; and structured table/boxed-answer rewards. The upstream implementation is coupled
to RLinf and large-scale infrastructure that is not viable on the project's RTX 2080 Ti hosts.
The environment semantics can be adopted without adopting that runtime.

## Decision

- Use the WideSeek-R1 task, tool, prompt, parser, and reward semantics as the primary training
  environment.
- Pin the upstream environment implementation to
  `d9f3d8a9db4d7aad1d641029293295503dd3eb2c`, the training data to
  `47832ea20581f78d32cd6b32b4b37b985cbbc9df`, and the Wiki-2018 corpus to
  `178d7d037f661be3159b0c3a8a4119b974f01880`.
- Port the environment behind provider-neutral HeteroSpawn contracts. Do not make the core
  domain, rollout, optimizer, checkpoint, or phase-transaction layers depend on RLinf.
- Keep two first-class training topologies:
  - a shared-policy WideSeek baseline with one joint update;
  - independent Main/Sub policies with Main-first fresh alternating updates.
- Represent policy-visible tasks with `ResearchTask`. Reference answers and evaluator-only
  metadata remain private to the dataset/evaluator boundary.
- Treat a Main turn with no valid tool call as `ANSWER`; one to four valid `subtask` calls create
  a spawn round. Enforce at most three Main rounds and eight total spawned Sub instances.
- Allow each Sub up to four turns and at most three Search/Access calls per turn. Access is legal
  only for URLs previously returned to that Sub by Search.
- Preserve actual generated token IDs and old log-probabilities for every Main and Sub model turn.
  Tool requests, results, failures, repairs, provenance, and deterministic event ordering are part
  of the immutable episode trace.
- Prefer the official Hugging Face endpoint for pinned assets. After three retryable connection,
  timeout, or server failures, use `https://hf-mirror.com`. Authorization, missing-revision, and
  integrity failures do not trigger source switching. Every source must satisfy the same trusted
  manifest.
- Use MiniMax only as a versioned, cached development semantic Judge. It is not an official
  WideSeek or xbench comparable evaluator, and terminal Judge failure fails the phase.
- Keep xbench as a held-out generalized evaluation path. Supersede ADR-0003 for new training runs;
  its xbench reward adapter remains only for reproducing earlier validation until the WideSeek
  evaluator fully replaces the training fixture.

## Consequences

- HeteroSpawn gains a complete RL environment while retaining the already validated lightweight
  LocalHF/PEFT training and optional vLLM rollout paths.
- The 156 GB corpus, model, embeddings, indexes, checkpoints, traces, and Judge payloads remain
  ignored runtime artifacts and are never committed.
- Exact upstream score reproduction is not claimed: MiniMax is a development Judge and the
  training stack is project-owned.
- Environment identity becomes part of phase identity. Dataset, corpus, retriever, tool, prompt,
  evaluator, Judge, and reward revisions must match before rollout reuse or recovery.
- Distributed optimizers and high-throughput scheduling remain out of scope; correctness and
  recovery are validated on short single-node cycles first.

## Validation

- CI uses miniature WideSeek-shaped JSONL, in-memory retrieval, and a fake Judge.
- GPU conformance uses the pinned Qwen2.5-0.5B model and proves independent or shared adapter
  updates without decode/re-encode.
- Full remote acceptance uses a clean checkout, the complete pinned Wiki-2018/Qdrant environment,
  one width and one depth task with two system rollouts per phase, and checkpoint recovery.
