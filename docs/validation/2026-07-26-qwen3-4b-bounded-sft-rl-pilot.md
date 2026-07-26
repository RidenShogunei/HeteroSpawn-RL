# Qwen3-4B bounded SFT-to-RL pilot

- Date: 2026-07-26
- Clean 192-task SFT: Passed
- Shared-policy RL learning signal: Passed after crash-safe phase recovery
- Fresh post-update rollout: Passed
- Spawn-retention gate: Not passed
- Officially comparable: No
- Source lineage: `d1d490950807f0dd2f61353705f3e3c955367836` plus the
  SDPA/recovery changes recorded in this validation

## Scope

This diagnostic recreated the audited shared-policy SFT schedule, initialized one bounded
WideSeek shared-policy RL cycle from its new immutable checkpoint, and restored the committed RL
checkpoint in a new rollout deployment. It used eight answer-independent, SFT-disjoint width
positions with eight complete system rollouts per task and exactly one joint optimizer update.
No semantic Judge or network API was used.

The run validates the SFT-to-rollout-to-reward-to-update-to-sync-to-restore path. It does not
override the ADR-0006 spawn-retention gate, establish reward improvement, or authorize a larger
RL run.

## Pinned identities

- Model: `Qwen/Qwen3-4B`
- Model revision: `1cfa9a7208912126459214e8b04321603b3df60c`
- Model-manifest digest:
  `7d1f0f5002a45efc43d4800d460e1e11d9849dee17a587a34d003709c5e57ce9`
- WideSeek data revision: `47832ea20581f78d32cd6b32b4b37b985cbbc9df`
- Offline environment revision:
  `2b90af41266aafdbf27af8743f7a80700567afaa24ec5993961b39150c193340`
- RL task positions: `1`, `2500`, `5000`, `7500`, `10000`, `12500`, `15000`,
  and `17500` in `width_20k`
- Rollouts per task: `8`
- Total sequence limit: `4096`
- Per-turn generation allowance: `1024`
- Search/Access display budgets: three results, 600 Search-content characters, and 800
  Access characters
- Sampling: raw policy, temperature `1.0`, top-p `1.0`, top-k `0`

The launcher reverified all 3,400 corpus files and 155,895,995,164 bytes, all nine E5 files, the
26,134,257-point green Qdrant collection, and a non-empty Search-to-Access probe. All services
were stopped after validation and all nine GPUs returned to idle.

## Clean SFT recreation

The clean run used 192 deterministic hybrid tasks, four tasks per optimizer step, one epoch, a
2,304-token training cap, and the 4,096/1,024 rollout identity.

| Measurement | Result |
|---|---:|
| Constructed examples | 540 |
| Skipped over-length examples | 87 |
| Optimizer steps | 48 |
| First-step loss / gradient norm | 3.922760 / 7.888940 |
| Last-step loss / gradient norm | 0.002157 / 0.111397 |
| Elapsed time | 1,361.686 s |
| Peak PyTorch-allocated VRAM | 8,205,477,888 B |

All exact-token, update, sync, stale-revision, immutable-checkpoint, and replacement-restore
checks passed. The final clean SFT checkpoint is optimizer step 48 with digest
`07029f5df3016006085288fc81d23e8d359202b9989be1ceff8d070f7521ec44`.
The ignored SFT report SHA-256 is
`c4a5aaa4428710d0f3b56fdc77db36b952960af18d58d66370ed1e853c08cfab`.

## RL transaction and recovery

The initial rollout command completed all 64 episodes and durably wrote the joint phase input
before update. The input contains 206 exact MODEL sequences and has digest
`1495973a3a88f4d6cad28ca039f0b057052420571d978a714d25cea5f4469f65`.
The eager-attention backward then failed before `optimizer.step()` on the longest 4,019-token
sequence. CUDA reported 9.54 GiB in use and an unsatisfied 1.87 GiB allocation. No pending
checkpoint, sync, or commit had been published, so the phase remained recoverable from step 48
without replaying any rollout.

The Qwen3 Turing profile was changed to explicit PyTorch SDPA, non-reentrant gradient
checkpointing, and response-only logit projection. The projection keeps the prompt-final causal
position plus every response-predicting position; it does not truncate prompt context, response
tokens, masks, or gradients. A real Qwen3 equivalence probe compared full and compact logits on
the same sequence:

| Check | Result |
|---|---:|
| Full/compact shapes equal | Yes |
| Maximum response-logit absolute error | 0.0 |
| Maximum selected-log-prob absolute error | 0.0 |
| Elementwise equality | Yes |

The previously failing 4,019-token sample then completed with a non-zero `0.003523` gradient at
7,634,836,992 B peak PyTorch-allocated VRAM. The durable phase was recovered without another
Search/Access or model-sampling pass.

| RL measurement | Result |
|---|---:|
| Episodes | 64 |
| Training sequences | 206 |
| Non-zero-advantage sequences | 206 |
| Non-degenerate task groups | 8 / 8 |
| Joint loss | 0.011386 |
| Gradient norm | 0.319293 |
| Old/new ratio mean | 1.0000004 |
| Approximate KL mean | -0.000000186 |
| Entropy mean | 0.208878 |
| Optimizer step | 48 → 49 |

The atomic commit digest is
`01225ea02123651d7977db65abdf7d8f1106412c3d2de620b9aa9110c250396a`.
The committed checkpoint digest is
`717d2e773181976f107da46b1eb8fbfb73c2791888ba877abe5d8a87885e4e23`.
A replacement process restored and synchronized this checkpoint. Re-running recovery left the
commit digest and optimizer step unchanged and produced one append-only deployment recovery
manifest.

The ignored recovery report SHA-256 is
`11cf8b500a6a6e0cdae4e452c10c6ca308b09cedd89f6673971d70af213d72f8`.
It passed persisted token/log-prob/mask alignment, single-step advancement, atomic commit,
checkpoint restore, adapter change, non-degenerate advantages, and finite non-zero gradient. All
training fields already present in the step-48 checkpoint matched the recovery runtime; the
report names the three newly introduced memory-profile fields as explicit config extensions.

The phase input predates the new attention/logit fields, so its original config digest cannot
name the memory fix. The effective SDPA and response-only settings are captured in ADR-0007, the
step-49 canonical checkpoint manifest, and the recovery report. This is an audited
numerically-equivalent recovery migration, not evidence that arbitrary runtime drift is safe.
Future recoveries must use the same pinned profile as their phase input.

## Fresh post-RL rollout

The committed step-49 checkpoint was loaded into a new backend, explicitly synchronized, and
evaluated on the unchanged fixed 16-task 4K/1024 profile. Exact token/log-prob alignment, stable
event order, and unchanged evaluation weights/adapters passed.

| Metric | Pre-RL step 48 | Post-RL step 49 |
|---|---:|---:|
| Successful episodes | 16 / 16 | 15 / 16 |
| Required format passed | 14 / 16 | 13 / 16 |
| Non-zero exact outcome | 5 / 16 | 6 / 16 |
| Mean exact outcome | 0.094051 | 0.204898 |
| Episodes with legal spawn | 7 / 16 | 7 / 16 |
| Episodes containing a length stop | 0 / 16 | 1 / 16 |
| Episodes with an invalid Main attempt | 0 / 16 | 1 / 16 |
| Episodes with a failed Sub | 0 / 16 | 0 / 16 |
| Model steps | 39 | 40 |
| Tool calls | 9 | 9 |
| Elapsed rollout time | 480.588 s | 556.196 s |
| Peak PyTorch-allocated policy memory | 4,636,782,080 B | 4,947,667,968 B |

This is an unpaired stochastic `G=1` diagnostic. The higher mean and non-zero count do not
establish an RL improvement, especially because success and format each fell by one. Spawn
remained 7/16, below the 12/16 readiness gate.

The ignored post-RL compliance report SHA-256 is
`615cd3dc93391c56d50231e928b85a1021c31518044df3c05f1aecfde9d2ad66`.

## Decision

The project now has a real Qwen3-4B SFT-to-RL update, synchronized immutable checkpoint,
crash-safe recovery, and fresh post-update rollout on one 2080 Ti. The next experiment should
not simply increase task count: shared-policy spawn retention remains the behavioral blocker.
Before scaling, select either independent Main/Sub initialization from the SFT lineage or a
targeted spawn-preservation training decision through a new ADR.

No prompt, reference answer, generated text, retrieved content, token array, credential,
hostname, service address, model cache, or checkpoint file is included in this record.
