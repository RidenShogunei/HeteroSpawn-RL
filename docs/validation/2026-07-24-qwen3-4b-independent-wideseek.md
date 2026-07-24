# Qwen3-4B independent-policy WideSeek validation

- Date: 2026-07-24
- Architecture contract: Passed
- Non-zero Main/Sub policy update: Not demonstrated
- Officially comparable: No
- Source commit: `7847213882317ba89e37adc997f33890e93dc280`
- Training host profile: one NVIDIA GeForce RTX 2080 Ti for QLoRA and one separate RTX
  2080 Ti for E5 retrieval

## Scope

This validation extends the shared-policy Qwen3-4B result with the real independent topology:

```text
(Main M0, Sub S0) rollout
-> Main M0 to M1 update, checkpoint, sync, and commit
-> fresh (Main M1, Sub S0) rollout
-> Sub S0 to S1 optimizer transaction, checkpoint, sync, and commit
```

It tests the pinned offline WideSeek Search/Access environment, exact raw-policy trajectories,
role-isolated QLoRA adapters, phase transactions, and checkpoint restore. It does not claim that
both roles received a useful learning signal.

## Pinned identities and preflight

- Model: `Qwen/Qwen3-4B`
- Model revision: `1cfa9a7208912126459214e8b04321603b3df60c`
- Model-manifest digest:
  `7d1f0f5002a45efc43d4800d460e1e11d9849dee17a587a34d003709c5e57ce9`
- WideSeek data revision: `47832ea20581f78d32cd6b32b4b37b985cbbc9df`
- Environment revision:
  `2b90af41266aafdbf27af8743f7a80700567afaa24ec5993961b39150c193340`
- Corpus revision: `178d7d037f661be3159b0c3a8a4119b974f01880`
- Corpus verification: 3,400 files and 155,895,995,164 bytes
- E5 verification: 9 files and 438,900,149 bytes
- Qdrant: green, 26,134,257 points, vector size 768
- Credential-safe preflight report digest:
  `cbd3df6a69a9fc48ac79d7d5756511e7a1903ce2bbacee0659a53d37e8e2f921`

The public repository clone failed because the remote TLS connection terminated during transfer.
A complete Git bundle containing the same merged commit was verified locally, copied over SSH,
and cloned into a new runtime directory. The existing dirty checkout was not modified.

The retrieval launcher reverified every pinned asset, served a non-empty Search-to-Access probe,
and owned Qdrant and retrieval as child process groups. It was terminated through its recorded
launcher PID after the runs; all nine GPUs returned to their idle state.

## Run 1: depth task, two rollouts

- Topology: independent
- Split/task: one `depth_20k` task
- Rollouts per task: 2
- Judge: none
- Elapsed: 96.793 seconds
- Peak allocated QLoRA VRAM: 5,610,544,640 bytes
- Main phase: 2 episodes, 4 samples, 2 spawned Subs, 4 tool calls, 0 degenerate groups
- Sub phase: 2 fresh episodes, 6 samples, 2 spawned Subs, 4 tool calls, 1 degenerate group
- Main gradient norm: `0.2156029`
- Sub gradient norm: `0.0`
- Main adapter changed: yes
- Sub adapter changed: no
- Safe report digest:
  `a95abbc718258c39e7c1a6f7330ee6ba78a39ee89806860afffd4a5e6adaaa0f`

Both Sub-phase system rewards were equal. The required MVP behavior set the Sub advantages to
zero and recorded the degenerate group. A non-empty Sub optimizer transaction still produced a
new checkpoint and revision, but it did not change the Sub adapter.

## Run 2: width task, four rollouts

The second run increased the group size instead of treating the first zero-gradient result as a
pass.

- Topology: independent
- Split/task: one `width_20k` task
- Rollouts per task: 4
- Judge: none
- Elapsed: 364.430 seconds
- Peak allocated QLoRA VRAM: 6,898,778,624 bytes
- Main phase: 4 episodes, 8 samples, 4 spawned Subs, 13 tool calls, 0 degenerate groups
- Sub phase: 4 fresh episodes, 17 samples, 5 spawned Subs, 15 tool calls, 1 degenerate group
- Main gradient norm: `0.4299037`
- Sub gradient norm: `0.0`
- Main adapter changed: yes
- Sub adapter changed: no
- Main phase commit:
  `fbda8a6c193f6de42db55bd6e62a22808c0df9dd89a466002455f41b7872a5b6`
- Sub phase commit:
  `990e1b3837004feea8c89a5f1174ec4650e6330b38d33e9953a0297fb638adea`
- Safe report digest:
  `ec6887af94425d5e4f157f5e9b9a13b9fe6d1465b66544317d78c4308ddab37a`

Increasing the group size did not create system-outcome variance for Sub. Main still received
non-zero advantages because its role total includes the configured spawn, search, token, and
invalid-action costs. Sub correctly used only the MVP system outcome, so its advantages and loss
remained zero.

## Contract results

Both runs passed:

- actual response token IDs and per-token old log-probabilities round-tripped without
  decode/re-encode;
- Main updated and committed before the Sub-phase rollout;
- the Sub phase used fresh episodes under the published Main revision and the original Sub
  revision;
- Main and Sub used distinct policy, optimizer, checkpoint, and rollout identities;
- both phase manifests were atomically published and bound to the dataset, corpus, tool, prompt,
  reward, and environment revisions;
- both Main and Sub checkpoints restored to their recorded `WeightVersion`;
- no external Judge call, credential, raw model response, token array, reference answer,
  checkpoint, hostname, or service address entered the tracked record.

## Interpretation and next gate

The real independent architecture is operational, but `optimizer_step == 1` is not evidence that
the corresponding adapter changed. The report's adapter digest and gradient metrics are the
authoritative learning-signal checks. These runs therefore close the independent rollout,
versioning, synchronization, transaction, and recovery gap, but do not close the non-zero Sub
learning gate.

Before a longer RL pilot, use one of two separately declared paths:

1. securely configure the versioned MiniMax development Judge at runtime and test whether its
   item-level semantic outcomes produce non-degenerate Sub groups; or
2. measure Qwen3 tool/answer-format compliance on a fixed multi-task sample and add a small
   warm-start SFT only if the base policy cannot produce reward-bearing answers.

Do not add Main costs to the Sub reward or enable local evidence advantage merely to manufacture
a gradient; either change would alter the accepted training semantics and require an ADR.
