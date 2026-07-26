# Qwen3-4B WideSeek role-targeted SFT warm start

- Date: 2026-07-26
- SFT training contract: Passed
- Post-SFT compliance gate: Not passed
- Direct-RL readiness: Not passed
- Officially comparable: No
- Source commit used by the runner: `807ab4ff91309b77c0269d45e3ff8ea935f2e713`

## Scope

This run applied exactly one shared-policy QLoRA optimizer step to the role-targeted examples
declared by ADR-0006, then restored the immutable checkpoint into a replacement backend and ran
the unchanged fixed 16-task compliance profile.

The SFT selection used `hybrid_20k` indices 1, 2, 3, 4, 7, 8, and 9. The command's overlap guard
verified that none occurs in the fixed compliance profile. The deterministic in-memory
constructor produced only post-evidence `main_final` and post-Access `sub_summary` targets; it did
not supervise Main spawn decisions or Sub Search/Access actions.

## Pinned identities

- Model: `Qwen/Qwen3-4B`
- Model revision: `1cfa9a7208912126459214e8b04321603b3df60c`
- Model-manifest digest:
  `7d1f0f5002a45efc43d4800d460e1e11d9849dee17a587a34d003709c5e57ce9`
- WideSeek data revision: `47832ea20581f78d32cd6b32b4b37b985cbbc9df`
- Training-data source digest:
  `74492dab6f32f318ae6b5d7e1c77f6c14aac3a8dfc0e749ea2c3bba4af32cd51`
- Offline environment revision:
  `2b90af41266aafdbf27af8743f7a80700567afaa24ec5993961b39150c193340`
- Compliance selection-profile digest:
  `52b65c562a2673dd7e96f6584ee6a35af0a9a6210bbd49e2209aa05a771cd5a5`
- Sampling seed: `20260722`

The complete corpus/E5 manifests, 26,134,257-point Qdrant collection, and a non-empty
Search-to-Access probe were verified before execution.

## Resource boundary and retry

The first eight-task attempt contained a 3,482-token supervised sequence. It terminated after
publishing only the initial step-zero checkpoint and before publishing an update or report. The
captured output did not include a CUDA exception, so this is recorded as an incomplete
resource-bound attempt rather than a successful or failed model-quality run.

The completed run excluded that one training index and retained the other seven disjoint tasks.
Its longest sequence was 2,101 tokens. This changed only the SFT construction selection; the
post-SFT compliance profile and its 4,096/512-token rollout limits were unchanged.

## SFT results

The completed batch contained 20 examples and 2,516 supervised target tokens:

| Behavior | Examples | Target tokens | Longest complete sequence |
|---|---:|---:|---:|
| `main_final` | 7 | 1,131 | 2,101 |
| `sub_summary` | 13 | 1,385 | 961 |

The role-balanced loss was `0.288014`, with gradient norm `0.739216`. The optimizer produced
step 1 and changed the train adapter. The rollout adapter remained unchanged before explicit
synchronization and matched the train adapter afterward.

All SFT contract checks passed:

- prompt/target IDs and target-only masks were used without decode/re-encode;
- both declared supervised behaviors were present and equally weighted by behavior;
- idempotent replay returned the original update;
- stale rollout revision was rejected;
- checkpoint restore reproduced the trained adapter and version;
- replacement-process synchronization published a fresh deployment identity.

The SFT/checkpoint/sync/restore portion took 104.947 seconds and peaked at 7,638,265,344
PyTorch-allocated bytes on one RTX 2080 Ti.

## Fixed post-SFT compliance results

The evaluator used the same fixed profile, seed, raw-policy sampling, and Search/Access display
budgets as the pre-SFT baseline. It performed no optimizer update and made no Judge request.

| Metric | Pre-SFT baseline | Post-SFT |
|---|---:|---:|
| Successful episodes | 15 / 16 | 15 / 16 |
| Required format passed | 1 / 16 | 1 / 16 |
| Non-zero exact outcome | 1 / 16 | 1 / 16 |
| Episodes with legal spawn | 16 / 16 | 16 / 16 |
| Tool calls | 55 | 58 |

Exact token/log-probability alignment, stable event order, unchanged evaluation weights, and
unchanged adapter hashes all passed. The compliance run took 535.003 seconds and peaked at
5,462,735,360 PyTorch-allocated policy bytes.

The ADR-0006 minimums require at least four format-valid answers and two non-zero outcomes.
Those two thresholds were not met. Legal spawn retention and all rollout-contract checks did
pass.

## Decision

The implementation is ready to perform real supervised updates, immutable checkpointing,
explicit rollout synchronization, replacement-process recovery, and held-out Search/Access
evaluation on RTX 2080 Ti hardware. The single small optimizer step did not measurably improve
the declared behavior gate, so its checkpoint must not be treated as an RL-ready initialization.

The next experiment should diagnose SFT strength and construction coverage while preserving the
same held-out gate. In particular, it should use a bounded multi-step or multi-epoch schedule,
retain the role-balanced objective, keep training indices disjoint, and record a validation curve
before any direct-RL scaling. It should not change the accepted reward, relax the format parser,
or train on the compliance tasks to manufacture a pass.

The ignored safe report SHA-256 values were:

- SFT contract report:
  `2e2c7ca51950ca0177256d080dd8491bab6e210f57420139459145a54f8607a9`
- Post-SFT compliance report:
  `5a30013bd0a3b76c3474c46990cb01e628be331e743e620a30f63e694b5d419e`

No prompt, reference answer, generated text, token array, retrieved content, checkpoint path,
credential, hostname, or service address is included in this record. The launcher released both
services, closed the retrieval ports, and returned the GPUs to idle state after the run.
