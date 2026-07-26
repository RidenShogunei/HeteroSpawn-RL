# Qwen3-4B WideSeek 192-task multi-step SFT

- Date: 2026-07-26
- Multi-step SFT contract: Passed
- Post-SFT compliance gate: Not passed
- Direct-RL readiness: Not passed
- Officially comparable: No
- Source commit used by the runner: `161a3aa64c97005955fcca2795aca58046ac8576`

## Scope

This run expanded the ADR-0006 shared-policy warm start from seven tasks and one optimizer step to
192 held-out-disjoint `hybrid_20k` tasks and 48 chained optimizer steps. It then restored the final
immutable checkpoint into a replacement backend and ran the unchanged fixed 16-task compliance
profile.

The scheduler kept two distinct limits:

- supervised examples were admitted only when every sequence in their task was at most 2,304
  tokens;
- rollout and compliance retained the baseline 4,096-token context and 512-token generation
  limit.

Selection was based only on dataset position and tokenized resource size. All fixed compliance
indices were excluded before construction. No compliance reference or model outcome affected
training selection.

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
- SFT schedule digest:
  `5abf82b833c339ce87827653a493c93140133d8179ec1f6a1261525acbba77e4`
- Sampling seed: `20260722`

Before execution, the launcher reverified all 3,400 corpus files and 155,895,995,164 bytes, all
nine E5 files, the 26,134,257-point Qdrant collection, and a non-empty Search-to-Access probe.

## Training data and schedule

The scheduler scanned answer-independent positions until 192 eligible tasks were found. It
skipped 87 entire tasks because at least one constructed sequence exceeded the training-only
limit. It produced:

| Behavior | Examples | Combined tokens | Target tokens | Longest sequence |
|---|---:|---:|---:|---:|
| `main_final` | 192 | 151,861 | 26,410 | 2,302 |
| `sub_summary` | 348 | 255,148 | 33,363 | 1,026 |
| total | 540 | 407,009 | 59,773 | 2,302 |

Four complete tasks were grouped per optimizer step. Each batch retained its Main target and all
associated Sub targets, so both behaviors were active in every role-balanced update. The run
completed one epoch and 48 optimizer steps.

Mean loss over the first four steps was `1.733126`; mean loss over the last four was `0.023094`,
and the final-step loss was `0.002157`. Mean gradient norm fell from `3.808091` over the first
four steps to `0.434879` over the last four. These values demonstrate strong fitting to the
constructed data; they do not by themselves establish held-out behavior.

All engineering checks passed:

- exact prompt/target IDs and target-only masks, without decode/re-encode;
- a non-zero train-adapter change and unchanged rollout adapter before sync;
- 48 chained `WeightVersion` optimizer steps and immutable checkpoints;
- idempotent final-batch replay and stale rollout-revision rejection;
- final explicit synchronization;
- exact checkpoint restore and fresh replacement deployment identity.

Training/checkpoint/sync/restore took 1,368.855 seconds. Peak PyTorch-allocated policy memory was
8,205,477,888 bytes on one RTX 2080 Ti.

## Fixed 4,096-token compliance result

The evaluator used the same fixed task profile, seed, raw-policy sampling, Search/Access display
budgets, and 4,096/512 limits as the pre-SFT baseline. It performed no optimizer update and made
no Judge request.

| Metric | Pre-SFT baseline | Post-SFT step 48 |
|---|---:|---:|
| Successful episodes | 15 / 16 | 13 / 16 |
| Required format passed | 1 / 16 | 10 / 16 |
| Non-zero exact outcome | 1 / 16 | 4 / 16 |
| Episodes with legal spawn | 16 / 16 | 6 / 16 |
| Tool calls | 55 | 7 |
| Mean exact outcome | 0.007576 | 0.197917 |

By split:

| Split | Format passed | Non-zero outcome | Legal spawn |
|---|---:|---:|---:|
| width | 1 / 6 | 0 / 6 | 4 / 6 |
| depth | 5 / 5 | 2 / 5 | 0 / 5 |
| hybrid | 4 / 5 | 2 / 5 | 2 / 5 |

Exact token/log-probability alignment, stable event order, unchanged evaluation weights, and
unchanged adapter hashes all passed. Compliance took 590.734 seconds and peaked at
5,170,221,568 PyTorch-allocated policy bytes.

## Gate and interpretation

The ADR-0006 format threshold (at least 4/16) and non-zero-outcome threshold (at least 2/16) both
passed. The legal-spawn threshold (at least 12/16) failed: only 6/16 episodes spawned, and every
depth task answered directly.

The larger SFT set therefore fixed much of the boxed/Markdown termination behavior and improved
held-out exact outcome, confirming that the one-step run was data-limited. It also created
substantial shared-policy interference. Although first-turn Main answers were never supervised,
repeated `main_final` updates made direct answering much more likely. Width-table behavior
remained weak and produced most invalid/length-truncated Main actions.

The final step-48 checkpoint is not an RL-ready initialization. Increasing the same SFT exposure
again would likely worsen spawn collapse. The next experiment should evaluate a small,
predeclared set of already-saved intermediate checkpoints under the identical fixed profile to
locate the format-versus-spawn frontier. If no intermediate checkpoint passes all thresholds,
adding first-turn spawn-preservation rehearsal or a KL constraint requires an ADR update because
it changes the initial warm-start target semantics.

The ignored safe report SHA-256 values were:

- multi-step SFT report:
  `a578ecee808c5943b9dab9c7ef2556efb0d80f68e8a83b2ca0659bb71eb65add`
- post-SFT compliance report:
  `32d7f98cf106e2e7abcc811874577492587a318db73c3b71ed2fd40f9572a3da`

No prompt, reference answer, generated text, token array, retrieved content, checkpoint path,
credential, hostname, or service address is included in this record. The recorded launcher
released both services, closed the retrieval ports, and returned the GPUs to idle state.
