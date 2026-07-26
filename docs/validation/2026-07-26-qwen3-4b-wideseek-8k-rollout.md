# Qwen3-4B WideSeek checkpoint-only 8K rollout

- Date: 2026-07-26
- Checkpoint-only rollout contract: Passed
- Single-card 8K feasibility: Passed
- Evidence that 8K improves the fixed profile: Not observed
- Officially comparable: No
- Source commit used by the runner: `73cc9252c4e57e1bbc99861040951925ae891bac`

## Scope

This experiment restored the final shared-policy checkpoint from the 192-task SFT run and
reran the fixed 16-task WideSeek compliance profile with an 8,192-token total sequence limit.
It changed neither policy weights nor the 512-token per-generation limit. Raw-policy sampling,
seed, task order, tool budgets, dataset, evaluator, and offline Search/Access environment matched
the earlier 4,096-token run.

The new checkpoint-only path reconstructed `CheckpointRef` from the checkpoint manifest, verified
its canonical identity, let the backend verify every checkpoint file and base-model identity,
restored optimizer and RNG state, and explicitly synchronized the rollout adapter before the
first episode. The report contains the loaded immutable weight identity and performs zero
optimizer updates.

## Pinned identities

- Model: `Qwen/Qwen3-4B`
- Model revision: `1cfa9a7208912126459214e8b04321603b3df60c`
- Model-manifest digest:
  `7d1f0f5002a45efc43d4800d460e1e11d9849dee17a587a34d003709c5e57ce9`
- Loaded policy: `shared`
- Loaded optimizer step: `48`
- Loaded checkpoint digest:
  `3d853e12e5ec5cd1e9d4568ee01e194d16d287e97bc65c6abbc5e2df7f2c3e55`
- WideSeek data revision: `47832ea20581f78d32cd6b32b4b37b985cbbc9df`
- Offline environment revision:
  `2b90af41266aafdbf27af8743f7a80700567afaa24ec5993961b39150c193340`
- Compliance selection-profile digest:
  `52b65c562a2673dd7e96f6584ee6a35af0a9a6210bbd49e2209aa05a771cd5a5`
- Sampling: temperature `1.0`, top-p `1.0`, top-k `0`, seed `20260722`
- Search/Access display budgets: three results, 600 Search-content characters, and 800 Access
  characters

Before rollout, the launcher reverified all 3,400 corpus files and 155,895,995,164 bytes, all
nine E5 files, the 26,134,257-point green Qdrant collection, and a non-empty Search-to-Access
probe.

## Result

The 8K run passed exact response-token/log-probability alignment, stable event ordering,
unchanged `WeightVersion`/`RolloutRevision`, and unchanged train/rollout adapter hashes.

| Metric | 4K run | 8K run |
|---|---:|---:|
| Successful episodes | 13 / 16 | 13 / 16 |
| Required format passed | 10 / 16 | 10 / 16 |
| Non-zero exact outcome | 4 / 16 | 4 / 16 |
| Episodes with legal spawn | 6 / 16 | 6 / 16 |
| Tool calls | 7 | 7 |
| Mean exact outcome | 0.197917 | 0.197917 |
| Episodes containing a length stop | 6 / 16 | 6 / 16 |
| Elapsed rollout time | 590.734 s | 744.501 s |
| Peak PyTorch-allocated policy memory | 5,170,221,568 B | 5,162,522,112 B |

The v2 8K report measured 44 model steps:

| Token count | p50 | p95 | max | total |
|---|---:|---:|---:|---:|
| Prompt | 583 | 1,489 | 1,904 | 31,636 |
| Response | 90 | 512 | 512 | 9,840 |
| Prompt + response | 723 | 1,976 | 2,416 | 41,476 |

The longest complete model sequence was only 2,416 tokens, leaving 1,680 tokens of headroom even
under the old 4,096-token limit. The observed length stops therefore came from individual
responses reaching `max_new_tokens=512`, not from exhaustion of the total context window.

The small difference in peak allocated memory is measurement noise at this workload. The single
run's 26% longer elapsed time is not a throughput benchmark; no repeated timing trials were run.
The material conclusions are that one RTX 2080 Ti can execute this 8K configuration without OOM,
and that raising only the total context limit cannot improve this fixed profile because its
actual sequences do not reach 4K.

## Decision

Keep 4,096 as the current default for short validation and training. Do not spend multiple GPUs
on a larger total context window yet. The next token-budget experiment should increase or
role-specialize the per-turn generation allowance while preserving a bounded total context,
then measure whether width-table completion and legal tool-call envelopes improve. Changing the
accepted role-specific rollout budget or truncation semantics requires an ADR update before it
becomes a training default.

The ignored 8K compliance report SHA-256 is
`6d1c5d60304b12b7d9b079a517de5463b9d42e71edcda5d57a3f1562e7417afb`.

No prompt, reference answer, generated text, token array, retrieved content, checkpoint path,
credential, hostname, or service address is included in this record. The launcher released both
services and returned all nine GPUs to idle state.
