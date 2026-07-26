# Qwen3-4B WideSeek 4K output-budget ablation

- Date: 2026-07-26
- Checkpoint-only rollout contract: Passed
- Single-card 4K/1024 feasibility: Passed
- Structural-completion improvement: Observed
- Spawn-retention gate: Not passed
- Officially comparable: No
- Source commit used by the runner: `73cc9252c4e57e1bbc99861040951925ae891bac`

## Scope

This experiment restored the final shared-policy checkpoint from the 192-task SFT run and
reran the fixed 16-task WideSeek compliance profile with a 4,096-token total sequence limit and
`max_new_tokens=1024`. The checkpoint, initial RNG state, sampling parameters, task order,
Search/Access display budgets, dataset, evaluator, and offline environment matched the earlier
4K/512 run.

This is a stochastic budget ablation, not paired token-level replay. Once a generation continues
past the old 512-token boundary, it consumes additional RNG values and can change every later
sample in the serial run. Structural stop/repair measurements directly diagnose the budget;
per-task reward differences remain noisy at one rollout per task.

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
- Sampling: temperature `1.0`, top-p `1.0`, top-k `0`, initial seed `20260722`
- Search/Access display budgets: three results, 600 Search-content characters, and 800 Access
  characters

Before rollout, the launcher reverified all 3,400 corpus files and 155,895,995,164 bytes, all
nine E5 files, the 26,134,257-point green Qdrant collection, and a non-empty Search-to-Access
probe.

## Result

The 1024-token run passed exact response-token/log-probability alignment, stable event ordering,
unchanged evaluation weight versions, and unchanged train/rollout adapter hashes.

| Metric | 4K / 512 | 4K / 1024 |
|---|---:|---:|
| Successful episodes | 13 / 16 | 16 / 16 |
| Required format passed | 10 / 16 | 14 / 16 |
| Non-zero exact outcome | 4 / 16 | 5 / 16 |
| Mean exact outcome | 0.197917 | 0.094051 |
| Episodes with legal spawn | 6 / 16 | 7 / 16 |
| Episodes containing a length stop | 6 / 16 | 0 / 16 |
| Episodes with an invalid Main attempt | 5 / 16 | 0 / 16 |
| Episodes with a failed Sub | 2 / 16 | 0 / 16 |
| Model steps | 44 | 39 |
| Tool calls | 7 | 9 |
| Total response tokens | 9,840 | 6,452 |
| Total prompt + response tokens | 41,476 | 29,960 |
| Elapsed rollout time | 590.734 s | 480.588 s |
| Peak PyTorch-allocated policy memory | 5,170,221,568 B | 4,636,782,080 B |

The 512-token telemetry comes from the checkpoint-only 8K rerun whose behavior metrics exactly
matched the original 4K run and whose longest sequence was below 4K. The 4K/1024 run measured:

| Token count | p50 | p95 | max | total |
|---|---:|---:|---:|---:|
| Prompt | 476 | 1,302 | 1,463 | 23,508 |
| Response | 57 | 698 | 751 | 6,452 |
| Prompt + response | 629 | 1,996 | 2,161 | 29,960 |

No generation reached 1024 tokens and no complete sequence approached the 4096-token limit.
The old 512 cap interrupted incomplete actions; the orchestrator then retained the invalid
attempt and generated repairs. Giving those actions enough room to close removed all length
stops and invalid Main attempts, reduced the number of model steps, and reduced total generated
tokens despite doubling the nominal per-turn allowance.

By split:

| Split | 512 format | 1024 format | 512 non-zero | 1024 non-zero | 512 spawn | 1024 spawn |
|---|---:|---:|---:|---:|---:|---:|
| width | 1 / 6 | 4 / 6 | 0 / 6 | 2 / 6 | 4 / 6 | 5 / 6 |
| depth | 5 / 5 | 5 / 5 | 2 / 5 | 1 / 5 | 0 / 5 | 1 / 5 |
| hybrid | 4 / 5 | 5 / 5 | 2 / 5 | 2 / 5 | 2 / 5 | 1 / 5 |

The structural benefit is strong on the previously truncated width tasks. Outcome quality is not
monotonic: non-zero outcomes increased by one, but mean outcome fell because stochastic samples
lost two high-scoring answers. With only one rollout per task, this run does not establish a
reward improvement.

## Decision

The 512-token cap is too short for this checkpoint's WideSeek action distribution. A 1024-token
experimental profile is justified for the next repeated-rollout readiness measurement, while
the repository default remains 512 until an ADR accepts a new rollout-budget identity.

The next measurement should use at least two complete rollouts per fixed task for both budgets,
with independently declared seeds, and compare structural validity separately from reward. Even
with the structural improvement, only 7/16 episodes spawned, so the ADR-0006 12/16 spawn gate
still blocks direct-RL scaling from step 48. Output budget alone does not repair shared-policy
spawn collapse.

The ignored 4K/1024 compliance report SHA-256 is
`c03f55a8887302f78168bb624c337dc22227b7a20675ab4a4a15c0d2d4f895c5`.

No prompt, reference answer, generated text, token array, retrieved content, checkpoint path,
credential, hostname, or service address is included in this record. The launcher released both
services and returned all nine GPUs to idle state.
