# Qwen3-4B held-out SFT and RL comparison

Date: 2026-07-27

## Verdict

The optimizer-free comparison completed for all three preregistered conditions:

1. shared SFT step 48;
2. shared-policy RL step 49;
3. independent Main step 49 and Sub step 49.

All 96 episodes passed the exact-trajectory, stable-event-order, unchanged-weight-version, and
unchanged-adapter checks. No evaluation condition performed an optimizer update.

The run does **not** establish a model-quality improvement from one RL cycle. Shared RL improved
several rate point estimates but slightly reduced mean outcome. Independent RL improved format
compliance while reducing non-zero-outcome rate and mean outcome. Every quality delta's 95%
task-cluster bootstrap interval touched or crossed zero.

This is a diagnostic fixed-profile comparison, not an official WideSeek benchmark result.

## Preregistered protocol

- Model: `Qwen/Qwen3-4B` at
  `1cfa9a7208912126459214e8b04321603b3df60c`
- Verified model-manifest identity:
  `7d1f0f5002a45efc43d4800d460e1e11d9849dee17a587a34d003709c5e57ce9`
- Dataset revision: `47832ea20581f78d32cd6b32b4b37b985cbbc9df`
- Fixed selection-profile digest:
  `52b65c562a2673dd7e96f6584ee6a35af0a9a6210bbd49e2209aa05a771cd5a5`
- Selection: 16 SFT-disjoint held-out tasks: 6 width, 5 depth, and 5 hybrid
- Sampling: two raw-policy rollouts per task and condition, 32 episodes per condition
- Context: 4,096 tokens; maximum generation: 1,024 tokens
- Tool display budgets: 3 Search results, 600 Search-content characters, and 800 Access
  characters
- Semantic Judge: disabled
- Training during evaluation: disabled
- Hardware: one RTX 2080 Ti for policy rollout and one RTX 2080 Ti for E5 retrieval; conditions
  ran serially
- Preregistration SHA-256:
  `3484ed2efecfe5d0ebe476c5cb54f7fca773b63bb6610f07abe506e1036bd9f6`

The same dataset, selection, evaluator revisions, environment revision, sampling parameters,
sequence limits, and tool budgets were present in all three reports.

## Checkpoint identities

| Condition | Policy checkpoint identity |
|---|---|
| SFT step 48 | shared `07029f5df3016006085288fc81d23e8d359202b9989be1ceff8d070f7521ec44` |
| Shared RL step 49 | shared `717d2e773181976f107da46b1eb8fbfb73c2791888ba877abe5d8a87885e4e23` |
| Independent RL step 49 | Main `2dd044c5469d8f3466302fd0f88264d356c546a3eebb85c9a101def0c8cbe76b`; Sub `d92dae5e2c78784f0311a142fd2737cf752ce8b229e681b91170d06b4cd8ab01` |

The independent loader verified the Main and Sub policy IDs separately and published a separate
rollout revision for each checkpoint.

## Aggregate results

| Metric | SFT 48 | Shared RL 49 | Independent RL 49 |
|---|---:|---:|---:|
| Episodes | 32 | 32 | 32 |
| Success rate | 84.38% | 96.88% | 84.38% |
| Required-format rate | 68.75% | 78.13% | 75.00% |
| Non-zero-outcome rate | 37.50% | 43.75% | 31.25% |
| Mean outcome | 0.2482 | 0.2300 | 0.1927 |
| Spawn rate | 34.38% | 31.25% | 31.25% |
| Invalid-Main episode rate | 18.75% | 12.50% | 15.63% |
| Failed-Sub episode rate | 0.00% | 3.13% | 3.13% |
| Length-truncated episode rate | 18.75% | 18.75% | 15.63% |
| Model steps | 71 | 71 | 71 |
| Tool calls | 11 | 14 | 13 |
| Total sequence tokens | 67,313 | 70,298 | 60,852 |
| Sequence-token p95 / max | 2,808 / 3,287 | 2,854 / 3,908 | 2,564 / 3,295 |
| Rollout elapsed seconds | 1,158.17 | 1,167.65 | 941.54 |
| Peak allocated VRAM | 4.85 GiB | 5.04 GiB | 4.89 GiB |

## Task-cluster bootstrap deltas

Each task's two rollouts were averaged first. The analysis then resampled the 16 task identities
10,000 times with replacement using seed `20260727`. Intervals below are percentile 95%
intervals for candidate minus SFT step 48.

| Metric | Shared RL delta [95% interval] | Independent RL delta [95% interval] |
|---|---:|---:|
| Mean outcome | -0.0182 [-0.1990, 0.1317] | -0.0555 [-0.2194, 0.0808] |
| Required-format rate | +9.38 pp [-6.25, 25.00] | +6.25 pp [-6.25, 18.75] |
| Non-zero-outcome rate | +6.25 pp [-15.63, 28.13] | -6.25 pp [-28.13, 12.50] |
| Spawn rate | -3.13 pp [-18.75, 9.38] | -3.13 pp [-12.50, 6.25] |
| Success rate | +12.50 pp [0.00, 28.13] | 0.00 pp [0.00, 0.00] |
| Invalid-Main rate | -6.25 pp [-15.63, 0.00] | -3.13 pp [-9.38, 0.00] |
| Failed-Sub rate | +3.13 pp [0.00, 9.38] | +3.13 pp [0.00, 9.38] |
| Length-truncation rate | 0.00 pp [-9.38, 9.38] | -3.13 pp [-9.38, 0.00] |
| Tool calls per episode | +0.0938 [-0.1250, 0.3438] | +0.0625 [-0.0938, 0.2500] |

The intervals are descriptive uncertainty for this fixed profile. They are not a claim about the
full WideSeek task distribution.

## Contract checks and recovery finding

Every final report recorded:

- `optimizer_updates: 0`;
- exact response-token and old-log-probability alignment;
- contiguous deterministic event indices;
- unchanged adapter hashes;
- unchanged `WeightVersion` values;
- the same environment revision
  `2b90af41266aafdbf27af8743f7a80700567afaa24ec5993961b39150c193340`.

The first shared-RL restore attempt failed before rollout because its legacy checkpoint had been
created while nine CUDA devices were visible, while the controlled evaluation exposed one.
`set_rng_state_all` attempted to restore all nine logical-device entries and raised an index
error. No report, model action, or optimizer update was produced by that failed attempt.

The LocalHF backend was corrected to select only the configured logical device from legacy
multi-device RNG arrays and to save only the configured device's RNG state in new checkpoints.
The default suite passed with 150 tests and one optional module skipped. In the isolated QLoRA
environment, three real tiny-backend checkpoint/fork/restore tests also passed. The completed SFT
report was retained by digest; the shared and independent conditions then resumed under the fixed
source without repeating SFT sampling.

## Interpretation

- One bounded shared-policy update changed behavior, but this profile does not show a reliable
  reward-quality gain.
- One fresh-alternating independent cycle does not yet outperform the shared SFT checkpoint or
  the shared-RL checkpoint. Its lower mean outcome and non-zero rate are negative point estimates,
  although uncertainty is wide.
- Format compliance is no longer the only bottleneck. Exact answer quality, spawn retention, and
  Sub reliability remain unresolved.
- The next research run should use more than one RL update and retain fixed held-out evaluation
  checkpoints. Before making a quality claim, use a larger held-out profile in addition to this
  16-task regression gate.

## Artifact digests

Ignored machine-readable artifacts remain outside Git:

| Artifact | SHA-256 |
|---|---|
| SFT step-48 report | `4a992a1a163fcd3ca9ee1cd98585cd5461236b041957d379cc2c287fd858e3d3` |
| Shared-RL step-49 report | `913c66af3a4436fb8890df7ea3a433d0d18c90c5a3a1aac07ba1d45664f760d0` |
| Independent-RL step-49 report | `3bc6ce3767123135cca763e01a08143f0c6c7a2650d08688a7c9082cdf337719` |
| Cluster-bootstrap comparison | `11be72b73adeaf889a53e21aed35f84922a4bf92de129be302f0c34a0ff633c9` |

No model text, reference answer, retrieved content, token array, credential, hostname, or
checkpoint file is tracked.
