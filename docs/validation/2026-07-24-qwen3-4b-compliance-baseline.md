# Qwen3-4B WideSeek compliance baseline

- Date: 2026-07-24
- Rollout contract: Passed
- Direct-RL readiness: Not passed
- Recommended next gate: small tool/output-format warm-start SFT
- Officially comparable: No
- Source commit used by the runner: `076c1e2f5a66215269d96f20e1ed8e4a8279628a`

## Scope

This was a rollout-only diagnostic of the untrained Qwen3-4B QLoRA policy under the complete
offline WideSeek environment. It used the fixed `heterospawn-wideseek-compliance-v1` profile:
six answer-independent positions from `width_20k`, five from `depth_20k`, and five from
`hybrid_20k`, with one rollout per task.

The run used shared-policy topology only to measure the initial policy behavior. It performed
zero optimizer updates, made no Judge calls, and did not use the reference answer in the policy
or tool environment. The evaluator consumed references privately after each episode.

## Pinned identities

- Model: `Qwen/Qwen3-4B`
- Model revision: `1cfa9a7208912126459214e8b04321603b3df60c`
- Model-manifest digest:
  `7d1f0f5002a45efc43d4800d460e1e11d9849dee17a587a34d003709c5e57ce9`
- WideSeek data revision: `47832ea20581f78d32cd6b32b4b37b985cbbc9df`
- Offline environment revision:
  `2b90af41266aafdbf27af8743f7a80700567afaa24ec5993961b39150c193340`
- Corpus revision: `178d7d037f661be3159b0c3a8a4119b974f01880`
- Selection-profile digest:
  `52b65c562a2673dd7e96f6584ee6a35af0a9a6210bbd49e2209aa05a771cd5a5`
- Sampling seed: `20260722`

Before rollout, the launcher reverified 3,400 corpus files and 155,895,995,164 bytes plus all
nine E5 files. Qdrant was green with 26,134,257 points, and a real Search-to-Access probe
returned a non-empty page.

## Runtime profile

- Policy GPU: one NVIDIA GeForce RTX 2080 Ti
- Retrieval GPU: one separate NVIDIA GeForce RTX 2080 Ti
- Model loading: NF4 4-bit base, FP16 compute, gradient checkpointing enabled
- Context/generation limits: 4,096 / 512 tokens
- Thinking: disabled
- Sampling: raw policy, temperature 1, top-p 1, top-k 0
- Search display: at most three results and 600 content characters
- Access display: at most 800 characters

The 16 episodes completed in 558.677 seconds. PyTorch reported 6,174,190,592 peak allocated
policy bytes; device-level observation peaked at about 7.5 GiB. Services were terminated through
the recorded launcher PID afterward, ports 6333/8000 closed, and all nine GPUs returned to their
idle state.

## Results

| Split | Episodes | Successful | Format OK | Non-zero outcome | Tool calls |
|---|---:|---:|---:|---:|---:|
| width | 6 | 5 | 1 | 1 | 33 |
| depth | 5 | 5 | 0 | 0 | 12 |
| hybrid | 5 | 5 | 0 | 0 | 10 |
| total | 16 | 15 | 1 | 1 | 55 |

Additional observations:

- every episode legally spawned at least one Sub; 15 spawned one and one spawned two;
- the run created 17 Sub instances and 95 model steps;
- the only non-zero item-level F1 was `0.121212`, for a global mean of `0.007576`;
- six episodes contained at least one failed Sub, while five of those episodes still produced a
  successful Main answer;
- one width episode exhausted Main repair attempts after length-truncated invalid actions;
- no episode had a zero-spawn path, so the base policy already understands the spawn affordance.

All four rollout-only checks passed:

- actual response token IDs and per-token old log-probabilities remained aligned;
- event indices remained stable;
- registry and backend rollout revisions remained unchanged;
- train and rollout adapter hashes remained unchanged.

The ignored safe report SHA-256 was
`7d3585f5682b39d407b5ef54259a8e1035041d949b14034be44160126eebfa02`.

## Decision

The base policy can use the architecture: it consistently emits legal spawn actions and invokes
real Search/Access. Its reward-bearing answer behavior is not yet a good direct-RL starting point:
only 6.25% of episodes passed the required output format, and the same single episode was the only
one with non-zero exact outcome.

The next implementation should therefore be a small, versioned warm-start SFT focused on:

1. completing a Sub evidence summary instead of continuing tool use until the turn limit;
2. synthesizing Main evidence into the required Markdown table or boxed-answer format;
3. ending with a valid answer after Sub results return;
4. retaining the existing legal spawn and Search/Access behavior.

After SFT, rerun this exact fixed profile with the same seed and revisions. Proceed to a short RL
pilot only after format and non-zero-outcome coverage are no longer near-zero. Do not change the
accepted Sub reward or add local evidence advantage merely to create gradients.

No prompt, reference answer, generated text, token array, retrieved content, checkpoint,
credential, hostname, or service address is included in this record.
