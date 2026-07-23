# Trainable Main/Sub episode cycle

Date: 2026-07-23

## Outcome

The first complete trainable `Main → Sub → Main` system cycle passed on the project-owned LocalHF
trainer and standalone vLLM rollout backend. Two full rollouts were generated for the same task in
each phase:

1. `M1/S1 rollout → Main update → Main sync`
2. `M2/S1 fresh rollout → Sub update → Sub sync`

This validates architecture and data semantics. It is not a benchmark result: the contract task
uses phase-specific regex-constrained decoding and an explicitly non-scientific synthetic reward
to ensure both optimizer paths execute deterministically.

## Runtime

- Source commit: `d0d0147`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Model revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- Model weight SHA-256:
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`
- Training: LocalHF, PyTorch `2.5.1+cu124`, PEFT `0.14.0`
- Rollout: vLLM `0.7.0`, V0, FP16 base, eager mode, XFormers `0.0.28.post3`
- Supporting pin: Transformers `4.48.1`
- Hardware allocation: one RTX 2080 Ti for training and one independently selected RTX 2080 Ti
  for each policy's rollout worker
- Search: deterministic local mock; no API credential or benchmark data was used

The isolated runtime used the exact source commit above. Generated text, token arrays, checkpoints,
worker logs, runtime paths, and machine identifiers remain in ignored artifacts.

## Contract results

| Contract | Result |
|---|---|
| Two complete system rollouts per task and phase | PASS |
| Initial Main performs exactly one legal spawn | PASS |
| Main/Sub/final Main exact response IDs and selected-token old log-probabilities | PASS |
| Raw trajectory values copied into training samples without decode/re-encode | PASS |
| Task-level system reward normalization | PASS |
| Episode-balanced aggregation weights | PASS |
| Main optimizer update and rollout synchronization | PASS |
| Sub phase uses a new rollout ID and the synchronized Main revision | PASS |
| Sub optimizer update and rollout synchronization | PASS |
| Main and Sub retain independent weight and rollout revisions | PASS |
| Pre-sync old revision service and post-sync stale revision rejection | PASS |
| Checkpoint restore | PASS |
| All owned workers exit and release their GPUs | PASS |

The Main phase produced four Main samples: initial action plus final action for each of two
episodes. The Sub phase produced two Sub samples. Both task groups had non-degenerate normalized
advantages, and neither phase contained an invalid Main action.

## Findings and fixes

The first diagnostic run found that PEFT's first adapter was FP32 while later-created adapters
inherited FP16 from the base model. The first Sub AdamW step could therefore publish non-finite
adapter weights, observed as NaN rollout log-probabilities after synchronization.

The reference trainer now keeps every LoRA train/rollout adapter in FP32 while the frozen base
remains FP16. It also rejects non-finite loss, gradients, or post-step adapter weights before
creating a checkpoint. A tiny-Qwen test verifies both Main and Sub exported adapters are finite
FP32 tensors over an FP16 base.

The next run proved numerical stability but also showed that a 0.5B model does not reliably choose
SPAWN from prompting alone. The vLLM compatibility layer therefore gained optional auditable regex
constrained decoding, and the orchestrator gained phase-specific sampling parameters. These are
enabled only by the deterministic smoke; the default training path remains unconstrained.

## Resources and evidence

- End-to-end elapsed time: `167.50` seconds
- Main worker at final measurement: `5,713.88 MiB` device use; `5,309.45 MiB` peak PyTorch
  allocation; `5,484.00 MiB` peak reservation
- Sub worker at final measurement: `5,623.88 MiB` device use; `5,291.18 MiB` peak PyTorch
  allocation; `5,396.00 MiB` peak reservation
- Credential-safe report schema: `2`
- Report SHA-256:
  `28f4dd142c31f137013407c9aeb9a0eece2789a3af6480153a3a93bef67867ab`
- Report checks: all `24` passed

After the run, the selected GPUs returned to their idle baseline and no owned rollout worker
remained. The same isolated environment passed four LocalHF/tiny-Qwen tests and twelve
episode/vLLM state-machine tests before the final GPU run.

## Boundaries

- The outcome reward is a contract fixture, not xbench scoring and not evidence of policy quality.
- Regex constrained decoding proves pipeline execution, not unconstrained instruction following.
- Main and Sub share one frozen base architecture and use independent LoRA policies.
- vLLM remains rollout-only; training, batching, advantages, checkpointing, and version
  transitions remain project-owned.
- This does not complete the deferred multi-node distributed-backend milestone.
