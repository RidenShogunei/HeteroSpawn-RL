# Standalone vLLM product rollout contract

Date: 2026-07-23

## Outcome

The product `VllmRolloutBackend` passed an end-to-end contract run with the project-owned LocalHF
LoRA trainer on RTX 2080 Ti hardware. This validates vLLM as an optional rollout engine behind the
backend-neutral contracts. vLLM remains rollout-only and is not a training backend.

## Runtime

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Model revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- Model weight SHA-256:
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`
- Training: LocalHF, PyTorch `2.5.1+cu124`, PEFT `0.14.0`
- Rollout: vLLM `0.7.0`, V0, FP16, eager mode, XFormers `0.0.28.post3`
- Supporting pins: Transformers `4.48.1`, NumPy `1.26.4`
- Hardware allocation: one RTX 2080 Ti for LocalHF training and one independently selected RTX
  2080 Ti for each of the Main and Sub rollout workers

The run used a fresh isolated source directory and the pre-existing isolated vLLM environment.
It did not modify the separate framework-spike checkout, use API credentials, or access benchmark
data.

## Contract results

| Contract | Result |
|---|---|
| Exact response IDs and selected-token old log-probabilities | PASS |
| Main optimizer update changes Main but not Sub | PASS |
| Old Main worker serves the old revision before synchronization | PASS |
| Main restart publishes one new replica-set revision | PASS |
| Stale Main revision rejected after synchronization | PASS |
| Four concurrent Sub instances share one SubPolicy revision | PASS |
| Sub optimizer update changes Sub but not Main | PASS |
| Sub restart publishes one new replica-set revision | PASS |
| Stale Sub revision rejected after synchronization | PASS |
| Fresh generation succeeds through both replacement workers | PASS |
| Checkpoint restore preserves the trained Sub version | PASS |
| All owned worker processes exit and release their GPUs | PASS |

The final run produced 14 tokens for each initial/fresh Main response, 14 tokens for each of four
Sub batch responses, and 16 tokens through the replacement Sub worker. Only counts and alignment
booleans were retained; no prompt token arrays, response token arrays, decoded output, checkpoint,
hostname, address, username, or credential is committed.

## Synchronization and failure semantics

The training checkpoint is converted to an immutable `peft-lora-v1` artifact containing
`adapter_config.json` and `adapter_model.safetensors`. The artifact digest covers both files and is
rechecked against the source checkpoint on reuse.

For synchronization, the service blocks new work, waits for active generations, stops the old
worker, starts a replacement with the new artifact, checks the worker-reported artifact digest,
and only then publishes the new `RolloutRevision`. CPU fault-injection tests additionally prove
that a failed replacement rebuilds the old worker without publishing a revision and that active
generation is never interrupted by synchronization.

Each worker is launched with an environment allowlist, an isolated `HOME`, offline Hugging Face
settings, `VLLM_USE_V1=0`, and `VLLM_ATTENTION_BACKEND=XFORMERS`. API-key environment variables
are not inherited.

## Resources

The final credential-isolated run completed in 111.64 seconds, including two initial worker loads,
two optimizer updates, two replacement worker loads, all generations, synchronization, and
checkpoint restore.

Each replacement rollout worker reported:

- total GPU memory: 10,821.94 MiB
- device memory in use at measurement: 5,675.88 MiB
- peak PyTorch allocation: 5,295.19 MiB
- peak PyTorch reservation: 5,446.00 MiB

Four worker logs were produced in the ignored runtime: initial Main/Sub plus their verified
replacements. After completion, the three selected GPUs returned to their idle memory baseline
and no owned rollout worker remained.

The ignored machine-readable report SHA-256 is
`1b3addfcb6862f14d0132b913910b7f1d222167498170e5e5afe85edf75a83b6`.

After the GPU run, the same isolated Linux environment passed 10 CPU/tiny-Qwen tests covering the
LocalHF artifact exporter and the vLLM service state machine. This included exact/idempotent
artifact export, rejection after artifact corruption, four-request sharing, stale revision
rejection, active-generation synchronization, replacement rollback, selected-log-probability
extraction, and credential filtering.

## Boundaries

- The contract uses worker restart, not online LoRA hot swap.
- Main and Sub use the same frozen base-model architecture with independent LoRA policies.
- This validates project-owned single-node training/rollout semantics, not a modern distributed
  framework or Milestone 2.5 completion.
- RLinf, verl, and OpenRLHF remain blocked on this Turing host; reevaluation requires more suitable
  hardware and the common backend matrix.
