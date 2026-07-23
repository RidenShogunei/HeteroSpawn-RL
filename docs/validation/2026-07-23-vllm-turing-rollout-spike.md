# Standalone vLLM Turing rollout spike

Date: 2026-07-23

## Outcome

Standalone vLLM is compatible with the RTX 2080 Ti host when pinned to its V0/XFormers path. This
result authorizes a future optional rollout engine; it does not select vLLM as a training backend
or complete Milestone 2.5.

## Pins

- vLLM `0.7.0`
- Python `3.11.14`
- PyTorch `2.5.1+cu124`
- Transformers `4.48.1`
- NumPy `1.26.4`
- XFormers `0.0.28.post3`
- PEFT `0.14.0` and Accelerate `1.2.1` for synthetic LoRA fixtures
- `VLLM_USE_V1=0`, FP16, eager execution, XFormers attention
- Model: `Qwen/Qwen2.5-0.5B-Instruct` revision
  `7ae557604adf67be50417f59c2c2f167def9a775`
- One explicitly selected idle RTX 2080 Ti (sm_75, 11 GB)

The isolated environment is under `$HOME/heterospawn-runtime/vllm-turing/`. The initial
unbounded dependency resolution selected Transformers 5.x, so the compatible Transformers and
NumPy versions above are mandatory pins rather than suggestions.

## Commands (abstracted)

1. Run the credential-safe remote preflight and select one idle GPU.
2. Create an isolated UV environment and install the pinned standalone vLLM stack.
3. Run a runtime-only probe with token-ID input and no adapter.
4. Create two deterministic synthetic LoRA revisions in the isolated runtime.
5. Run the same probe in two separate worker processes, loading one distinct adapter revision in
   each process.
6. Verify the per-run canonical digests and aggregate the redacted reports.

The originally selected GPU became occupied by an unrelated process before execution, so the
probe moved to another idle GPU without terminating or modifying that process.

## Results

| Contract | Result | Evidence |
|---|---|---|
| Native startup | PASS | vLLM rejected FA2 on Turing and selected XFormers automatically |
| Exact trajectory | PASS | Prompt token IDs round-tripped; each sampled response token had one finite selected-token log-probability |
| Dynamic 0/1/4 requests | PASS | Zero skipped GPU execution; one and four requests completed |
| LoRA loading | PASS | Two distinct PEFT adapters loaded through vLLM's Punica path |
| Adapter revision refresh | PASS with boundary | Separate worker restarts loaded distinct adapter digests |

No raw prompt, response text, or token array is present in the report. Token IDs and selected
log-probabilities are represented only by counts, alignment booleans, and SHA-256 digests.

## Resources

- Startup: 12.15 seconds without LoRA; 17.50 and 13.37 seconds for the two LoRA runs
- GPU memory after startup: 5,600–5,608 MiB
- GPU memory after generation: 5,686–5,696 MiB
- Base model weights reported by vLLM: about 0.93 GiB
- Aggregate report digest:
  `b024396c6761f6d43c719b49f678a2dd29fcfc311b3ec89d73abd4580a93fb65`

## Boundaries

- vLLM is rollout-only. Optimizer updates and checkpoint recovery remain in the project-owned
  PyTorch/Transformers/PEFT training backend.
- The first implementation will restart rollout workers after an adapter checkpoint changes,
  verify the adapter digest, and only then publish a new `RolloutRevision`.
- Online hot swap and stale-revision rejection were not validated here; the HeteroSpawn
  `PolicyService` adapter must enforce them.
- This older stack must remain an isolated optional dependency. It must not constrain the core or
  training environment.
- Modern V1/FlashInfer configurations remain blocked on this host.
