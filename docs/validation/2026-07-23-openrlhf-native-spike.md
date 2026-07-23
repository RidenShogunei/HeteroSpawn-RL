# OpenRLHF native capability spike (BLOCKED)

Date: 2026-07-23

## Purpose

Evaluate unmodified OpenRLHF native generation, update, sync, and checkpoint capability on one
idle RTX 2080 Ti against the same HeteroSpawn Milestone 2.5 contract matrix used for RLinf and
verl. The runtime was isolated; no credentials or xbench plaintext were used.

## Pins

- Upstream commit: `bc71bb19464aca306b33080b2d2bb45d154e2f49` (main tarball via
  codeload; GitHub clone timed out)
- Tarball SHA-256: `e6ece724569b733a311a20328fb60155d577fbe2f26f32130e763b83cf7a852b`
- Package: `openrlhf==0.10.4`
- Install: isolated UV environment under `$HOME/heterospawn-runtime/openrlhf/`; requirements
  without `flash-attn`, editable install with `--no-deps`, then the official `vllm==0.22.1` pin
- Python 3.11.14; PyTorch `2.11.0+cu130`; vLLM `0.22.1`; DeepSpeed `0.19.1`;
  Ray `2.55.0`; Transformers `4.57.6`
- Docker/NGC PyTorch: not used
- Model: `Qwen/Qwen2.5-0.5B-Instruct` revision
  `7ae557604adf67be50417f59c2c2f167def9a775`
- Model weight SHA-256:
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`
- GPUs: one; not expanded to two

## Commands (abstracted)

1. Run `scripts/remote_preflight.py`.
2. Install OpenRLHF and its official vLLM pin in an isolated UV environment.
3. Prepare a single-GPU colocated REINFORCE++/PPO probe using FP16, SDPA, and tiny local
   synthetic prompts.
4. Record the import failure and run a standalone vLLM sm_75 probe as secondary evidence.

## Contract results

| Experiment | Status |
|---|---|
| Native startup | BLOCKED at CLI import |
| Exact trajectory | BLOCKED |
| One update and sync | BLOCKED |
| Checkpoint resume | BLOCKED |
| Two-policy feasibility | BLOCKED; not reached |

## Blocker

The stop condition was that the pinned native example could not run on Turing without unreviewed
OpenRLHF or environment patches.

- `requirements.txt` pins `flash-attn==2.8.3`.
- `ring_attn_utils.py` imports `flash_attn.bert_padding` at module import time, before
  `--ds.attn_implementation sdpa` can take effect.
- No matching FlashAttention wheel was available for the installed Python/PyTorch/CUDA
  combination. Source builds failed against both the host toolkit and the isolated CUDA compiler
  because of toolkit/header incompatibility.
- A skip-build FlashAttention package could not import `flash_attn_2_cuda`.
- OpenRLHF sets `VLLM_USE_V1=1` in its vLLM engine.
- Standalone vLLM 0.22.1 also failed EngineCore initialization on sm_75 while JIT-compiling its
  FlashInfer path.
- No HeteroSpawn core or OpenRLHF source patch was applied.

## Resources

- Native training loop: not reached
- Peak GPU memory for an update: not measured
- Machine-readable report digest:
  `7bcd3d8b662e959107d748bf75e76d08e88cf09514bf38c53367bafe4779de68`

This is a capability-spike record, not a backend-selection result. RLinf, verl, and OpenRLHF are
all blocked on this host profile.
