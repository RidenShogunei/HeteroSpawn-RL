# RLinf native capability spike (BLOCKED)

Date: 2026-07-23

## Purpose

Evaluate unmodified RLinf native generation / update / sync / checkpoint capability on one idle RTX 2080 Ti against the HeteroSpawn Milestone 2.5 contract matrix. No credentials or xbench plaintext were used.

## Pins

- Upstream commit: `0f9ea98c7a6d9e3ade24e8f4846c64d3b135dbcc`
- Package: `rlinf==0.3.0`
- Install: UV `requirements/install.sh agentic --no-root --no-flash-attn --install-rlinf --use-mirror` into `$HOME/heterospawn-runtime/rlinf/.venv`
- Python 3.11.14; PyTorch `2.6.0+cu124`; vLLM `0.8.5`; SGLang `0.4.6.post5`
- Docker image: not used (math image unavailable/unused; UV agentic stack used)
- Model: `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775`
- Model weight SHA-256: `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`
- GPUs: 1 (`CUDA_VISIBLE_DEVICES=0`); not expanded to two

## Commands (abstracted)

1. Preflight inventory via `scripts/remote_preflight.py`
2. Isolated agentic UV install under `$HOME/heterospawn-runtime/rlinf/`
3. Memory-scaled single-GPU GRPO config derived from official math FSDP/single-GPU examples
4. `CUDA_VISIBLE_DEVICES=0 python examples/reasoning/main_grpo.py --config-name qwen2.5-0.5b-grpo-single-gpu-spike`

## Contract results

| Experiment | Status |
|---|---|
| native_startup | blocked (SGLang rollout initialized; FSDP actor failed) |
| exact_trajectory | blocked |
| one_update_and_sync | blocked |
| checkpoint_resume | blocked |
| two_policy_feasibility | blocked (not reached) |

## Blocker

Stop condition: pinned native example cannot run on Turing / RTX 2080 Ti without unreviewed RLinf patches.

- RLinf vLLM worker hardcodes `VLLM_USE_V1=1`, which raises on compute capability `< 8.0`
- RLinf FSDP model manager hardcodes `attn_implementation="flash_attention_2"` while FlashAttention2 is unavailable/inappropriate on sm_75
- Megatron Transformer Engine build failed against host CUDA toolkit 11.5 (`requires CUDA 12.0 or newer`)
- No HeteroSpawn core or RLinf source patches were applied

Partial observation: SGLang rollout worker did initialize and allocate KV cache on the selected GPU before actor init failed.

## Resources

- Observed wall time to failure ~53 s for the SGLang attempt
- Peak GPU memory for a completed update step: not measured (blocked before training step)
- Machine-readable report digest: `dfff9974baeed3791513af0ffcd6c4a425a8b2fe253b421150b639b788e3d1c5`

This is a capability-spike record, not a backend-selection ADR.
