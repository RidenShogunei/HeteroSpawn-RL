# verl native capability spike (BLOCKED)

Date: 2026-07-23

## Purpose

Evaluate unmodified verl native generation / update / sync / checkpoint capability on one idle RTX 2080 Ti against the same HeteroSpawn Milestone 2.5 contract matrix used for RLinf. Ran only after RLinf teardown. No credentials or xbench plaintext were used.

## Pins

- Upstream commit: `a35908ca3c9632859c58d6a2855d858918ae21dc`
- Package: `verl==0.9.0.dev0`
- Install: isolated UV venv under `$HOME/heterospawn-runtime/verl/` with `torch==2.6.0+cu124`, editable verl, and `requirements.txt`
- Python 3.11.14; vLLM `0.8.5`; SGLang `0.4.6.post5`; Ray `2.56.1`; TRL `0.16.1`
- Docker: `verlai/verl:vllm017.latest` pull failed (`registry EOF`); no image digest recorded
- Model: `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775`
- Model weight SHA-256: `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`
- GPUs: 1 (`CUDA_VISIBLE_DEVICES=0`); not expanded to two

## Commands (abstracted)

1. Preflight inventory via `scripts/remote_preflight.py`
2. Isolated UV install under `$HOME/heterospawn-runtime/verl/`
3. Synthetic tiny GSM8K-like parquet under the runtime data directory
4. Memory-scaled single-step GRPO: `python -m verl.trainer.main_ppo ... algorithm.adv_estimator=grpo trainer.n_gpus_per_node=1 trainer.total_training_steps=1` with `VLLM_USE_V1=0`

## Contract results

| Experiment | Status |
|---|---|
| native_startup | blocked (config+Ray init only) |
| exact_trajectory | blocked |
| one_update_and_sync | blocked |
| checkpoint_resume | blocked |
| two_policy_feasibility | blocked (not reached) |

## Blocker

Stop condition: pinned native example cannot run on Turing / RTX 2080 Ti under available official install paths.

- Official quickstart states a GPU with at least 24 GB HBM; this host provides 11 GB
- Official install states CUDA >= 12.8; host toolkit remains 11.5 (driver advertises newer compatibility)
- Official Docker pull failed; spike continued with isolated UV as documented fallback
- After dependency alignment, worker init failed with `ImportError: cannot import name 'run_headless' from vllm.entrypoints.cli.serve` (vLLM 0.8.5 vs this verl commit)
- SGLang path blocked separately: `sgl_kernel` fails to load for compute capability 7.5
- `load_valuehead_model` still hardcodes FlashAttention2 for critic/value-head paths
- No HeteroSpawn core or verl source patches were applied

Partial observation: Hydra config validation and Ray cluster startup succeeded once OpenTelemetry pins were aligned.

## Resources

- Observed wall time to failure ~50 s for the final vLLM attempt
- Peak GPU memory for a completed update step: not measured
- Machine-readable report digest: `2d1b3e40747093f7560068d9e17888091b091ea884e25a188e5b902280e07354`

This is a capability-spike record, not a backend-selection ADR. Both candidates remain BLOCKED on this host profile.
