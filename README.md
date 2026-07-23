# HeteroSpawn-RL

HeteroSpawn-RL studies dynamic heterogeneous agent spawning with fresh-rollout alternating policy optimization for deep-research tasks.

The project is deliberately staged:

1. CPU-only domain contracts and mock orchestration.
2. API-first benchmark validation with xbench-DeepSearch and MiniMax.
3. Exact-token local-model rollout and independent Main/Sub training backends.

The authoritative architecture is [HeteroSpawn_DeepResearch_RL_Project_Design.md](HeteroSpawn_DeepResearch_RL_Project_Design.md). Significant decisions are recorded under `docs/adr/`; implementation work is linked to GitHub issues, milestones, commits, and pull requests.

## Development setup

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m mypy src
```

### Optional local single-GPU contract

Install CUDA PyTorch for the host first, then install the isolated local-model dependencies:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
.venv/Scripts/python -m pip install -e ".[dev,local]"
.venv/Scripts/heterospawn local-contract-smoke --allow-model-download
```

The smoke uses the pinned `Qwen/Qwen2.5-0.5B-Instruct` commit with FP16 and separate Main/Sub LoRA train and rollout adapters. Checkpoints and the credential-safe JSON report are written under ignored `artifacts/`. A previously downloaded model directory can be supplied with `--model-path`; its weight SHA-256 must match the pinned revision.

### Optional standalone vLLM rollout

Linux hosts with Turing GPUs can keep training in the project-owned LocalHF backend while moving
generation to isolated vLLM V0/XFormers workers:

```bash
uv venv "$HOME/heterospawn-runtime/vllm-product/.venv" --python 3.11
uv pip install \
  --python "$HOME/heterospawn-runtime/vllm-product/.venv/bin/python" \
  -e ".[dev,vllm-turing]"

"$HOME/heterospawn-runtime/vllm-product/.venv/bin/heterospawn" \
  vllm-rollout-contract-smoke \
  --model-path /absolute/path/to/Qwen2.5-0.5B-Instruct \
  --training-device cuda:3 \
  --main-rollout-device 1 \
  --sub-rollout-device 2
```

The conformance command intentionally uses one training GPU and one rollout GPU per independently
versioned policy. Each rollout worker receives an environment allowlist, an isolated home
directory, offline Hugging Face settings, and only a verified local base model plus immutable PEFT
LoRA artifact. Synchronization stops the old worker, loads and hashes the replacement, and
publishes a new `RolloutRevision` only after verification; a failed replacement rebuilds the
previous worker.

The pinned compatibility stack is an optional rollout-only dependency. It does not own optimizer
state, training batches, checkpoint recovery, or advantage semantics. See
[the product validation record](docs/validation/2026-07-23-vllm-product-rollout-contract.md).

### Remote backend capability spikes

Remote agents must follow [the remote backend spike runbook](docs/runbooks/remote-backend-spike.md) and run `python3 scripts/remote_preflight.py` before installing or evaluating verl/RLinf. The runbook keeps candidate environments isolated, forbids credentials and benchmark data, and defines the evidence required before a backend-selection ADR.

API keys are read only from environment variables. Copy `.env.example` to `.env` for local use, and never commit the resulting file. API-backed episodes are evaluation artifacts and are not eligible for RL training unless the policy backend supplies exact rollout token IDs, old log-probabilities, and an auditable rollout revision.

The first runnable slice is documented in [docs/benchmarks/xbench-deepsearch.md](docs/benchmarks/xbench-deepsearch.md). It keeps xbench ground truth behind the evaluator, uses MiniMax through the current OpenAI-compatible endpoint for policy calls and optional development judging, and keeps search behind a provider-neutral interface with deterministic mock, Tavily, and MiniMax Token Plan MCP backends.

## Current status

Architecture Baseline v0.2, the API-first benchmark slice, and the Milestone 2 CPU training
contracts are complete. Exact-token LocalHF training and optional restart-synchronized vLLM
rollout are validated reference paths; API-backed episodes continue to be explicitly
non-trainable.
