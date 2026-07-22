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

API keys are read only from environment variables. Copy `.env.example` to `.env` for local use, and never commit the resulting file. API-backed episodes are evaluation artifacts and are not eligible for RL training unless the policy backend supplies exact rollout token IDs, old log-probabilities, and an auditable rollout revision.

The first runnable slice is documented in [docs/benchmarks/xbench-deepsearch.md](docs/benchmarks/xbench-deepsearch.md). It keeps xbench ground truth behind the evaluator, uses MiniMax through the current OpenAI-compatible endpoint for policy calls and optional development judging, and keeps search behind a provider-neutral interface with deterministic mock, Tavily, and MiniMax Token Plan MCP backends.

## Current status

Architecture Baseline v0.2, the API-first benchmark slice, and the Milestone 2 CPU training contracts are complete. Exact-token local-model rollout remains an opt-in backend: API-backed episodes continue to be explicitly non-trainable.
