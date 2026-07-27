# Qwen3-4B + WideSeek runnable experiment

This is the operational source of truth for the current HeteroSpawn-RL experiment. A clean clone
plus the committed config runs verified assets through SFT, both RL topologies, held-out
evaluation, and one statistical comparison. Do not assemble the experiment from historical
validation clones or copy old commands from the development log.

## Fixed profile

- model: `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`;
- policy backend: project-owned LocalHF, NF4 QLoRA, FP16 compute, rank-8 adapters;
- context: 4,096 total tokens, at most 1,024 generated tokens;
- environment: pinned WideSeek-R1 tasks and offline Wiki-2018/Qdrant/E5 Search/Access;
- training: role-targeted shared SFT, shared-policy RL, and independent Main/Sub fresh
  alternating RL;
- evaluation: the same fixed 16 tasks, two rollouts per task and condition, no optimizer update,
  paired task-cluster bootstrap;
- hardware: one policy GPU and one retrieval GPU; no distributed optimizer.

The exact schedule is
[`configs/wideseek-qwen3-4b-2080ti.json`](../../configs/wideseek-qwen3-4b-2080ti.json).
Changing it creates a different experiment identity; it is never an implicit resume.

## 1. Clean clone and preflight

Use Linux, Python 3.11, Git, at least two idle CUDA GPUs, and about 180 GB of asset space.
The examples map physical GPU 0 to the policy process and physical GPU 1 to E5 retrieval.

```bash
export HS_REPO="$HOME/HeteroSpawn-RL"
export HS_RUNTIME="$HOME/heterospawn-runtime"

git clone https://github.com/RidenShogunei/HeteroSpawn-RL.git "$HS_REPO"
cd "$HS_REPO"
git status --short --branch
git rev-parse HEAD

python3 scripts/remote_preflight.py \
  --output "$HS_RUNTIME/preflight/report.json" \
  --require-gpu-count 2
nvidia-smi
```

The checkout must be clean. Runtime reports belong below `$HS_RUNTIME`, not in Git.

## 2. Isolated environments

Policy environment:

```bash
python3.11 -m venv "$HS_RUNTIME/policy/.venv"
source "$HS_RUNTIME/policy/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install \
  transformers==4.53.3 peft==0.14.0 accelerate==1.2.1 \
  bitsandbytes==0.45.5 safetensors==0.5.2 numpy==1.26.4
python -m pip install -e "$HS_REPO[dev,qlora,wideseek]"
heterospawn --help
deactivate
```

Retrieval environment:

```bash
python3.11 -m venv "$HS_RUNTIME/retrieval/.venv"
source "$HS_RUNTIME/retrieval/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install \
  fastapi numpy qdrant-client sentence-transformers tqdm uvicorn
deactivate

git clone https://github.com/RLinf/RLinf.git "$HS_RUNTIME/retrieval/RLinf"
git -C "$HS_RUNTIME/retrieval/RLinf" \
  checkout d9f3d8a9db4d7aad1d641029293295503dd3eb2c
```

RLinf supplies only the pinned upstream retrieval-server file. Its training runtime is not used.
Never mix the policy and retrieval environments.

## 3. Fetch and verify assets

```bash
cd "$HS_REPO"
source "$HS_RUNTIME/policy/.venv/bin/activate"

heterospawn wideseek-fetch-assets \
  --manifest manifests/qwen3-4b.json \
  --destination "$HS_RUNTIME/models/Qwen3-4B" \
  --endpoint auto
heterospawn wideseek-fetch-assets \
  --manifest manifests/wideseek-train-data.json \
  --destination "$HS_RUNTIME/wideseek/train-data" \
  --endpoint auto
heterospawn wideseek-fetch-assets \
  --manifest manifests/wideseek-wiki-2018-corpus.json \
  --destination "$HS_RUNTIME/wideseek/wiki-2018-corpus" \
  --endpoint auto
heterospawn wideseek-fetch-assets \
  --manifest manifests/wideseek-e5-base-v2.json \
  --destination "$HS_RUNTIME/wideseek/e5-base-v2" \
  --endpoint auto
```

`auto` tries the official Hugging Face endpoint first, then `hf-mirror.com` after bounded
connection, timeout, or 5xx retries. Authentication errors, missing revisions, and digest
mismatches are hard failures. Assets copied from another machine must pass the same command with
`--verify-only`.

Credential-safe data inspection:

```bash
for split in width_20k depth_20k hybrid_20k; do
  heterospawn wideseek-inspect-data \
    --data-dir "$HS_RUNTIME/wideseek/train-data" \
    --split "$split"
done
```

## 4. Start and verify offline retrieval

In a dedicated terminal:

```bash
cd "$HS_REPO"
export HS_RUNTIME="$HOME/heterospawn-runtime"
export CUDA_VISIBLE_DEVICES=1
export PATH="$HS_RUNTIME/policy/.venv/bin:$PATH"
export HETEROSPAWN_WIKI_DIR="$HS_RUNTIME/wideseek/wiki-2018-corpus"
export HETEROSPAWN_E5_DIR="$HS_RUNTIME/wideseek/e5-base-v2"
export HETEROSPAWN_RLINF_DIR="$HS_RUNTIME/retrieval/RLinf"
export HETEROSPAWN_RUNTIME_DIR="$HS_RUNTIME/wideseek/service"
export HETEROSPAWN_RETRIEVAL_PYTHON="$HS_RUNTIME/retrieval/.venv/bin/python"

bash scripts/start_wideseek_offline.sh
```

Leave it running. In the policy terminal:

```bash
cd "$HS_REPO"
source "$HS_RUNTIME/policy/.venv/bin/activate"

heterospawn wideseek-check-environment \
  --corpus-dir "$HS_RUNTIME/wideseek/wiki-2018-corpus" \
  --retriever-dir "$HS_RUNTIME/wideseek/e5-base-v2"
heterospawn wideseek-rollout-smoke \
  --report "$HS_RUNTIME/results/environment/rollout-smoke.json"
```

Do not train if Qdrant is unhealthy, Access is empty, or a revision check fails.

## 5. Run the complete experiment

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HS_RUN="$HS_RUNTIME/results/wideseek-qwen3-4b-2080ti-v1"

heterospawn wideseek-run \
  --config configs/wideseek-qwen3-4b-2080ti.json \
  --run-dir "$HS_RUN" \
  --model-path "$HS_RUNTIME/models/Qwen3-4B" \
  --model-manifest manifests/qwen3-4b.json \
  --data-manifest manifests/wideseek-train-data.json \
  --data-dir "$HS_RUNTIME/wideseek/train-data" \
  --device cuda:0
```

The command owns this sequence:

```text
SFT shared checkpoint
├── shared RL cycle(s) ────────> held-out shared evaluation
├── independent Main/Sub cycle(s) -> held-out independent evaluation
└──────────────────────────────> held-out SFT evaluation
                                  └─> paired comparison
```

All branches start from the same SFT checkpoint. Later independent cycles restore the complete
Main/Sub checkpoint pair, including separate optimizer states. The final safe outputs are:

```text
$HS_RUN/state.json
$HS_RUN/report.json
$HS_RUN/reports/sft.json
$HS_RUN/reports/shared-cycle-*.json
$HS_RUN/reports/independent-cycle-*.json
$HS_RUN/reports/eval-*.json
$HS_RUN/reports/comparison.json
```

Checkpoints, transactions, raw environment logs, and reports stay in the ignored runtime. Safe
aggregate validation summaries can be written separately only after manual redaction review.

## 6. Resume and recover

After a normal interruption between committed stages, rerun the exact command with `--resume`.
The runner verifies the saved config digest, report digests, and checkpoint identities before
continuing:

```bash
heterospawn wideseek-run \
  --resume \
  --config configs/wideseek-qwen3-4b-2080ti.json \
  --run-dir "$HS_RUN" \
  --model-path "$HS_RUNTIME/models/Qwen3-4B" \
  --model-manifest manifests/qwen3-4b.json \
  --data-manifest manifests/wideseek-train-data.json \
  --data-dir "$HS_RUNTIME/wideseek/train-data" \
  --device cuda:0
```

If the process died inside a training phase, the outer runner fails closed. Inspect
`$HS_RUN/transactions` and recover that exact transaction without replaying its rollout:

```bash
heterospawn wideseek-recover-phase \
  --transaction-id "<experiment:cycle:phase>" \
  --transaction-dir "$HS_RUN/transactions" \
  --model-profile qwen3-4b \
  --model-path "$HS_RUNTIME/models/Qwen3-4B" \
  --model-manifest manifests/qwen3-4b.json \
  --device cuda:0 \
  --max-sequence-length 4096 \
  --max-new-tokens 1024 \
  --artifact-dir "$HS_RUN/checkpoints/recovery" \
  --report "$HS_RUN/reports/recovery.json"
```

Do not immediately rerun the incomplete cycle: v1 does not silently adopt a partially completed
cycle into `state.json`. Preserve the recovery report and transaction directory for operator
review. A stage-boundary interruption is automatically resumable; a mid-phase interruption is
durably recoverable but intentionally requires reconciliation. Never delete transaction records,
replay an optimizer step manually, or edit `state.json`.

## 7. Interpreting the result

`report.json` means the configured pipeline completed and every stage contract passed. It does
not mean reward improved. `reports/comparison.json` contains SFT-relative deltas and paired 95%
intervals for outcome, format, spawn, failure, truncation, and tool-use metrics. A quality claim
requires the interval and preregistered metric to support it; a successful systems run alone is
not such a claim.

The default config uses no external Judge and spends no API credits. If a future config selects
`minimax-development`, provide the credential only through the process environment and add
`--allow-network`. That result remains non-official.

## 8. Repository and safety rules

- Use the latest clean `main`; do not run from the old dirty `$HOME/HeteroSpawn-RL` checkout or a
  historical per-experiment clone.
- Do not commit model/data assets, checkpoints, runtime state, prompts, reference answers,
  retrieved content, token arrays, raw generations, host details, or credentials.
- Do not change task selection, rollout count, sampling, tool budgets, or evaluator settings for
  one condition only.
- `wideseek-sft-train`, `wideseek-train-cycle`, `wideseek-compliance-baseline`, and
  `wideseek-recover-phase` are diagnostic stage commands. They are not a substitute for the
  canonical experiment.
- Stop the retrieval launcher with `Ctrl-C`; it terminates its Qdrant and retrieval child groups.
