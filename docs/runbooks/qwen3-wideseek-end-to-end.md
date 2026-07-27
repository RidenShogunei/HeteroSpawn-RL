# Qwen3-4B + WideSeek end-to-end guide

This is the canonical operational guide for the current HeteroSpawn-RL experiment. It covers a
clean host through verified assets, offline retrieval, SFT, shared and independent RL, checkpoint
evaluation, and crash recovery.

## Scope

The supported research profile is:

- model: `Qwen/Qwen3-4B` at revision
  `1cfa9a7208912126459214e8b04321603b3df60c`;
- policy backend: project-owned LocalHF, NF4 QLoRA, FP16 compute, rank-8 adapters;
- context: 4,096 total tokens and at most 1,024 generated tokens;
- environment: pinned WideSeek-R1 tasks plus offline Wiki-2018/Qdrant/E5 Search/Access;
- topology: shared-policy joint update or independent Main/Sub fresh alternating updates;
- hardware: one policy GPU and one retrieval GPU; no distributed optimizer.

One `wideseek-train-smoke` invocation runs one complete cycle. Independent training can start by
explicitly forking one verified shared SFT checkpoint, but the command does not yet accept an
existing independent Main/Sub checkpoint pair for a second training cycle. Use independent
checkpoint pairs for evaluation, not for an undocumented multi-cycle continuation.

This workflow validates rollout, reward, update, synchronization, checkpoint identity, and
recovery. It is not a reproduction of the upstream distributed training stack or an official
WideSeek benchmark score.

## Host and storage requirements

Use Linux, Python 3.11, Git, and two idle CUDA GPUs. The validated host used RTX 2080 Ti cards:
one for Qwen3-4B and one for E5 retrieval. Do not run policy and retrieval in the same Python
environment.

Reserve at least 180 GB for verified assets, plus checkpoint and cache space. The pinned corpus
contains 3,400 files and about 156 GB. Its `wiki_webpages.jsonl` is about 26.6 GB and is indexed
in host memory by the upstream service. The launcher copies the verified Qdrant source with
`cp --reflink=auto`; a filesystem without copy-on-write needs space for another physical copy.

## 1. Clone and preflight

Use a clean checkout. Keep all large or mutable state outside it:

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

The preflight report is runtime evidence and must not be committed.

## 2. Create two isolated environments

Create the policy environment:

```bash
python3.11 -m venv "$HS_RUNTIME/policy/.venv"
source "$HS_RUNTIME/policy/.venv/bin/activate"
python -m pip install --upgrade pip
```

Install the CUDA build of PyTorch appropriate for the host, then install the project:

```bash
python -m pip install \
  torch==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install \
  transformers==4.53.3 \
  peft==0.14.0 \
  accelerate==1.2.1 \
  bitsandbytes==0.45.5 \
  safetensors==0.5.2 \
  numpy==1.26.4
python -m pip install -e "$HS_REPO[dev,qlora,wideseek]"
heterospawn --help
deactivate
```

Create the retrieval environment separately:

```bash
python3.11 -m venv "$HS_RUNTIME/retrieval/.venv"
source "$HS_RUNTIME/retrieval/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install \
  torch==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install \
  fastapi numpy qdrant-client sentence-transformers tqdm uvicorn
deactivate
```

RLinf is used only as the pinned source of the upstream retrieval server. Its training runtime is
not installed or invoked:

```bash
git clone https://github.com/RLinf/RLinf.git "$HS_RUNTIME/retrieval/RLinf"
git -C "$HS_RUNTIME/retrieval/RLinf" \
  checkout d9f3d8a9db4d7aad1d641029293295503dd3eb2c
```

## 3. Fetch and verify assets

Activate the policy environment and fetch every resource through its committed manifest:

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

`auto` tries the official Hugging Face endpoint first and switches to `hf-mirror.com` only after
bounded connection, timeout, or 5xx retries. Authentication errors, unknown revisions, and digest
mismatches are hard failures. If assets are copied from another machine, run the same commands
with `--verify-only`; never bypass the manifest.

Inspect all three task shapes without printing questions or reference answers:

```bash
for split in width_20k depth_20k hybrid_20k; do
  heterospawn wideseek-inspect-data \
    --data-dir "$HS_RUNTIME/wideseek/train-data" \
    --split "$split"
done

heterospawn wideseek-sft-dry-run \
  --data-dir "$HS_RUNTIME/wideseek/train-data" \
  --split hybrid_20k \
  --task-limit 8
```

## 4. Start the offline retrieval service

In a dedicated terminal, choose one physical GPU for E5. The example uses GPU 1:

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

The launcher verifies the pinned checkout and every corpus/retriever file, clones Qdrant into a
mutable runtime directory, starts Qdrant and the upstream Search/Access server, and stops both
child process groups when it exits. Verification of the 156 GB corpus on every start is expected.
Logs can contain retrieved text and must remain outside Git.

Leave that terminal running. In a second terminal, verify the environment:

```bash
cd "$HS_REPO"
export HS_RUNTIME="$HOME/heterospawn-runtime"
source "$HS_RUNTIME/policy/.venv/bin/activate"

heterospawn wideseek-check-environment \
  --corpus-dir "$HS_RUNTIME/wideseek/wiki-2018-corpus" \
  --retriever-dir "$HS_RUNTIME/wideseek/e5-base-v2"

heterospawn wideseek-rollout-smoke \
  --report "$HS_RUNTIME/results/environment/rollout-smoke.json"
```

Do not continue if the collection is unhealthy, Access returns no page, or any environment
revision or digest check fails.

## 5. Select the policy GPU

All following model commands run in the policy terminal. The example maps physical GPU 0 to
logical `cuda:0`:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HS_MODEL="$HS_RUNTIME/models/Qwen3-4B"
export HS_DATA="$HS_RUNTIME/wideseek/train-data"
export HS_RESULTS="$HS_RUNTIME/results/qwen3-wideseek"
mkdir -p "$HS_RESULTS"
```

The fixed model arguments used below are:

```text
--model-profile qwen3-4b
--model-path "$HS_MODEL"
--model-manifest manifests/qwen3-4b.json
--device cuda:0
--max-sequence-length 4096
--max-new-tokens 1024
```

Always pass them explicitly. `wideseek-train-smoke` retains a small 0.5B default for cheap
contract tests; relying on that default would run the wrong experiment.

## 6. Record the pre-training baseline

Run the fixed, answer-independent 16-task compliance profile with two stochastic rollouts per
task:

```bash
heterospawn wideseek-compliance-baseline \
  --topology shared \
  --rollouts-per-task 2 \
  --data-dir "$HS_DATA" \
  --model-profile qwen3-4b \
  --model-path "$HS_MODEL" \
  --model-manifest manifests/qwen3-4b.json \
  --device cuda:0 \
  --max-sequence-length 4096 \
  --max-new-tokens 1024 \
  --do-sample \
  --max-search-message-results 3 \
  --max-search-content-characters 600 \
  --max-access-characters 800 \
  --artifact-dir "$HS_RESULTS/base-eval/checkpoints" \
  --report "$HS_RESULTS/base-eval/report.json"
```

Compliance performs zero optimizer updates. It fails if token/log-probability alignment, stable
event ordering, or unchanged policy identities fail.

## 7. Run the bounded SFT warm start

The pinned training files contain task answers and format metadata, not native Main/Sub
trajectories. The constructor therefore supervises only:

- a Main final answer after a legal spawn and ordered worker results;
- a Sub evidence summary after a legal Search-to-Access history.

It never supervises spawn count, decomposition, query selection, or source selection.

Run the declared 192-task, 48-step shared-policy warm start:

```bash
heterospawn wideseek-sft-train \
  --split hybrid_20k \
  --task-limit 192 \
  --tasks-per-step 4 \
  --epochs 1 \
  --training-max-sequence-length 2304 \
  --max-sequence-length 4096 \
  --max-new-tokens 1024 \
  --data-dir "$HS_DATA" \
  --model-profile qwen3-4b \
  --model-path "$HS_MODEL" \
  --model-manifest manifests/qwen3-4b.json \
  --device cuda:0 \
  --run-compliance \
  --artifact-dir "$HS_RESULTS/sft/checkpoints" \
  --report "$HS_RESULTS/sft/report.json" \
  --compliance-report "$HS_RESULTS/sft/compliance.json"
```

The constructor automatically excludes fixed compliance indices and skips a complete task if one
of its examples exceeds the 2,304-token training cap. The 4,096-token rollout limit remains
unchanged. Record the final immutable directory, normally
`shared_step-48_<checkpoint-digest>`:

```bash
export HS_SFT_CHECKPOINT="$HS_RESULTS/sft/checkpoints/shared_step-48_<checkpoint-digest>"
```

Replace the placeholder using the report's `checkpoint.checkpoint_id`: convert its colons to
underscores to obtain the directory name. Do not select a checkpoint by modification time.

## 8. Run one shared-policy RL cycle

ADR-0007 declares eight answer-independent `width_20k` tasks and eight complete rollouts per
task:

```bash
heterospawn wideseek-train-smoke \
  --topology shared \
  --split width_20k \
  --task-index 1 \
  --task-index 2500 \
  --task-index 5000 \
  --task-index 7500 \
  --task-index 10000 \
  --task-index 12500 \
  --task-index 15000 \
  --task-index 17500 \
  --rollouts-per-task 8 \
  --data-dir "$HS_DATA" \
  --checkpoint-dir "$HS_SFT_CHECKPOINT" \
  --model-profile qwen3-4b \
  --model-path "$HS_MODEL" \
  --model-manifest manifests/qwen3-4b.json \
  --device cuda:0 \
  --max-sequence-length 4096 \
  --max-new-tokens 1024 \
  --do-sample \
  --max-search-message-results 3 \
  --max-search-content-characters 600 \
  --max-access-characters 800 \
  --require-learning-signal \
  --artifact-dir "$HS_RESULTS/shared-rl/checkpoints" \
  --transaction-dir "$HS_RESULTS/shared-rl/transactions" \
  --report "$HS_RESULTS/shared-rl/report.json"
```

This runs exactly one `joint_update`, creates an immutable checkpoint, synchronizes rollout
weights, and atomically publishes the phase commit. Reward improvement is not an acceptance
condition; exact identities and a real finite learning signal are.

## 9. Run one independent Main/Sub RL cycle

Use a separate transaction and checkpoint directory. Supplying the shared SFT checkpoint invokes
the explicit ADR-0008 fork: both roles inherit the verified adapter and optimizer state, receive
distinct checkpoint identities, and then update independently.

```bash
heterospawn wideseek-train-smoke \
  --topology independent \
  --split width_20k \
  --task-index 1 \
  --task-index 2500 \
  --task-index 5000 \
  --task-index 7500 \
  --task-index 10000 \
  --task-index 12500 \
  --task-index 15000 \
  --task-index 17500 \
  --rollouts-per-task 8 \
  --data-dir "$HS_DATA" \
  --checkpoint-dir "$HS_SFT_CHECKPOINT" \
  --model-profile qwen3-4b \
  --model-path "$HS_MODEL" \
  --model-manifest manifests/qwen3-4b.json \
  --device cuda:0 \
  --max-sequence-length 4096 \
  --max-new-tokens 1024 \
  --do-sample \
  --max-search-message-results 3 \
  --max-search-content-characters 600 \
  --max-access-characters 800 \
  --require-sub-update \
  --require-learning-signal \
  --artifact-dir "$HS_RESULTS/independent-rl/checkpoints" \
  --transaction-dir "$HS_RESULTS/independent-rl/transactions" \
  --report "$HS_RESULTS/independent-rl/report.json"
```

The order is:

1. rollout with initial Main and Sub;
2. Main update, checkpoint, sync, and commit;
3. fresh rollout with updated Main and unchanged Sub;
4. Sub update, checkpoint, sync, and commit.

A failed Sub episode does not cancel its siblings. A genuine all-zero-spawn group ordinarily
produces no Sub update; `--require-sub-update` turns that valid skip into an experiment-level
failure because this particular pilot must demonstrate both policies.

## 10. Compare checkpoints without training

Use the same fixed tasks, sampling, token limits, and Search/Access budgets for every condition.
The commands below perform no optimizer step.

Evaluate SFT:

```bash
heterospawn wideseek-compliance-baseline \
  --topology shared \
  --checkpoint-dir "$HS_SFT_CHECKPOINT" \
  --rollouts-per-task 2 \
  --data-dir "$HS_DATA" \
  --model-profile qwen3-4b \
  --model-path "$HS_MODEL" \
  --model-manifest manifests/qwen3-4b.json \
  --device cuda:0 \
  --max-sequence-length 4096 \
  --max-new-tokens 1024 \
  --do-sample \
  --max-search-message-results 3 \
  --max-search-content-characters 600 \
  --max-access-characters 800 \
  --report "$HS_RESULTS/eval/sft.json"
```

Repeat the same command with the shared-RL checkpoint as `--checkpoint-dir`. For the independent
condition, use both role-specific checkpoints:

```bash
heterospawn wideseek-compliance-baseline \
  --topology independent \
  --main-checkpoint-dir "$HS_RESULTS/independent-rl/checkpoints/main_step-49_<digest>" \
  --sub-checkpoint-dir "$HS_RESULTS/independent-rl/checkpoints/sub_step-49_<digest>" \
  --rollouts-per-task 2 \
  --data-dir "$HS_DATA" \
  --model-profile qwen3-4b \
  --model-path "$HS_MODEL" \
  --model-manifest manifests/qwen3-4b.json \
  --device cuda:0 \
  --max-sequence-length 4096 \
  --max-new-tokens 1024 \
  --do-sample \
  --max-search-message-results 3 \
  --max-search-content-characters 600 \
  --max-access-characters 800 \
  --report "$HS_RESULTS/eval/independent.json"
```

Replace every digest placeholder with the exact directory recorded in the training report. Both
independent arguments are required, and role mismatches are rejected.

## 11. Recover an interrupted phase

Phase input, rollout IDs, batch digest, environment revisions, RNG state, and base versions are
durable before an optimizer update. If a process stops before commit publication, recover the
transaction without replaying model or Search/Access calls:

```bash
heterospawn wideseek-recover-phase \
  --transaction-id \
    "wideseek-train-smoke-shared:width_20k-cycle-0:joint_update" \
  --transaction-dir "$HS_RESULTS/shared-rl/transactions" \
  --model-profile qwen3-4b \
  --model-path "$HS_MODEL" \
  --model-manifest manifests/qwen3-4b.json \
  --device cuda:0 \
  --max-sequence-length 4096 \
  --max-new-tokens 1024 \
  --require-learning-signal \
  --artifact-dir "$HS_RESULTS/shared-rl/checkpoints" \
  --report "$HS_RESULTS/shared-rl/recovery.json"
```

Independent transaction IDs end in `main_update` or `sub_update` and use experiment ID
`wideseek-train-smoke-independent`. Use the same model, sequence, reward, dataset, prompt, tool,
and environment identities as the failed phase. If a commit already exists, recovery is
idempotent and must not advance the optimizer again.

## Acceptance and reports

Before treating a run as valid, check that its safe report confirms:

- model, dataset, corpus, retriever, prompt, tool, reward, and Judge revisions;
- exact response-token/old-log-probability alignment and stable event ordering;
- non-stale expected `WeightVersion` and `RolloutRevision`;
- target adapter changed while the partner stayed unchanged;
- checkpoint manifest and file digests verified;
- sync published only after the rollout adapter hash matched;
- phase commit exists and recovery is idempotent;
- peak GPU memory and elapsed time are recorded.

Zero reward variance is valid but records a degenerate group and zero advantage. It is not
evidence of learning. The current bounded held-out comparison did not show a statistically
reliable gain over SFT, so reports must describe these runs as systems and experimental
validation rather than a benchmark improvement.

## Security and cleanup

- Keep models, corpus, checkpoints, raw traces, Judge responses, caches, and service logs below
  `$HS_RUNTIME`, not in Git.
- Never place API keys in commands, files, reports, or remote URLs.
- MiniMax is optional and non-official. If deliberately enabled, pass
  `--judge minimax-development --allow-network` and provide its credential only through the
  process environment.
- Do not commit prompts, reference answers, retrieved content, token arrays, or generated text.
- Stop the retrieval launcher with `Ctrl-C`; it terminates its Qdrant and retrieval child process
  groups. Confirm GPU state with `nvidia-smi`.

For provider semantics and reward details, see
[WideSeek environment semantics](../benchmarks/wideseek-r1.md). Architecture decisions live
under [ADRs](../adr/), and measured outcomes are indexed in
[validation reports](../validation/README.md).
