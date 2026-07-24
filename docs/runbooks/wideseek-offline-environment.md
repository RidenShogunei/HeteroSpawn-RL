# WideSeek offline environment runbook

This runbook starts the pinned Wiki-2018/Qdrant/E5 Search and Access environment on one Linux
machine. It does not install or invoke the RLinf training runtime.

## Resource envelope

- Reserve at least 180 GB of free disk before download. The pinned corpus manifest contains 3,400
  files and 155,895,995,164 bytes; temporary and Hugging Face cache files need additional space.
- The launcher clones the verified Qdrant directory into a mutable deployment with
  `cp --reflink=auto`. A copy-on-write filesystem avoids duplicating the index; a filesystem
  without reflink support needs enough space for a second physical copy. The verified source is
  never started directly because Qdrant mutates its WAL and metadata.
- `wiki_webpages.jsonl` is about 26.6 GB and the pinned upstream service builds an in-memory URL
  map. Confirm sufficient host RAM before launch.
- Use one CUDA GPU for `intfloat/e5-base-v2`. The checked configuration uses 768-dimensional
  vectors, cosine distance, collection `wiki_collection_m32_cef512`, `m=32`,
  `ef_construct=512`, and query `hnsw_ef=256`.
- Put every download, service log, and cache below an ignored runtime directory. Do not put the
  corpus inside the Git checkout.

## Prepare isolated runtime

From a new clean HeteroSpawn-RL clone:

```bash
python3.11 -m venv "$HOME/heterospawn-runtime/wideseek/.venv"
source "$HOME/heterospawn-runtime/wideseek/.venv/bin/activate"
python -m pip install -e ".[wideseek]"
```

Clone the semantic source only and pin it exactly:

```bash
git clone https://github.com/RLinf/RLinf.git \
  "$HOME/heterospawn-runtime/wideseek/RLinf"
git -C "$HOME/heterospawn-runtime/wideseek/RLinf" \
  checkout d9f3d8a9db4d7aad1d641029293295503dd3eb2c
```

Install the pinned upstream retrieval server's Python dependencies in this venv. This environment
is isolated from HeteroSpawn training and rollout environments:

```bash
python -m pip install \
  fastapi numpy qdrant-client sentence-transformers torch tqdm uvicorn
```

Download through the official endpoint with automatic mirror fallback:

```bash
heterospawn wideseek-fetch-assets \
  --manifest manifests/wideseek-wiki-2018-corpus.json \
  --destination "$HOME/heterospawn-runtime/wideseek/wiki-2018-corpus"
heterospawn wideseek-fetch-assets \
  --manifest manifests/wideseek-e5-base-v2.json \
  --destination "$HOME/heterospawn-runtime/wideseek/e5-base-v2"
```

If both endpoints are unavailable, copy those directories from another machine and rerun each
command with `--verify-only`. Do not bypass the manifest. Interrupted files remain under the
destination and Hub cache for resume; a digest mismatch is quarantined and is phase-fatal.

## Start and verify

Select the retrieval GPU before launch and set only non-secret paths:

```bash
export CUDA_VISIBLE_DEVICES=0
export HETEROSPAWN_WIKI_DIR="$HOME/heterospawn-runtime/wideseek/wiki-2018-corpus"
export HETEROSPAWN_E5_DIR="$HOME/heterospawn-runtime/wideseek/e5-base-v2"
export HETEROSPAWN_RLINF_DIR="$HOME/heterospawn-runtime/wideseek/RLinf"
export HETEROSPAWN_RUNTIME_DIR="$HOME/heterospawn-runtime/wideseek/service"
export HETEROSPAWN_RETRIEVAL_PYTHON="$HOME/heterospawn-runtime/wideseek/.venv/bin/python"
bash scripts/start_wideseek_offline.sh
```

`HETEROSPAWN_RETRIEVAL_PYTHON` may point at an isolated retrieval environment while the
`heterospawn` command comes from a separate training environment. When it is omitted, the
launcher uses `python` from `PATH`.

The launcher:

1. rejects a non-pinned RLinf checkout;
2. verifies all corpus and retriever bytes;
3. starts the corpus-provided Qdrant binary;
4. starts the pinned upstream retrieval service with official-shaped `/retrieve` and `/access`;
5. validates collection state/configuration and a real Search-to-Access probe;
6. prints only a safe revision/resource report and stops both children on exit.

The readiness report must contain environment revision
`2b90af41266aafdbf27af8743f7a80700567afaa24ec5993961b39150c193340` after the implementation
is released. Treat any different revision, non-green collection, empty Access page, or asset
digest mismatch as a hard failure. Logs may contain retrieved text and therefore remain ignored.

## Handoff to training

First prove one real Search-to-Access loop. The report contains only counts, revisions, and
provenance checks; the trace and retrieved text stay below ignored `artifacts/`:

```bash
heterospawn wideseek-rollout-smoke \
  --service-url http://127.0.0.1:8000 \
  --qdrant-url http://127.0.0.1:6333 \
  --report artifacts/wideseek-rollout-smoke/report.json
```

Then run both short training topologies from separate transaction directories. Reusing a
transaction directory with different inputs is intentionally rejected:

```bash
heterospawn wideseek-train-smoke \
  --topology shared \
  --split width_20k \
  --task-index 0 \
  --rollouts-per-task 2 \
  --model-path "$HOME/heterospawn-runtime/models/Qwen2.5-0.5B-Instruct" \
  --device cuda:0 \
  --max-sequence-length 4096 \
  --max-new-tokens 512 \
  --artifact-dir "$HOME/heterospawn-runtime/results/shared/checkpoints" \
  --transaction-dir "$HOME/heterospawn-runtime/results/shared/transactions" \
  --report "$HOME/heterospawn-runtime/results/shared/report.json"

heterospawn wideseek-train-smoke \
  --topology independent \
  --split depth_20k \
  --task-index 0 \
  --rollouts-per-task 2 \
  --model-path "$HOME/heterospawn-runtime/models/Qwen2.5-0.5B-Instruct" \
  --device cuda:0 \
  --max-sequence-length 4096 \
  --max-new-tokens 512 \
  --artifact-dir "$HOME/heterospawn-runtime/results/independent/checkpoints" \
  --transaction-dir "$HOME/heterospawn-runtime/results/independent/transactions" \
  --report "$HOME/heterospawn-runtime/results/independent/report.json"
```

For the Qwen3-4B research profile, run the training command from an environment installed with
`.[qlora]` and replace the model arguments with:

```bash
  --model-profile qwen3-4b \
  --model-path "$HOME/heterospawn-runtime/models/Qwen3-4B" \
  --model-manifest manifests/qwen3-4b.json \
  --max-search-message-results 3 \
  --max-search-content-characters 600 \
  --max-access-characters 800 \
  --do-sample
```

The QLoRA and retrieval environments may be separate. Select the E5 environment with
`HETEROSPAWN_RETRIEVAL_PYTHON` when launching Search/Access, and select the QLoRA environment's
`heterospawn` command for training. Do not install their dependency stacks into each other.
`--do-sample` explicitly uses temperature 1, top-p 1, and top-k 0 so the recorded old log-probs
and update-time raw policy log-probs have identical semantics. Warped sampling requires a future
training contract that also reconstructs the same warper during update.
The Qwen3 CLI profile disables thinking so a complete action fits the bounded 4096-token smoke
window. A response that still reaches the generation length limit without a complete tool call is
an invalid repair attempt, not an `ANSWER`.
Search and Access model-visible content is deterministically bounded by the two character-budget
flags. Full response digests and URL provenance remain in the audit trace; changing either budget
changes the prompt and phase identity.

Use `--require-sub-update` only when the selected acceptance task must demonstrate a non-empty Sub
batch. A genuine 0-spawn group remains valid: it contributes to the Sub reward baseline, produces
no Sub sample, and does not advance the Sub optimizer or rollout revision. A zero-variance reward
group records a degenerate-group metric and a zero advantage; do not misreport its optimizer
transaction as evidence of reward improvement.

For the optional non-official development Judge, export the MiniMax credential only in the process
environment and add `--judge minimax-development --allow-network`. The implementation caps
provider requests at 128, caches exact-revision verdicts, and fails the phase if the final Judge
attempt fails. Never put the credential in a command, report, trace, or Git file.

Every Search/Access outcome persists the provider revision. The phase transaction binds the same
dataset, corpus, retriever, collection, prompt, tool, Judge, and reward revisions. Never resume a
phase when the environment identity differs.
