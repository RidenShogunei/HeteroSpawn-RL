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
bash scripts/start_wideseek_offline.sh
```

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

Pass `http://127.0.0.1:8000` to `WideSeekLocalToolService`. Persist its provider revision in every
Search/Access outcome and bind the same corpus, retriever, collection, prompt, tool, Judge, and
reward revisions into the phase transaction. Never resume a phase when the environment identity
differs.
