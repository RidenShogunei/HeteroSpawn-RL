#!/usr/bin/env bash
set -euo pipefail

required_vars=(
  HETEROSPAWN_WIKI_DIR
  HETEROSPAWN_E5_DIR
  HETEROSPAWN_RLINF_DIR
  HETEROSPAWN_RUNTIME_DIR
)
for variable in "${required_vars[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "$variable is required" >&2
    exit 2
  fi
done

expected_upstream="d9f3d8a9db4d7aad1d641029293295503dd3eb2c"
collection="wiki_collection_m32_cef512"
corpus_revision="178d7d037f661be3159b0c3a8a4119b974f01880"
service_port="${HETEROSPAWN_RETRIEVAL_PORT:-8000}"
qdrant_port="${HETEROSPAWN_QDRANT_PORT:-6333}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="$(realpath -m "$HETEROSPAWN_RUNTIME_DIR")"
wiki_dir="$(realpath "$HETEROSPAWN_WIKI_DIR")"
e5_dir="$(realpath "$HETEROSPAWN_E5_DIR")"
rlinf_dir="$(realpath "$HETEROSPAWN_RLINF_DIR")"

if [[ "$(git -C "$rlinf_dir" rev-parse HEAD)" != "$expected_upstream" ]]; then
  echo "RLinf checkout is not at the pinned WideSeek revision" >&2
  exit 2
fi

mkdir -p "$runtime_dir/logs"
heterospawn wideseek-fetch-assets \
  --manifest "$project_root/manifests/wideseek-wiki-2018-corpus.json" \
  --destination "$wiki_dir" \
  --verify-only
heterospawn wideseek-fetch-assets \
  --manifest "$project_root/manifests/wideseek-e5-base-v2.json" \
  --destination "$e5_dir" \
  --verify-only

qdrant_deploy="$runtime_dir/qdrant-$corpus_revision"
if [[ ! -d "$qdrant_deploy" ]]; then
  cp --archive --reflink=auto "$wiki_dir/qdrant" "$qdrant_deploy"
fi
qdrant_binary="$qdrant_deploy/qdrant"
retrieval_server="$rlinf_dir/examples/agent/tools/search_local_server_qdrant/local_retrieval_server.py"
if [[ ! -f "$qdrant_binary" || ! -f "$retrieval_server" ]]; then
  echo "verified runtime is missing Qdrant or the pinned retrieval server" >&2
  exit 2
fi
chmod u+x "$qdrant_binary"

qdrant_pid=""
service_pid=""
cleanup() {
  if [[ -n "$service_pid" ]]; then
    kill "$service_pid" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
  fi
  if [[ -n "$qdrant_pid" ]]; then
    kill "$qdrant_pid" 2>/dev/null || true
    wait "$qdrant_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

pushd "$qdrant_deploy" >/dev/null
QDRANT__SERVICE__HTTP_PORT="$qdrant_port" \
  ./qdrant >"$runtime_dir/logs/qdrant.log" 2>&1 &
qdrant_pid=$!
popd >/dev/null

qdrant_ready=false
for _ in $(seq 1 120); do
  if curl --fail --silent \
    "http://127.0.0.1:${qdrant_port}/collections/${collection}" >/dev/null; then
    qdrant_ready=true
    break
  fi
  sleep 2
done
if [[ "$qdrant_ready" != true ]] || ! kill -0 "$qdrant_pid" 2>/dev/null; then
  echo "Qdrant failed to become ready" >&2
  exit 1
fi

python -u "$retrieval_server" \
  --pages_path "$wiki_dir/wiki_webpages.jsonl" \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model "$e5_dir" \
  --qdrant_collection_name "$collection" \
  --qdrant_url "http://127.0.0.1:${qdrant_port}" \
  --qdrant_search_param '{"hnsw_ef":256}' \
  --port "$service_port" \
  >"$runtime_dir/logs/retrieval-service.log" 2>&1 &
service_pid=$!

service_ready=false
for _ in $(seq 1 300); do
  if curl --fail --silent \
    -H "Content-Type: application/json" \
    -d '{"queries":["Red Bull"],"topk":1,"return_scores":true}' \
    "http://127.0.0.1:${service_port}/retrieve" >/dev/null; then
    service_ready=true
    break
  fi
  if ! kill -0 "$service_pid" 2>/dev/null; then
    echo "retrieval service exited before readiness" >&2
    exit 1
  fi
  sleep 2
done
if [[ "$service_ready" != true ]]; then
  echo "retrieval service failed to become ready" >&2
  exit 1
fi

heterospawn wideseek-check-environment \
  --corpus-dir "$wiki_dir" \
  --retriever-dir "$e5_dir" \
  --service-url "http://127.0.0.1:${service_port}" \
  --qdrant-url "http://127.0.0.1:${qdrant_port}"

echo "WideSeek offline environment is ready; press Ctrl-C to stop."
wait "$service_pid"
