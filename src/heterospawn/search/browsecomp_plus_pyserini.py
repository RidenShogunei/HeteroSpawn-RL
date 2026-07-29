"""BrowseComp-Plus tools backed by official Anserini/Lucene BM25 index."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from heterospawn.errors import ConfigurationError, SearchRequestError
from heterospawn.search.base import (
    AccessRequest,
    AccessResponse,
    SearchItem,
    SearchRequest,
    SearchResponse,
)

BCPLUS_PYSERINI_REVISION = "browsecomp-plus-official-bm25-anserini-v1"


class BrowseCompPlusPyseriniTools:
    """Offline ResearchToolService over Tevatron Lucene BM25 index.

    Uses Anserini ``SimpleSearcher`` via Pyjnius (same stack as Pyserini),
    without importing Pyserini's dense/OpenAI optional modules.

    Fair-comparison defaults match vendor docs: top-k=5 (caller-controlled),
    search snippets truncated to ``snippet_max_tokens`` (default 512).
    """

    def __init__(
        self,
        index_path: Path,
        *,
        snippet_max_tokens: int = 512,
        tokenizer_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        if not index_path.is_dir():
            raise ConfigurationError(f"BrowseComp-Plus BM25 index missing: {index_path}")
        try:
            from pyserini.pyclass import autoclass
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ConfigurationError(
                "pyserini/pyjnius required for official BM25; install pyserini + Java 21"
            ) from exc
        j_simple = autoclass("io.anserini.search.SimpleSearcher")
        self._searcher = j_simple(str(index_path))
        self._snippet_max_tokens = snippet_max_tokens
        self._tokenizer = None
        if snippet_max_tokens > 0:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name, trust_remote_code=True
                )
            except Exception:
                self._tokenizer = None
        self.provider_revision = BCPLUS_PYSERINI_REVISION
        self.index_path = index_path
        try:
            self.doc_count = int(self._searcher.get_total_num_docs())
        except Exception:
            self.doc_count = -1
        self.retriever_name = "BM25-official-anserini"

    def _truncate_snippet(self, text: str) -> str:
        if self._snippet_max_tokens <= 0:
            return text
        if self._tokenizer is not None:
            tokens = self._tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > self._snippet_max_tokens:
                tokens = tokens[: self._snippet_max_tokens]
                return self._tokenizer.decode(tokens, skip_special_tokens=True)
            return text
        limit = self._snippet_max_tokens * 4
        return text if len(text) <= limit else text[:limit]

    def _doc_text(self, docid: str) -> str | None:
        doc = self._searcher.doc(docid)
        if doc is None:
            return None
        raw = doc.get("raw")
        if not raw:
            return None
        try:
            payload = json.loads(str(raw))
        except json.JSONDecodeError:
            return str(raw)
        return str(payload.get("contents") or payload.get("text") or "")

    async def search(self, request: SearchRequest) -> SearchResponse:
        query = request.query.strip()
        if not query:
            raise SearchRequestError("BrowseComp-Plus search query is empty")
        try:
            hits = self._searcher.search(query, int(request.max_results))
        except Exception as exc:
            raise SearchRequestError(f"Anserini BM25 search failed: {exc}") from exc
        results: list[SearchItem] = []
        for hit in hits:
            docid = str(hit.docid)
            text = self._doc_text(docid) or ""
            snippet = self._truncate_snippet(text).replace("\n", " ")
            results.append(
                SearchItem(
                    title=f"doc:{docid}",
                    url=f"bcplus://doc/{docid}",
                    content=snippet,
                    score=float(hit.score),
                )
            )
        payload = json.dumps(
            {
                "query": request.query,
                "docids": [item.url for item in results],
                "backend": "anserini-bm25",
            },
            ensure_ascii=False,
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return SearchResponse(
            request_id=request.request_id,
            provider="browsecomp-plus-bm25-anserini",
            provider_revision=self.provider_revision,
            provider_request_id=digest[:16],
            query=request.query,
            results=tuple(results),
            credits=None,
            raw_response_digest=digest,
        )

    async def access(self, request: AccessRequest) -> AccessResponse:
        if not request.url.startswith("bcplus://doc/"):
            raise SearchRequestError("BrowseComp-Plus Access only accepts bcplus://doc/{docid} URLs")
        docid = request.url[len("bcplus://doc/") :]
        try:
            text = self._doc_text(docid)
        except Exception as exc:
            raise SearchRequestError(f"Anserini doc lookup failed: {exc}") from exc
        if text is None:
            raise SearchRequestError(f"Unknown BrowseComp-Plus docid: {docid}")
        digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        truncated = len(text) > request.max_characters
        return AccessResponse(
            request_id=request.request_id,
            provider="browsecomp-plus-anserini",
            provider_revision=self.provider_revision,
            provider_request_id=f"bcplus:{docid}:{digest[:16]}",
            url=request.url,
            content=text[: request.max_characters],
            truncated=truncated,
            raw_response_digest=digest,
        )


def ensure_java_home(java_home: str | None = None) -> str | None:
    """Prefer an explicit JAVA_HOME (e.g. local Temurin 21) for Anserini/Pyjnius."""
    if java_home:
        os.environ["JAVA_HOME"] = java_home
        bin_dir = str(Path(java_home) / "bin")
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        libjvm = Path(java_home) / "lib" / "server" / "libjvm.so"
        if libjvm.is_file():
            os.environ["JVM_PATH"] = str(libjvm)
        return java_home
    return os.environ.get("JAVA_HOME")
