"""BrowseComp-Plus offline local Search/Access over a fixed document corpus."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from heterospawn.errors import ConfigurationError, SearchRequestError
from heterospawn.search.base import (
    AccessRequest,
    AccessResponse,
    SearchItem,
    SearchRequest,
    SearchResponse,
)

BCPLUS_TOOL_REVISION = "heterospawn-browsecomp-plus-bm25-v2"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class BrowseCompPlusLocalTools:
    """Offline ResearchToolService using BM25 over a JSONL corpus.

    Search returns synthetic doc URLs of the form ``bcplus://doc/{docid}``.
    Access reads the full stored document text for that URL.

    Snippet truncation defaults to 512 tokens (vendor fair-comparison protocol).
    Prefer :class:`BrowseCompPlusPyseriniTools` when the official Lucene index
    is available.
    """

    def __init__(
        self,
        corpus_path: Path,
        *,
        max_docs: int | None = None,
        snippet_max_tokens: int = 512,
        tokenizer_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        if not corpus_path.is_file():
            raise ConfigurationError(f"BrowseComp-Plus corpus missing: {corpus_path}")
        self._docs: dict[str, str] = {}
        self._doc_ids: list[str] = []
        self._doc_tfs: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._avgdl = 0.0
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
        self._load_corpus(corpus_path, max_docs=max_docs)
        self.provider_revision = BCPLUS_TOOL_REVISION
        self.corpus_path = corpus_path
        self.doc_count = len(self._doc_ids)
        self.retriever_name = "HeteroSpawn-BM25-jsonl"

    def _load_corpus(self, corpus_path: Path, *, max_docs: int | None) -> None:
        lengths: list[int] = []
        with corpus_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                docid = str(row.get("docid") or row.get("id") or "")
                text = str(row.get("text") or row.get("contents") or "")
                if not docid or not text:
                    continue
                self._docs[docid] = text
                tokens = _tokenize(text)
                tf = Counter(tokens)
                self._doc_ids.append(docid)
                self._doc_tfs.append(tf)
                lengths.append(max(1, sum(tf.values())))
                for term in tf:
                    self._df[term] += 1
                if max_docs is not None and len(self._doc_ids) >= max_docs:
                    break
        if not self._doc_ids:
            raise ConfigurationError("BrowseComp-Plus corpus is empty")
        self._avgdl = sum(lengths) / len(lengths)
        self._N = len(self._doc_ids)

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

    async def search(self, request: SearchRequest) -> SearchResponse:
        query_terms = _tokenize(request.query)
        if not query_terms:
            raise SearchRequestError("BrowseComp-Plus search query produced no tokens")
        scores: list[tuple[float, int]] = []
        for index, tf in enumerate(self._doc_tfs):
            score = _bm25_score(query_terms, tf, self._df, self._N, self._avgdl)
            if score > 0:
                scores.append((score, index))
        scores.sort(reverse=True)
        results: list[SearchItem] = []
        for score, index in scores[: request.max_results]:
            docid = self._doc_ids[index]
            text = self._docs[docid]
            snippet = self._truncate_snippet(text).replace("\n", " ")
            results.append(
                SearchItem(
                    title=f"doc:{docid}",
                    url=f"bcplus://doc/{docid}",
                    content=snippet,
                    score=float(score),
                )
            )
        payload = json.dumps(
            {"query": request.query, "docids": [item.url for item in results]},
            ensure_ascii=False,
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return SearchResponse(
            request_id=request.request_id,
            provider="browsecomp-plus-bm25",
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
        text = self._docs.get(docid)
        if text is None:
            raise SearchRequestError(f"Unknown BrowseComp-Plus docid: {docid}")
        digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        truncated = len(text) > request.max_characters
        return AccessResponse(
            request_id=request.request_id,
            provider="browsecomp-plus-local",
            provider_revision=self.provider_revision,
            provider_request_id=f"bcplus:{docid}:{digest[:16]}",
            url=request.url,
            content=text[: request.max_characters],
            truncated=truncated,
            raw_response_digest=digest,
        )


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _bm25_score(
    query_terms: list[str],
    tf: Counter[str],
    df: Counter[str],
    n_docs: int,
    avgdl: float,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    score = 0.0
    doc_len = float(sum(tf.values()) or 1)
    for term in query_terms:
        freq = tf.get(term, 0)
        if freq <= 0:
            continue
        doc_freq = df.get(term, 0)
        idf = math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        denom = freq + k1 * (1.0 - b + b * doc_len / avgdl)
        score += idf * (freq * (k1 + 1.0) / denom)
    return score
