"""Retrieval backends: dense, hybrid (dense + BM25), and hybrid + reranker.

Modes:
    dense         - embedding similarity only (the v0 baseline)
    hybrid        - dense + BM25 keyword search, fused with Reciprocal Rank
                    Fusion (RRF)
    hybrid_rerank - hybrid to get ~20 candidates, then a cross-encoder
                    re-scores each (question, chunk) pair and keeps the best

Why BM25 helps this corpus: code questions contain exact identifiers
("APIRouter", "jsonable_encoder"). Embeddings blur those into meaning;
BM25 matches them literally. The two are complementary, and RRF merges
their rankings without needing to calibrate scores against each other.

Why the reranker helps: the embedding compares question and chunk as two
separate vectors. A cross-encoder reads them TOGETHER, token by token, so
it is much better at judging true relevance - but too slow to run on all
1300 chunks, which is why it only re-scores the top candidates.
"""

import json
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

DATA_DIR = Path("data")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CANDIDATES = 20   # how many candidates hybrid gathers before final cut
RRF_K = 60        # standard RRF constant

# Lazy singletons so models/indexes load once per process, not per query
_embedder = None
_reranker = None
_collection = None
_bm25 = None
_chunk_ids: list[str] = []
_chunk_by_id: dict[str, dict] = {}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _load() -> None:
    global _embedder, _collection, _bm25, _chunk_ids, _chunk_by_id
    if _embedder is not None:
        return
    _embedder = SentenceTransformer(EMBED_MODEL)
    _collection = chromadb.PersistentClient(
        path=str(DATA_DIR / "chroma")).get_collection("chunks")
    chunks = [json.loads(line) for line in
              (DATA_DIR / "chunks.jsonl").read_text().splitlines()]
    _chunk_ids = [c["id"] for c in chunks]
    _chunk_by_id = {c["id"]: c for c in chunks}
    _bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)  # first run downloads ~1GB
    return _reranker


def _dense_ids(question: str, n: int) -> list[str]:
    emb = _embedder.encode(QUERY_PREFIX + question, normalize_embeddings=True)
    res = _collection.query(query_embeddings=[emb.tolist()], n_results=n)
    return res["ids"][0]


def _bm25_ids(question: str, n: int) -> list[str]:
    scores = _bm25.get_scores(_tokenize(question))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [_chunk_ids[i] for i in ranked[:n]]


def _rrf_fuse(rankings: list[list[str]], weights: tuple) -> list[str]:
    """Weighted RRF: score(id) = sum of w/(RRF_K + rank) over lists."""
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, cid in enumerate(ranking, 1):
            scores[cid] = scores.get(cid, 0.0) + w / (RRF_K + rank)
    return sorted(scores, key=scores.get, reverse=True)


def _to_hit(cid: str) -> dict:
    c = _chunk_by_id[cid]
    return {
        "id": cid,
        "text": c["text"],
        "meta": {
            "source_type": c["source_type"],
            "path": c["path"],
            "symbol": c["symbol"] or "",
            "url": c["url"],
        },
    }

def _rerank(question: str, candidates: list[str]) -> list[str]:
    pairs = [(question, _chunk_by_id[cid]["text"][:2000])
             for cid in candidates]
    scores = _get_reranker().predict(pairs)
    return [cid for _, cid in sorted(zip(scores, candidates), reverse=True)]

def retrieve(question: str, k: int = 5, mode: str = "dense") -> list[dict]:
    _load()
    if mode == "dense":
        ids = _dense_ids(question, k)
    elif mode == "dense_rerank":
        ids = _rerank(question, _dense_ids(question, CANDIDATES))[:k]
    elif mode in ("hybrid", "hybrid_rerank"):
        fused = _rrf_fuse([_dense_ids(question, CANDIDATES),
                                _bm25_ids(question, CANDIDATES)],
                                weights=(2.0, 1.0))
        candidates = fused[:CANDIDATES]
        if mode == "hybrid_rerank":
            candidates = _rerank(question, candidates)
        ids = candidates[:k]
    else:
        raise ValueError(f"unknown mode: {mode}")
    return [_to_hit(cid) for cid in ids]
