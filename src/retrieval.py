"""Retrieval backends: dense, dense + query rewriting, hybrid, and reranked.

Modes:
    dense         - embedding similarity only (the v0 baseline)
    dense_rw      - LLM expands the question with likely code identifiers
                    and file names BEFORE embedding it (query rewriting)
    dense_rerank  - dense candidates re-scored by a cross-encoder
    hybrid        - dense + BM25 keyword search, fused with weighted RRF
    hybrid_rerank - hybrid candidates re-scored by a cross-encoder

Ablation findings on this corpus (42-question eval set):
    - dense won over hybrid: BM25 pulled in issue chunks that share common
      query words, displacing correct chunks from the top-5.
    - Two reranker generations (bge-reranker-base and v2-m3) both LOWERED
      MRR: cross-encoders demote raw code chunks for "where is X defined"
      questions because code *is* the answer without discussing it.
    - dense_rw targets the remaining misses: questions whose words are all
      too common to retrieve well ("Query, Path, and Body parameter
      functions") get expanded with identifiers an LLM can guess.
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
CANDIDATES = 20   # how many candidates hybrid/rerank gather before final cut
RRF_K = 60        # standard RRF constant
RRF_WEIGHTS = (2.0, 1.0)  # dense vs BM25 in the fused ranking

# --- query rewriting ---
REWRITE_CACHE = DATA_DIR / "rewrite_cache.json"
REWRITE_PROMPT = """\
You improve search queries for a search engine over the FastAPI repository
(source code, docs, GitHub issues). Rewrite the question as a short search
query, adding likely Python identifiers, class/function names, and file
names from FastAPI. Output ONLY the query, no explanation.

Question: {question}"""

# Lazy singletons so models/indexes load once per process, not per query
_embedder = None
_loaded = False
_reranker = None
_collection = None
_bm25 = None
_chunk_ids: list[str] = []
_chunk_by_id: dict[str, dict] = {}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _load() -> None:
    global _loaded, _embedder, _collection, _bm25, _chunk_ids, _chunk_by_id
    if _loaded:
        return
    _embedder = SentenceTransformer(EMBED_MODEL)
    _collection = chromadb.PersistentClient(
        path=str(DATA_DIR / "chroma")).get_collection("chunks")
    chunks = [json.loads(line) for line in
              (DATA_DIR / "chunks.jsonl").read_text().splitlines()]
    _chunk_ids = [c["id"] for c in chunks]
    _chunk_by_id = {c["id"]: c for c in chunks}
    _bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])
    _loaded = True


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)  # first run downloads ~2.3GB
    return _reranker


def _rewrite(question: str) -> str:
    """Expand the question with likely identifiers via one cached LLM call.

    Falls back to the original question on any failure (no key, rate
    limit, network) so retrieval never breaks because of the rewriter.
    """
    import os
    cache = (json.loads(REWRITE_CACHE.read_text())
             if REWRITE_CACHE.exists() else {})
    if question in cache:
        return cache[question]
    try:
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        out = client.chat.completions.create(
            model=model,
            messages=[{"role": "user",
                       "content": REWRITE_PROMPT.format(question=question)}],
            temperature=0.0).choices[0].message.content.strip()
        expanded = f"{question} {out}"
    except Exception as e:
        print(f"  [rewrite failed ({e.__class__.__name__}); using original]")
        return question
    cache[question] = expanded
    REWRITE_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    return expanded


def _dense_ids(question: str, n: int,
               with_scores: bool = False):
    emb = _embedder.encode(QUERY_PREFIX + question, normalize_embeddings=True)
    res = _collection.query(query_embeddings=[emb.tolist()], n_results=n)
    ids = res["ids"][0]
    if with_scores:
        # cosine distance -> similarity (1.0 = identical, 0.0 = unrelated)
        return ids, [1.0 - d for d in res["distances"][0]]
    return ids


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


def _rerank(question: str, candidates: list[str]) -> list[str]:
    pairs = [(question, _chunk_by_id[cid]["text"][:2000])
             for cid in candidates]
    scores = _get_reranker().predict(pairs)
    return [cid for _, cid in sorted(zip(scores, candidates), reverse=True)]


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


def retrieve(question: str, k: int = 5, mode: str = "dense") -> list[dict]:
    _load()
    if mode == "dense":
        ids, scores = _dense_ids(question, k, with_scores=True)
        hits = [_to_hit(cid) for cid in ids]
        for h, s in zip(hits, scores):
            h["score"] = s
        return hits
    elif mode == "dense_rw":
        ids = _dense_ids(_rewrite(question), k)
    elif mode == "dense_rerank":
        ids = _rerank(question, _dense_ids(question, CANDIDATES))[:k]
    elif mode in ("hybrid", "hybrid_rerank"):
        fused = _rrf_fuse([_dense_ids(question, CANDIDATES),
                           _bm25_ids(question, CANDIDATES)],
                          weights=RRF_WEIGHTS)
        candidates = fused[:CANDIDATES]
        if mode == "hybrid_rerank":
            candidates = _rerank(question, candidates)
        ids = candidates[:k]
    else:
        raise ValueError(f"unknown mode: {mode}")
    return [_to_hit(cid) for cid in ids]
