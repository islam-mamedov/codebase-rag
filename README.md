---
title: FastAPI Codebase Q&A
emoji: ⚡
colorFrom: yellow
colorTo: gray
sdk: streamlit
app_file: app.py
pinned: false
---

# ⚡ FastAPI Codebase Q&A — an eval-driven RAG system

Ask questions about the [FastAPI](https://github.com/fastapi/fastapi)
codebase and get answers grounded in its actual source code, documentation,
and GitHub issues — with citations linking to the exact lines, and honest
refusals when the corpus doesn't contain the answer.

**Live demo:** https://huggingface.co/spaces/islam-mamedov/fastapi-codebase-qa

Every architectural decision in this project was made by measuring, not
guessing: a hand-labeled 42-question eval set gates every change, and the
ablation results below include the things that *didn't* work and why.

## Architecture

```
GitHub repo ──► Ingestion ──► Chunking ──► Indexing ──► Retrieval ──► Generation
  code            clone         AST-aware     ChromaDB     dense         LLM with
  docs (en)       API pull      (tree-sitter) bge-small    (winner of    citations +
  issues          resumable     md-aware      + BM25       6 ablations)  refusal rules
                                issue-aware
                                     │                          │
                                     └────── Eval harness ◄─────┘
                                       recall@5 · MRR · faithfulness ·
                                       correctness · refusal accuracy
```

- **Corpus:** 46 source files, 161 English docs, 175 closed issues →
  **1,352 chunks** (363 code, 814 doc, 175 issue)
- **Chunking:** tree-sitter AST parsing — one chunk per function/class,
  classes >150 lines split per-method, every chunk carries a context header
  (`# File: fastapi/routing.py | Symbol: APIRouter`) and exact line numbers
  for citations
- **Stack:** Python, ChromaDB, sentence-transformers (`bge-small-en-v1.5`),
  rank-bm25, Groq (LLM), Streamlit — total cost: $0

## Evaluation

42 hand-labeled questions across four categories: **api_usage** (16),
**behavior** (11), **location** ("where is X defined", 8), and
**unanswerable** (7 — questions FastAPI's corpus genuinely can't answer,
to measure refusal). Gold labels mark which file/symbol contains the
answer; retrieval is scored by recall@5 and MRR, generation by an
LLM-as-judge (faithfulness, correctness) plus string-matched refusals.

### Retrieval ablations

| Configuration | recall@5 | MRR | Verdict |
|---|---|---|---|
| **dense (bge-small)** | **0.91** | **0.71** | **winner — shipped** |
| hybrid (dense + BM25, RRF) | 0.86 | 0.70 | BM25 noise displaces gold chunks |
| hybrid, weighted RRF 2:1 | 0.86 | 0.71 | weighting doesn't recover recall |
| dense + bge-reranker-base | 0.91 | 0.63 | reranker *hurts* ranking |
| dense + bge-reranker-v2-m3 | 0.83 | 0.67 | newer reranker also hurts |
| hybrid + bge-reranker-v2-m3 | 0.86 | 0.68 | — |
| dense + LLM query rewriting | 0.91 | 0.69 | no gain; noisy expansions add drift |

**Why hybrid lost:** diagnostic runs showed dense@20 = hybrid@20 = 0.97 —
BM25 contributed zero unique gold candidates on this corpus, while its
common-token matches (issue chunks full of "request", "body", "json")
crowded correct chunks out of the top 5.

**Why the rerankers lost:** both generations of cross-encoder demoted raw
code chunks on "where is X defined" questions. A code chunk *is* the
answer without *discussing* the answer — and rerankers are trained to
score passages that discuss. dense@20 recall of 0.97 vs reranked@5 recall
of 0.83–0.86 shows they were given the right candidates and ranked them
worse.

### Generation (dense, llama-4-scout via Groq)

| Metric | Score |
|---|---|
| Faithfulness (LLM judge) | 0.89 |
| Correctness (LLM judge) | 0.91 |
| Refusal accuracy (7 unanswerable questions) | 7/7 |

Refusal was the most fragile metric: it swung 6/7 → 0/7 → 7/7 across three
system-prompt variants. The fix was naming the failure mode in the prompt
("the context chunks are search results: they may be loosely RELATED to
the question without actually answering it"). Without the eval harness
this regression would have shipped silently.

A score-threshold refusal gate was also tested and **rejected**: top-1
similarity scores for answerable (0.74–0.89) and unanswerable (0.72–0.79)
questions overlap heavily — semantic similarity can't distinguish "related
to FastAPI" from "answerable from FastAPI's corpus".

## Failure analysis (what still breaks)

1. **Common-token queries.** "Where are the Query, Path, and Body parameter
   functions defined?" (`fastapi/param_functions.py`) misses even at k=20
   in every configuration — all its query tokens are ubiquitous. LLM query
   rewriting that injects likely identifiers is the targeted fix.
2. **Docs without code.** FastAPI's docs embed examples via
   `{* ../../docs_src/... *}` include-directives; the referenced snippets
   aren't inlined into chunks, so some answers describe code they can't
   show. Resolving the includes at chunking time is the top corpus
   improvement.
3. **Judge strictness.** Spot-checking answers flagged `[incorrect]` showed
   at least one (JWT auth) is actually correct — the LLM judge penalizes
   answers referencing code the truncated context implies but doesn't
   show. Judge scores are treated as directional, not absolute.

## Engineering notes

- **Idempotent, resumable ingestion** — issues already fetched are skipped,
  so network failures mid-run just need a re-run
- **Everything cached** — LLM answers, judgments, and query rewrites are
  cached to disk keyed by (model, prompt), making eval re-runs free on
  Groq's rate-limited free tier
- **Rate-limit resilience** — retry with backoff; eval survived TPM/TPD
  limits and a mid-run model deprecation (Groq retired llama-3.3 mid-sprint)
- **Self-building deploy** — the HF Space rebuilds the vector index from
  committed `chunks.jsonl` on first boot; no binary artifacts in git
- **Unit-tested chunkers** and an eval suite that doubles as a regression
  gate

## Run it yourself

```bash
git clone https://github.com/islam-mamedov/codebase-rag && cd codebase-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt PyGithub tree-sitter tree-sitter-python pytest
echo "GROQ_API_KEY=gsk_..." > .env          # free key: console.groq.com

python src/ingest.py --repo fastapi/fastapi  # needs GITHUB_TOKEN for issues
python src/chunk.py
python src/index.py
python src/ask.py "How do I return a custom status code?"
python src/eval.py --mode dense --answers    # reproduce the numbers above
streamlit run app.py
```

## What I'd build next

- Resolve `docs_src` include-directives into doc chunks (highest-impact
  corpus fix)
- Make ingestion repo-agnostic (`--repo any/repo`) — turn the project into
  a tool
- Embedding-model ablation (bge-small vs bge-base vs code-specific models)
- Ingest GitHub Discussions, where most FastAPI Q&A actually lives
- CI job that fails the build if recall@5 drops below baseline
