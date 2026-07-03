"""Evaluate the RAG pipeline against a hand-labeled question set.

Metrics:
    recall@k   - did any gold source appear in the top-k retrieved chunks?
    MRR        - 1/rank of the first gold hit (higher = ranked better)
    refusal    - (with --answers) did unanswerable questions get a refusal?
    faithful/correct - (with --answers) LLM-as-judge on generated answers

Gold labels are substrings matched (case-insensitive) against each retrieved
chunk's id/path/symbol, so you label "where the answer lives", not exact ids.

LLM answers and judgments are cached in data/eval_cache.json so re-runs
are free and fast (important on Groq's free-tier rate limits).

Usage:
    python src/eval.py                 # retrieval metrics only (no LLM, fast)
    python src/eval.py --answers      # + generation, refusal, judge metrics
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
EVAL_SET = DATA_DIR / "eval_set.jsonl"
CACHE_FILE = DATA_DIR / "eval_cache.json"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
K = 5
REFUSAL_TEXT = "I couldn't find this in the indexed codebase"
SLEEP_BETWEEN_LLM_CALLS = 2  # stay under free-tier rate limits

JUDGE_PROMPT = """\
You are grading a RAG system's answer. Given the question, the context the
system retrieved, and its answer, output ONLY a JSON object:
{{"faithful": true/false, "correct": true/false}}

faithful = every claim in the answer is supported by the context
correct  = the answer actually answers the question accurately

Question: {question}

Context:
{context}

Answer:
{answer}"""


def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def cache_key(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()[:16]


def is_gold_hit(chunk_id: str, meta: dict, gold: list[str]) -> bool:
    haystack = f"{chunk_id} {meta.get('path', '')} {meta.get('symbol', '')}".lower()
    return any(g.lower() in haystack for g in gold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true",
                        help="also generate answers and run the LLM judge")
    parser.add_argument("--k", type=int, default=K)
    args = parser.parse_args()

    items = [json.loads(line)
             for line in EVAL_SET.read_text().splitlines() if line.strip()]
    print(f"[eval] {len(items)} questions "
          f"({sum(i['answerable'] for i in items)} answerable)")

    model = SentenceTransformer(EMBED_MODEL)
    collection = chromadb.PersistentClient(
        path=str(DATA_DIR / "chroma")).get_collection("chunks")
    cache = load_cache()

    # -------- retrieval metrics --------
    recalls, mrrs = [], []
    retrieved_per_q = []  # keep for the answer phase
    for item in items:
        emb = model.encode(QUERY_PREFIX + item["question"],
                           normalize_embeddings=True)
        res = collection.query(query_embeddings=[emb.tolist()],
                               n_results=args.k)
        hits = list(zip(res["ids"][0], res["metadatas"][0],
                        res["documents"][0]))
        retrieved_per_q.append(hits)

        if not item["answerable"]:
            continue
        rank = next((r for r, (cid, meta, _) in enumerate(hits, 1)
                     if is_gold_hit(cid, meta, item["gold"])), None)
        recalls.append(1.0 if rank else 0.0)
        mrrs.append(1.0 / rank if rank else 0.0)
        if not rank:
            print(f"  [miss] {item['question']}")

    print(f"\n=== Retrieval (k={args.k}) ===")
    print(f"recall@{args.k}: {sum(recalls)/len(recalls):.2f}  "
          f"({int(sum(recalls))}/{len(recalls)})")
    print(f"MRR:       {sum(mrrs)/len(mrrs):.2f}")

    if not args.answers:
        print("\n(retrieval-only run; add --answers for generation metrics)")
        return

    # -------- generation + judge metrics --------
    from ask import SYSTEM_PROMPT, build_prompt  # reuse the real pipeline
    import os
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    llm_model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

    def llm(prompt: str, system: str | None = None) -> str:
        key = cache_key(llm_model, system or "", prompt)
        if key in cache:
            return cache[key]
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        out = client.chat.completions.create(
            model=llm_model, messages=messages,
            temperature=0.1).choices[0].message.content
        cache[key] = out
        save_cache(cache)
        time.sleep(SLEEP_BETWEEN_LLM_CALLS)
        return out

    refusal_ok, faithful, correct = [], [], []
    for item, hits in zip(items, retrieved_per_q):
        hit_dicts = [{"text": doc, "meta": meta}
                     for cid, meta, doc in hits]
        ans = llm(build_prompt(item["question"], hit_dicts),
                  system=SYSTEM_PROMPT)

        if not item["answerable"]:
            ok = REFUSAL_TEXT.lower() in ans.lower()
            refusal_ok.append(1.0 if ok else 0.0)
            if not ok:
                print(f"  [no refusal] {item['question']}")
            continue

        context = "\n\n".join(h["text"][:1500] for h in hit_dicts)
        verdict_raw = llm(JUDGE_PROMPT.format(
            question=item["question"], context=context, answer=ans))
        try:
            start = verdict_raw.index("{")
            end = verdict_raw.rindex("}") + 1
            verdict = json.loads(verdict_raw[start:end])
        except (ValueError, json.JSONDecodeError):
            print(f"  [judge parse fail] {item['question']}")
            continue
        faithful.append(1.0 if verdict.get("faithful") else 0.0)
        correct.append(1.0 if verdict.get("correct") else 0.0)
        if not verdict.get("correct"):
            print(f"  [incorrect] {item['question']}")

    print("\n=== Generation ===")
    if faithful:
        print(f"faithful:  {sum(faithful)/len(faithful):.2f}")
        print(f"correct:   {sum(correct)/len(correct):.2f}")
    if refusal_ok:
        print(f"refusal:   {sum(refusal_ok)/len(refusal_ok):.2f}  "
              f"({int(sum(refusal_ok))}/{len(refusal_ok)})")


if __name__ == "__main__":
    main()
