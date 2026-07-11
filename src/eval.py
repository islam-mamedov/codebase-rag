"""Evaluate the RAG pipeline against a hand-labeled question set.

Now supports retrieval modes for ablation runs:
    python src/eval.py --mode dense
    python src/eval.py --mode hybrid
    python src/eval.py --mode hybrid_rerank
    python src/eval.py --mode hybrid_rerank --answers

Metrics:
    recall@k   - did any gold source appear in the top-k retrieved chunks?
    MRR        - 1/rank of the first gold hit (higher = ranked better)
    refusal    - (with --answers) did unanswerable questions get a refusal?
    faithful/correct - (with --answers) LLM-as-judge on generated answers

LLM answers/judgments are cached in data/eval_cache.json (keyed by model
and prompt), so re-runs only pay for what changed.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from retrieval import retrieve
from dotenv import load_dotenv
load_dotenv(override=True)

DATA_DIR = Path("data")
EVAL_SET = DATA_DIR / "eval_set.jsonl"
CACHE_FILE = DATA_DIR / "eval_cache.json"
K = 5
REFUSAL_TEXT = "I couldn't find this in the indexed codebase"
SLEEP_BETWEEN_LLM_CALLS = 5

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


def is_gold_hit(hit: dict, gold: list[str]) -> bool:
    meta = hit["meta"]
    haystack = f"{hit['id']} {meta.get('path', '')} {meta.get('symbol', '')}".lower()
    return any(g.lower() in haystack for g in gold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="dense",
                            choices=["dense", "dense_rw", "dense_rerank",
                                    "hybrid", "hybrid_rerank"])
    parser.add_argument("--answers", action="store_true",
                        help="also generate answers and run the LLM judge")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument(
        "--min-recall",
        type=float,
        default=None,
        help="Exit with code 1 when recall@k is below this threshold.",
    )
    args = parser.parse_args()

    items = [json.loads(line)
             for line in EVAL_SET.read_text().splitlines() if line.strip()]
    print(f"[eval] mode={args.mode}, {len(items)} questions "
          f"({sum(i['answerable'] for i in items)} answerable)")

    cache = load_cache()

    # -------- retrieval metrics --------
    recalls, mrrs = [], []
    retrieved_per_q = []

    for item in items:
        hits = retrieve(item["question"], k=args.k, mode=args.mode)
        retrieved_per_q.append(hits)

        if not item["answerable"]:
            continue

        rank = next(
            (
                position
                for position, hit in enumerate(hits, 1)
                if is_gold_hit(hit, item["gold"])
            ),
            None,
        )

        recalls.append(1.0 if rank else 0.0)
        mrrs.append(1.0 / rank if rank else 0.0)

        if not rank:
            print(f"  [miss] {item['question']}")

    if not recalls:
        raise RuntimeError(
            "The evaluation set contains no answerable questions."
        )

    recall_at_k = sum(recalls) / len(recalls)
    mrr = sum(mrrs) / len(mrrs)

    print(f"\n=== Retrieval (mode={args.mode}, k={args.k}) ===")
    print(
        f"recall@{args.k}: {recall_at_k:.2f}  "
        f"({int(sum(recalls))}/{len(recalls)})"
    )
    print(f"MRR:       {mrr:.2f}")

    if args.min_recall is not None:
        if recall_at_k < args.min_recall:
            print(
                f"\n[gate] FAIL: recall@{args.k}={recall_at_k:.4f} "
                f"is below the required {args.min_recall:.4f}"
            )
            raise SystemExit(1)

        print(
            f"\n[gate] PASS: recall@{args.k}={recall_at_k:.4f} "
            f"meets the required {args.min_recall:.4f}"
        )

    if not args.answers:
        print(
            "\n(retrieval-only run; add --answers "
            "for generation metrics)"
        )
        return

    # -------- generation + judge metrics --------
    import os

    from ask import SYSTEM_PROMPT, build_prompt
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    llm_model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    print(f"[eval] llm={llm_model}")

    def llm(prompt: str, system: str | None = None) -> str:
        key = cache_key(llm_model, system or "", prompt)
        if key in cache:
            return cache[key]
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        out = None
        for attempt in range(4):
            try:
                out = client.chat.completions.create(
                    model=llm_model, messages=messages,
                    temperature=0.1).choices[0].message.content
                break
            except Exception as e:
                print(f"  [retry {attempt + 1}/4] {str(e)[:160]}")
                time.sleep(30)
        if out is None:
            raise RuntimeError("LLM call failed 4 times; try again later")
        cache[key] = out
        save_cache(cache)
        time.sleep(SLEEP_BETWEEN_LLM_CALLS)
        return out

    refusal_ok, faithful, correct = [], [], []
    for item, hits in zip(items, retrieved_per_q):
        ans = llm(build_prompt(item["question"], hits),
                  system=SYSTEM_PROMPT)

        if not item["answerable"]:
            ok = REFUSAL_TEXT.lower() in ans.lower()
            refusal_ok.append(1.0 if ok else 0.0)
            if not ok:
                print(f"  [no refusal] {item['question']}")
            continue

        context = "\n\n".join(h["text"][:1200] for h in hits)
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

    print(f"\n=== Generation (mode={args.mode}) ===")
    if faithful:
        print(f"faithful:  {sum(faithful)/len(faithful):.2f}")
        print(f"correct:   {sum(correct)/len(correct):.2f}")
    if refusal_ok:
        print(f"refusal:   {sum(refusal_ok)/len(refusal_ok):.2f}  "
              f"({int(sum(refusal_ok))}/{len(refusal_ok)})")


if __name__ == "__main__":
    main()
