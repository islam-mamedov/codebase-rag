"""Ask a question about the codebase — the full RAG pipeline, v0 (baseline).

Flow:
    1. Embed your question with the same model used for the chunks
    2. Ask ChromaDB for the 5 most similar chunks (dense retrieval only, for now)
    3. Hand those chunks to an LLM and have it answer USING ONLY THEM
    4. Print the answer plus links to the sources

Usage:
    export GROQ_API_KEY=gsk_...          # free key from console.groq.com
    python src/ask.py "How do I return a custom status code?"
    python src/ask.py --show-chunks "..."   # also print retrieved chunks
"""

import argparse
import os
import sys
from pathlib import Path

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# BGE models retrieve better when queries carry this instruction prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
LLM_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
TOP_K = 5

SYSTEM_PROMPT = """\
You are a precise assistant answering questions about the FastAPI codebase.
Answer ONLY from the numbered context chunks provided. Rules:
- Cite chunks inline like [1] or [2][3] after each claim they support.
- If the context does not contain the answer, say exactly:
  "I couldn't find this in the indexed codebase." Do not guess.
- Prefer short code examples when the context contains them.
- Be concise."""


def retrieve(question: str, k: int) -> list[dict]:
    model = SentenceTransformer(EMBED_MODEL)
    query_emb = model.encode(QUERY_PREFIX + question,
                             normalize_embeddings=True)
    client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma"))
    collection = client.get_collection("chunks")
    res = collection.query(query_embeddings=[query_emb.tolist()], n_results=k)
    return [{
        "text": doc,
        "meta": meta,
        "distance": dist,
    } for doc, meta, dist in zip(res["documents"][0],
                                 res["metadatas"][0],
                                 res["distances"][0])]


def build_prompt(question: str, hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[{i}] ({h['meta']['source_type']}: "
                     f"{h['meta']['path']})\n{h['text']}")
    context = "\n\n---\n\n".join(parts)
    return f"Context chunks:\n\n{context}\n\nQuestion: {question}"


def answer(question: str, hits: list[dict]) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("Set GROQ_API_KEY first (free key at console.groq.com).")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits)},
        ],
        temperature=0.1,  # low = factual, less creative
    )
    return response.choices[0].message.content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--show-chunks", action="store_true",
                        help="print retrieved chunks (debugging/learning)")
    args = parser.parse_args()

    hits = retrieve(args.question, args.k)

    if args.show_chunks:
        for i, h in enumerate(hits, 1):
            print(f"\n=== [{i}] dist={h['distance']:.3f} "
                  f"{h['meta']['path']} :: {h['meta']['symbol']} ===")
            print(h["text"][:500])
        print("\n" + "=" * 60)

    print("\n" + answer(args.question, hits))

    print("\nSources:")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] {h['meta']['url']}")


if __name__ == "__main__":
    main()
