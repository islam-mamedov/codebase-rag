"""Build the search index: embed every chunk and store it in ChromaDB.

What "embedding" means: the model converts each chunk's text into a vector
(384 numbers) that captures its MEANING. Chunks about similar things end up
with similar vectors, so later we can find relevant chunks even when the
question uses different words than the code/docs do.

Reads  data/chunks.jsonl   (from chunk.py)
Writes data/chroma/        (persistent vector database)

Usage:
    python src/index.py
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
COLLECTION = "chunks"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # small, strong, runs fine on CPU
BATCH = 128


def main() -> None:
    chunks = [json.loads(line)
              for line in (DATA_DIR / "chunks.jsonl").read_text().splitlines()]
    print(f"[index] {len(chunks)} chunks to embed")

    print(f"[index] loading embedding model {EMBED_MODEL} "
          "(first run downloads ~130MB)...")
    model = SentenceTransformer(EMBED_MODEL)

    # Embed all chunk texts. normalize_embeddings=True -> cosine similarity
    # becomes a simple dot product, which is what Chroma will compute.
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=BATCH,
                              show_progress_bar=True,
                              normalize_embeddings=True)

    client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma"))
    # Start fresh each run so re-indexing never leaves stale chunks behind
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION,
                                          metadata={"hnsw:space": "cosine"})

    # Chroma metadata can't hold None values, so default line numbers to 0
    metadatas = [{
        "source_type": c["source_type"],
        "path": c["path"],
        "symbol": c["symbol"] or "",
        "url": c["url"],
        "start_line": c["start_line"] or 0,
        "end_line": c["end_line"] or 0,
    } for c in chunks]

    for i in range(0, len(chunks), BATCH):
        j = i + BATCH
        collection.add(
            ids=[c["id"] for c in chunks[i:j]],
            embeddings=embeddings[i:j].tolist(),
            documents=texts[i:j],
            metadatas=metadatas[i:j],
        )
    print(f"[done] indexed {collection.count()} chunks -> {DATA_DIR / 'chroma'}")


if __name__ == "__main__":
    main()
