---
title: FastAPI Codebase Q&A
emoji: ⚡
colorFrom: yellow
colorTo: gray
sdk: streamlit
app_file: app.py
pinned: false
---

<div align="center">

# ⚡ FastAPI Codebase Q&A

### An evaluation-driven RAG system for exploring a real production codebase

Ask technical questions about FastAPI and receive answers grounded in its source code, documentation, and resolved GitHub issues.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/islam-mamedov/fastapi-codebase-qa)
[![GitHub](https://img.shields.io/badge/Source_Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/islam-mamedov/codebase-rag)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Tests and Retrieval Eval](https://github.com/islam-mamedov/codebase-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/islam-mamedov/codebase-rag/actions/workflows/ci.yml)

<br>

<img
  src="assets/codebase-rag-demo.gif"
  alt="FastAPI Codebase Q&A retrieving evidence and generating a grounded answer"
  width="1000"
/>

<br>

<p>
  <b>Question → Retrieval → Grounded answer → Source evidence</b>
</p>

</div>

---

## Overview

FastAPI Codebase Q&A is a Retrieval-Augmented Generation system designed to answer questions about the FastAPI repository.

Instead of relying only on an LLM's general knowledge, the system searches a custom knowledge base built from:

- FastAPI source code
- English documentation
- Closed GitHub issues

The retrieved evidence is passed to the language model, which generates an answer with references to the relevant files, symbols, and line numbers.

The project was built around one principle:

> Every important retrieval decision should be measured rather than guessed.

A hand-labelled evaluation set was used to compare dense retrieval, hybrid search, reranking, and query rewriting before selecting the final production configuration.

## Live Demo

Try the deployed application:

**[Open FastAPI Codebase Q&A on Hugging Face Spaces](https://huggingface.co/spaces/islam-mamedov/fastapi-codebase-qa)**

Example questions:

```text
How do I return a custom status code in FastAPI?
```

```text
Where is APIRouter defined?
```

```text
How does FastAPI validate request bodies?
```

```text
Can FastAPI automatically deploy my application to AWS?
```

The final example tests the system's refusal behaviour. When the indexed corpus does not contain enough evidence, the application should say so instead of inventing an answer.

---

## What the System Does

The application follows a complete RAG workflow:

1. Ingests content from the FastAPI repository.
2. Splits code, documentation, and issues using content-aware chunking.
3. Converts chunks into vector embeddings.
4. Stores and retrieves relevant chunks using ChromaDB.
5. Passes the retrieved evidence to an LLM.
6. Generates a grounded answer with citations.
7. Refuses questions that cannot be answered from the available evidence.
8. Evaluates retrieval and answer quality using a labelled benchmark.

---

## Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│                         FastAPI Repository                           │
│                                                                      │
│              Source Code · Documentation · GitHub Issues             │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                              Ingestion                               │
│                                                                      │
│       Repository cloning · GitHub API · Resumable issue fetching     │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                              Chunking                                │
│                                                                      │
│       AST-aware code · Markdown-aware docs · Issue-aware chunks      │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                               Indexing                               │
│                                                                      │
│       BGE embeddings · ChromaDB vector store · Chunk metadata        │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                              Retrieval                               │
│                                                                      │
│          Dense semantic search selected through evaluation           │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                              Generation                              │
│                                                                      │
│        Groq LLM · Grounded answers · Citations · Refusal rules       │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                            Evaluation                                │
│                                                                      │
│    Recall@5 · MRR · Faithfulness · Correctness · Refusal accuracy    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Indexed Corpus

The knowledge base contains:

| Source | Quantity |
|---|---:|
| FastAPI source files | 46 |
| English documentation files | 161 |
| Closed GitHub issues | 175 |
| Total chunks | 1,352 |

Chunk distribution:

| Chunk type | Quantity |
|---|---:|
| Code chunks | 363 |
| Documentation chunks | 814 |
| Issue chunks | 175 |

Every chunk includes metadata such as:

- File path
- Content type
- Symbol name
- Start and end lines
- Repository information
- Context header

Example context header:

```text
# File: fastapi/routing.py | Symbol: APIRouter
```

This metadata allows the application to produce answers that point back to the relevant part of the repository.

---

## Content-Aware Chunking

Different content types require different chunking strategies.

### Source Code

Python files are parsed using Tree-sitter.

The chunker creates separate chunks for:

- Functions
- Classes
- Methods
- Module-level content

Large classes are split by method when necessary instead of being stored as one oversized block.

This preserves the structure of the code and improves retrieval for questions such as:

```text
Where is the APIRouter class defined?
```

### Documentation

Markdown documents are divided using heading and section boundaries.

This avoids splitting related explanations in the middle of a section and preserves useful context.

### GitHub Issues

Issues are stored with their:

- Title
- Body
- Resolution context
- Relevant metadata

Issue-aware chunking helps answer behavioural questions that may not be explained clearly in the source code or documentation.

---

## Evaluation Dataset

The system is evaluated using 42 manually labelled questions.

| Category | Questions |
|---|---:|
| API usage | 16 |
| Behaviour | 11 |
| Code location | 8 |
| Unanswerable | 7 |
| **Total** | **42** |

Each answerable question includes a gold reference identifying the file or symbol expected to contain the answer.

The evaluation measures two separate parts of the system.

### Retrieval Metrics

- **Recall@5:** whether a correct chunk appears within the first five retrieved results
- **MRR:** how highly the first correct result is ranked

### Generation Metrics

- **Faithfulness:** whether the answer is supported by the retrieved context
- **Correctness:** whether the response answers the question accurately
- **Refusal accuracy:** whether unsupported questions are rejected correctly

---

## Retrieval Experiments

Several retrieval configurations were tested before selecting the production approach.

| Configuration | Recall@5 | MRR | Result |
|---|---:|---:|---|
| **Dense retrieval — BGE Small** | **0.91** | **0.71** | **Selected for production** |
| Hybrid dense + BM25 using RRF | 0.86 | 0.70 | Lower recall |
| Hybrid weighted RRF at 2:1 | 0.86 | 0.71 | No recall improvement |
| Dense + BGE reranker base | 0.91 | 0.63 | Ranking became worse |
| Dense + BGE reranker v2-m3 | 0.83 | 0.67 | Lower recall and MRR |
| Hybrid + BGE reranker v2-m3 | 0.86 | 0.68 | No improvement |
| Dense + LLM query rewriting | 0.91 | 0.69 | No measurable gain |

### Why Dense Retrieval Won

Dense retrieval produced the best overall combination of recall and ranking quality.

Diagnostic experiments showed:

```text
Dense recall@20:  0.97
Hybrid recall@20: 0.97
```

BM25 did not introduce additional correct candidates. Instead, common words such as `request`, `body`, and `json` caused unrelated GitHub issue chunks to move above more useful code and documentation chunks.

Because hybrid retrieval added noise without adding new correct results, the simpler dense configuration was selected.

### Why Reranking Was Rejected

Both rerankers frequently pushed raw code chunks lower in the results.

This especially affected code-location questions.

A source-code definition may directly contain the answer without explaining it in natural language. Cross-encoders often prefer passages that discuss a topic rather than code that implements it.

The retrievers were already finding strong candidates at `k=20`, but the rerankers were reordering them incorrectly.

### Why Query Rewriting Was Rejected

LLM query rewriting produced:

```text
Recall@5: 0.91
MRR:      0.69
```

It did not improve recall over the original dense query and slightly reduced ranking quality.

Some rewritten queries added useful-looking but incorrect identifiers, creating semantic drift. The feature was therefore not included in the deployed retrieval pipeline.

---

## Generation Results

The reported generation evaluation used dense retrieval with a Groq-hosted LLM.

| Metric | Score |
|---|---:|
| Faithfulness | 0.89 |
| Correctness | 0.91 |
| Refusal accuracy | 7/7 |

### Refusal Behaviour

Refusal accuracy was one of the most sensitive parts of the project.

Across three prompt versions, the result changed from:

```text
6/7 → 0/7 → 7/7
```

The main problem was that retrieved chunks could be related to the question without actually containing enough information to answer it.

The final prompt explicitly explains this distinction to the model:

```text
The context chunks are search results. They may be loosely related to the
question without actually containing the answer.
```

This small prompt change prevented the model from treating all retrieved content as valid evidence.

### Why a Similarity Threshold Was Not Used

A retrieval-score threshold was tested as an alternative refusal mechanism.

However, the score ranges overlapped:

```text
Answerable questions:   0.74–0.89
Unanswerable questions: 0.72–0.79
```

A high similarity score only shows that a chunk is related to the question. It does not prove that the chunk contains the answer.

For this reason, refusal is handled through evidence-aware prompting rather than a fixed similarity threshold.

---

## Engineering Features

### AST-Aware Code Processing

Code is chunked using syntax structure instead of fixed character or token windows.

### Grounded Citations

Answers include references to the source files and line ranges used to generate the response.

### Honest Refusals

The system is instructed to refuse unsupported questions rather than produce confident but ungrounded answers.

### Resumable Ingestion

Previously downloaded issues are skipped, allowing interrupted ingestion runs to continue without restarting from the beginning.

### Evaluation Caching

LLM answers, judgments, and query rewrites are cached using the model and prompt as part of the cache key.

This reduces repeated API usage and makes evaluation reruns faster.

### Rate-Limit Handling

API calls use retries and backoff to handle temporary rate limits and service interruptions.

### Self-Building Deployment

The Hugging Face Space can rebuild its vector index from committed chunk data during startup.

Large generated database files do not need to be stored in Git.

### Regression Testing

The project includes unit tests for the chunking pipeline and an evaluation set that can be used as a retrieval regression gate.

---

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| User interface | Streamlit |
| Vector database | ChromaDB |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Sparse retrieval experiments | BM25 |
| Code parsing | Tree-sitter |
| LLM provider | Groq |
| GitHub ingestion | PyGithub |
| Testing | Pytest |
| Deployment | Hugging Face Spaces |

---

## Project Structure

```text
codebase-rag/
├── app.py
├── README.md
├── LICENSE
├── requirements.txt
├── .env
├── .gitignore
│
├── data/
│   ├── chunks.jsonl
│   ├── evaluation data
│   └── cached results
│
├── src/
│   ├── ask.py
│   ├── chunk.py
│   ├── eval.py
│   ├── index.py
│   ├── ingest.py
│   └── retrieval.py
│
└── tests/
    └── test_chunk.py
```

### Main Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit chat interface |
| `src/ingest.py` | Downloads repository content and GitHub issues |
| `src/chunk.py` | Creates code, documentation, and issue chunks |
| `src/index.py` | Builds the vector index |
| `src/retrieval.py` | Runs retrieval strategies |
| `src/ask.py` | Generates grounded answers |
| `src/eval.py` | Runs retrieval and generation evaluation |
| `tests/test_chunk.py` | Tests chunking behaviour |

---

## Local Installation

### 1. Clone the Repository

```bash
git clone https://github.com/islam-mamedov/codebase-rag.git
cd codebase-rag
```

### 2. Create a Virtual Environment

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For ingestion, development, and testing:

```bash
pip install PyGithub tree-sitter tree-sitter-python pytest
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GITHUB_TOKEN=your_github_token
```

`GROQ_API_KEY` is required for answer generation.

`GITHUB_TOKEN` is used when downloading GitHub issues during ingestion.

Never commit `.env` or expose API keys in terminal screenshots, documentation, or chat messages.

You can confirm that `.env` is ignored with:

```bash
git check-ignore .env
```

---

## Running the Application

Start the Streamlit interface:

```bash
streamlit run app.py
```

Then open the local URL displayed in the terminal.

It is usually:

```text
http://localhost:8501
```

---

## Running the Pipeline

### Ingest FastAPI Content

```bash
python src/ingest.py --repo fastapi/fastapi
```

### Create Chunks

```bash
python src/chunk.py
```

### Build the Vector Index

```bash
python src/index.py
```

### Ask a Question from the Terminal

```bash
python src/ask.py "How do I return a custom status code?"
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Running Evaluation

Run dense retrieval evaluation:

```bash
python src/eval.py --mode dense
```

Run retrieval and answer-generation evaluation:

```bash
python src/eval.py --mode dense --answers
```

The evaluation results can be used to compare retrieval changes against the current baseline:

```text
Recall@5: 0.91
MRR:      0.71
```

A future CI pipeline could reject changes that reduce retrieval performance below these values.

---

## Current Limitations

### Common-Token Queries

Some questions contain words that appear throughout the repository.

For example:

```text
Where are the Query, Path, and Body parameter functions defined?
```

The relevant file is:

```text
fastapi/param_functions.py
```

However, terms such as `query`, `path`, and `body` appear in many unrelated chunks.

LLM query rewriting was tested but did not improve aggregate retrieval performance. A stronger next step would be symbol-aware retrieval, exact identifier matching, or metadata filtering.

### Documentation Include Directives

FastAPI documentation sometimes references external examples using include directives such as:

```text
{* ../../docs_src/... *}
```

The current chunker does not automatically insert the referenced source code into the documentation chunk.

As a result, some retrieved documentation explains an example without containing the complete implementation.

### LLM-as-Judge Limitations

Generation metrics are useful for comparison, but they are not absolute.

During manual review, at least one response marked incorrect appeared to be valid. The judge penalised it because part of the supporting code was implied by truncated context rather than shown directly.

For this reason, judge scores are treated as directional measurements and are combined with manual inspection.

### Repository-Specific Pipeline

The current system is designed around FastAPI.

Some ingestion and metadata assumptions would need to be generalised before the project could support arbitrary repositories.

---

## Future Improvements

- Resolve documentation include directives during ingestion
- Add symbol-aware and identifier-aware retrieval
- Make the ingestion pipeline repository-agnostic
- Compare BGE Small with larger and code-specific embedding models
- Add GitHub Discussions as another knowledge source
- Add conversation-aware follow-up questions
- Add automated retrieval regression tests in CI
- Track latency and token usage
- Add answer feedback and failure logging
- Evaluate retrieval separately for code, documentation, and issue questions

---

## Key Lessons

This project produced several practical findings:

1. More retrieval components do not automatically produce better results.
2. Hybrid search can reduce quality when sparse retrieval adds noisy candidates.
3. Rerankers trained on natural-language passages may perform poorly on raw code.
4. Similarity does not guarantee answerability.
5. Refusal behaviour must be evaluated directly.
6. Query rewriting can introduce semantic drift.
7. A small labelled evaluation set can prevent weak architectural choices from reaching production.

---

## License

This project is released under the [MIT License](LICENSE).

```text
Copyright (c) 2026 Islam Mamedov
```

---

## Author

**Islam Mamedov**

AI Engineer focused on retrieval systems, agentic AI, computer vision, and production-oriented machine learning applications.

- GitHub: [islam-mamedov](https://github.com/islam-mamedov)
- Live project: [FastAPI Codebase Q&A](https://huggingface.co/spaces/islam-mamedov/fastapi-codebase-qa)

---

<div align="center">

Built as an evaluation-driven AI engineering project.

**[Try the live demo](https://huggingface.co/spaces/islam-mamedov/fastapi-codebase-qa)**

</div>

## License

This project is released under the [MIT License](LICENSE).

```text
Copyright (c) 2026 Islam Mamedov