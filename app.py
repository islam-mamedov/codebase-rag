"""Streamlit UI for the FastAPI Codebase Q&A RAG system.

Run locally:
    export GROQ_API_KEY=gsk_...
    streamlit run app.py

On Hugging Face Spaces, set GROQ_API_KEY as a Space secret.
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv(override=True)
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

from ask import SYSTEM_PROMPT, build_prompt
from retrieval import retrieve

LLM_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

EXAMPLES = [
    "How do I return a custom status code from an endpoint?",
    "How does dependency injection work?",
    "Where is the APIRouter class defined?",
    "How do I connect FastAPI to MongoDB?",  # tests honest refusal
]

st.set_page_config(page_title="FastAPI Codebase Q&A", page_icon="⚡",
                   layout="wide")
st.title("⚡ FastAPI Codebase Q&A")
st.caption("Retrieval-augmented answers over FastAPI's source code, docs, "
           "and GitHub issues — every claim cited, honest refusals when "
           "the corpus doesn't know.")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. Your question is embedded (`bge-small-en-v1.5`)\n"
        "2. Top-5 chunks retrieved from ~1,350 AST-aware chunks "
        "(code, docs, issues)\n"
        "3. An LLM answers **only** from those chunks\n"
        "4. Citations link to the exact GitHub lines"
    )
    mode = st.selectbox("Retrieval mode", ["dense", "dense_rw", "hybrid"],
                        help="dense won the ablation; others shown for "
                             "comparison")
    st.divider()
    st.markdown(
        "**Eval results (42-question set)**\n\n"
        "recall@5 **0.91** · MRR **0.71**\n\n"
        "faithful **0.89** · correct **0.91** · refusal **7/7**"
    )
    # TODO: replace with your repo URL
    st.markdown("[Source & write-up](https://github.com/YOUR_USERNAME/codebase-rag)")


@st.cache_data(show_spinner=False)
def cached_retrieve(question: str, mode: str) -> list[dict]:
    return retrieve(question, k=5, mode=mode)


def generate(question: str, hits: list[dict]) -> str:
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY is not set.")
        st.stop()
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits)},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content


# --- example question buttons ---
st.write("Try one:")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state["question"] = ex

question = st.text_input("Ask about FastAPI's codebase",
                         key="question",
                         placeholder="How do I handle file uploads?")

if question:
    with st.spinner("Searching the codebase..."):
        hits = cached_retrieve(question, mode)
    with st.spinner("Writing the answer..."):
        answer = generate(question, hits)

    st.markdown(answer)

    st.divider()
    st.subheader("Sources")
    for i, h in enumerate(hits, 1):
        meta = h["meta"]
        label = (f"[{i}] {meta['source_type']} · {meta['path']}"
                 + (f" · {meta['symbol']}" if meta["symbol"] else ""))
        with st.expander(label):
            st.markdown(f"[Open on GitHub]({meta['url']})")
            st.code(h["text"][:1500])
