"""Streamlit UI for the FastAPI Codebase Q&A RAG system.

Run locally:
    streamlit run app.py          # reads GROQ_API_KEY from .env

On Hugging Face Spaces, set GROQ_API_KEY as a Space secret. The Space has
no prebuilt index, so ensure_index() builds it from data/chunks.jsonl on
first boot (~3 min, cached afterwards).
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)
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
                   layout="centered")

st.markdown("""
<style>
    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.35);
        font-size: 0.85rem;
        transition: border-color 0.2s;
    }
    .stButton > button:hover { border-color: #f0b429; color: #f0b429; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    [data-testid="stExpander"] {
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.25);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="First boot: building the search index "
                                "(~3 min, one time only)...")
def ensure_index() -> bool:
    """Build the Chroma index from chunks.jsonl if it doesn't exist yet."""
    import chromadb
    client = chromadb.PersistentClient(path="data/chroma")
    try:
        if client.get_collection("chunks").count() > 0:
            return True
    except Exception:
        pass
    import index  # src/index.py
    index.main()
    return True


ensure_index()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("⚙️ How it works")
    st.markdown(
        "1. Your question is embedded (`bge-small-en-v1.5`)\n"
        "2. Top-5 chunks retrieved from ~1,350 AST-aware chunks "
        "(code, docs, issues)\n"
        "3. An LLM answers **only** from those chunks\n"
        "4. Citations link to the exact GitHub lines"
    )
    mode = st.selectbox("Retrieval mode", ["dense", "dense_rw", "hybrid"],
                        help="dense won a 7-configuration ablation; "
                             "others shown for comparison")
    st.divider()
    st.subheader("📊 Eval (42-question benchmark)")
    c1, c2 = st.columns(2)
    c1.metric("recall@5", "0.91")
    c2.metric("MRR", "0.71")
    c3, c4 = st.columns(2)
    c3.metric("correct", "0.91")
    c4.metric("refusal", "7/7")
    st.divider()
    st.markdown("[💻 Source & write-up]"
                "(https://github.com/islam-mamedov/codebase-rag)")
    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------- header ----------------
st.title("⚡ FastAPI Codebase Q&A")
st.caption("Ask anything about FastAPI's source code, docs, and GitHub "
           "issues. Every claim is cited. When the corpus doesn't know, "
           "it says so.")

# ---------------- helpers ----------------


@st.cache_data(show_spinner=False)
def cached_retrieve(question: str, mode: str) -> list[dict]:
    return retrieve(question, k=5, mode=mode)


def stream_answer(question: str, hits: list[dict]):
    """Yield the answer token by token (looks alive, no behavior change)."""
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY is not set.")
        st.stop()
    client = Groq(api_key=api_key)
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits)},
        ],
        temperature=0.1,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def render_sources(hits: list[dict]) -> None:
    st.markdown("##### 📎 Sources")
    for i, h in enumerate(hits, 1):
        meta = h["meta"]
        icon = {"code": "🧩", "doc": "📄", "issue": "🐛"}.get(
            meta["source_type"], "📄")
        label = (f"{icon} [{i}] {meta['path']}"
                 + (f" · {meta['symbol']}" if meta["symbol"] else ""))
        with st.expander(label):
            st.markdown(f"[Open on GitHub ↗]({meta['url']})")
            st.code(h["text"][:1200])


# ---------------- conversation state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])
        if msg.get("timing"):
            st.caption(msg["timing"])

# example buttons (only before the first question, to keep the video clean)
pending = None
if not st.session_state.messages:
    st.write("**Try one:**")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, use_container_width=True):
            pending = ex

prompt = st.chat_input("Ask about FastAPI's codebase...") or pending

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        t0 = time.perf_counter()
        with st.spinner("Searching 1,352 chunks..."):
            hits = cached_retrieve(prompt, mode)
        t_retrieve = time.perf_counter() - t0

        t1 = time.perf_counter()
        answer = st.write_stream(stream_answer(prompt, hits))
        t_answer = time.perf_counter() - t1

        timing = (f"retrieved in {t_retrieve:.2f}s · "
                  f"answered in {t_answer:.1f}s · mode: {mode}")
        st.caption(timing)
        render_sources(hits)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": hits,
        "timing": timing,
    })
