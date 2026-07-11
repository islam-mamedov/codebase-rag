"""Professional Streamlit UI for the FastAPI Codebase Q&A RAG system.

Run locally:
    streamlit run app.py

Required environment variable:
    GROQ_API_KEY

On Hugging Face Spaces, add GROQ_API_KEY under:
    Settings -> Variables and secrets
"""

import os
import sys
import time
from pathlib import Path
from textwrap import dedent
from typing import Iterator

from dotenv import load_dotenv

load_dotenv(override=True)

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import streamlit as st

from ask import SYSTEM_PROMPT, build_prompt
from retrieval import retrieve


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

LLM_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

GITHUB_URL = "https://github.com/islam-mamedov/codebase-rag"
SPACE_URL = (
    "https://huggingface.co/spaces/"
    "islam-mamedov/fastapi-codebase-qa"
)

EXAMPLES = [
    {
        "title": "Custom status codes",
        "question": "How do I return a custom status code from an endpoint?",
        "icon": "↗",
    },
    {
        "title": "Dependency injection",
        "question": "How does FastAPI's dependency injection system work?",
        "icon": "◫",
    },
    {
        "title": "Find a class",
        "question": "Where is the APIRouter class defined?",
        "icon": "⌘",
    },
    {
        "title": "Test refusal",
        "question": "How do I connect FastAPI to MongoDB?",
        "icon": "◇",
    },
]

MODE_LABELS = {
    "dense": "Dense retrieval",
    "dense_rw": "Dense + query rewriting",
    "hybrid": "Hybrid dense + BM25",
}

MODE_DESCRIPTIONS = {
    "dense": (
        "Recommended. Best evaluation result with "
        "0.91 recall@5 and 0.71 MRR."
    ),
    "dense_rw": (
        "Uses an LLM to expand the search query before retrieval. "
        "It matched dense recall but slightly reduced ranking quality."
    ),
    "hybrid": (
        "Combines dense retrieval with BM25 keyword search. "
        "Included for comparison with the selected dense pipeline."
    ),
}

SOURCE_ICONS = {
    "code": "⌘",
    "doc": "▤",
    "issue": "◉",
}


# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="FastAPI Codebase Q&A",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_html(content: str) -> None:
    """Render trusted static HTML without Markdown parsing."""
    st.html(dedent(content).strip())


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------

render_html(
    """
    <style>
        :root {
            --app-accent: #f59e0b;
            --app-accent-strong: #d97706;
            --app-accent-soft: rgba(245, 158, 11, 0.11);
            --app-green: #22c55e;
            --app-green-soft: rgba(34, 197, 94, 0.09);
            --app-border: rgba(128, 128, 128, 0.22);
            --app-border-strong: rgba(128, 128, 128, 0.36);
            --app-surface: rgba(128, 128, 128, 0.045);
            --app-surface-hover: rgba(128, 128, 128, 0.085);
        }

        html {
            scroll-behavior: smooth;
        }

        .block-container {
            max-width: 1100px;
            padding-top: 1.6rem;
            padding-bottom: 7rem;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(
                    circle at 60% -10%,
                    rgba(245, 158, 11, 0.08),
                    transparent 31rem
                );
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid var(--app-border);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.4rem;
        }

        [data-testid="stSidebarContent"] {
            padding-bottom: 2rem;
        }

        .hero-card {
            position: relative;
            overflow: hidden;
            padding: 2.35rem 2.5rem;
            margin-bottom: 1.6rem;
            border: 1px solid var(--app-border);
            border-radius: 24px;
            background:
                linear-gradient(
                    135deg,
                    rgba(245, 158, 11, 0.11),
                    rgba(128, 128, 128, 0.018)
                );
            box-shadow:
                0 18px 45px rgba(0, 0, 0, 0.045);
        }

        .hero-card::before {
            content: "";
            position: absolute;
            width: 250px;
            height: 250px;
            right: -110px;
            top: -125px;
            border-radius: 999px;
            background: rgba(245, 158, 11, 0.11);
            filter: blur(1px);
            pointer-events: none;
        }

        .hero-card::after {
            content: "";
            position: absolute;
            width: 130px;
            height: 130px;
            left: -70px;
            bottom: -75px;
            border-radius: 999px;
            background: rgba(245, 158, 11, 0.06);
            pointer-events: none;
        }

        .hero-eyebrow {
            position: relative;
            z-index: 1;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
            padding: 0.42rem 0.75rem;
            border: 1px solid rgba(34, 197, 94, 0.28);
            border-radius: 999px;
            background: var(--app-green-soft);
            font-size: 0.78rem;
            font-weight: 700;
        }

        .status-dot {
            width: 7px;
            height: 7px;
            display: inline-block;
            border-radius: 999px;
            background: var(--app-green);
            box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.11);
        }

        .hero-title {
            position: relative;
            z-index: 1;
            max-width: 790px;
            margin: 0;
            font-size: clamp(2.15rem, 5vw, 3.5rem);
            line-height: 1.04;
            letter-spacing: -0.055em;
            font-weight: 780;
        }

        .hero-subtitle {
            position: relative;
            z-index: 1;
            max-width: 790px;
            margin: 1.05rem 0 1.45rem 0;
            font-size: 1.03rem;
            line-height: 1.72;
            opacity: 0.75;
        }

        .hero-pills {
            position: relative;
            z-index: 1;
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
        }

        .hero-pill {
            padding: 0.5rem 0.78rem;
            border: 1px solid var(--app-border);
            border-radius: 999px;
            background: rgba(128, 128, 128, 0.035);
            font-size: 0.79rem;
            font-weight: 650;
        }

        .section-label {
            margin-top: 1.55rem;
            margin-bottom: 0.28rem;
            font-size: 0.75rem;
            font-weight: 750;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            opacity: 0.53;
        }

        .section-title {
            margin: 0 0 0.95rem 0;
            font-size: 1.3rem;
            font-weight: 730;
            letter-spacing: -0.025em;
        }

        .welcome-note {
            padding: 0.92rem 1rem;
            margin-bottom: 1rem;
            border-left: 3px solid var(--app-accent);
            border-radius: 0 13px 13px 0;
            background: var(--app-accent-soft);
            font-size: 0.88rem;
            line-height: 1.6;
        }

        .stButton > button {
            min-height: 3.35rem;
            border: 1px solid var(--app-border);
            border-radius: 14px;
            background: var(--app-surface);
            font-weight: 620;
            text-align: left;
            white-space: normal;
            transition:
                transform 0.15s ease,
                border-color 0.15s ease,
                background 0.15s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            border-color: rgba(245, 158, 11, 0.62);
            background: var(--app-accent-soft);
            color: inherit;
        }

        .stButton > button:focus {
            border-color: var(--app-accent);
            box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.13);
        }

        [data-testid="stLinkButton"] a {
            border-radius: 12px;
            border-color: var(--app-border);
            background: var(--app-surface);
        }

        [data-testid="stLinkButton"] a:hover {
            border-color: rgba(245, 158, 11, 0.62);
            background: var(--app-accent-soft);
            color: inherit;
        }

        [data-testid="stMetric"] {
            padding: 0.88rem 0.92rem;
            border: 1px solid var(--app-border);
            border-radius: 14px;
            background: var(--app-surface);
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.75rem;
            opacity: 0.65;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.35rem;
            font-weight: 740;
            letter-spacing: -0.04em;
        }

        [data-testid="stChatMessage"] {
            padding: 1rem 1.1rem;
            margin-bottom: 0.75rem;
            border: 1px solid var(--app-border);
            border-radius: 18px;
            background: rgba(128, 128, 128, 0.022);
        }

        [data-testid="stChatMessage"]:has(
            [data-testid="chatAvatarIcon-user"]
        ) {
            border-color: rgba(245, 158, 11, 0.22);
            background: rgba(245, 158, 11, 0.05);
        }

        [data-testid="stChatInput"] {
            border-radius: 16px;
            box-shadow: none;
        }

        [data-testid="stChatInput"] textarea {
            min-height: 54px;
        }

        [data-testid="stExpander"] {
            margin-bottom: 0.5rem;
            border: 1px solid var(--app-border);
            border-radius: 13px;
            overflow: hidden;
            background: rgba(128, 128, 128, 0.022);
        }

        [data-testid="stExpander"]:hover {
            border-color: var(--app-border-strong);
        }

        [data-testid="stCodeBlock"] {
            border-radius: 12px;
        }

        [data-baseweb="select"] > div {
            border-radius: 12px;
        }

        .source-heading {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 1.15rem;
            margin-bottom: 0.6rem;
        }

        .source-heading-title {
            font-size: 0.9rem;
            font-weight: 720;
        }

        .source-heading-count {
            font-size: 0.74rem;
            opacity: 0.55;
        }

        .app-footer {
            margin-top: 2.2rem;
            padding-top: 1.2rem;
            border-top: 1px solid var(--app-border);
            text-align: center;
            font-size: 0.78rem;
            opacity: 0.55;
        }

        @media (max-width: 700px) {
            .block-container {
                padding-top: 0.9rem;
            }

            .hero-card {
                padding: 1.45rem;
                border-radius: 18px;
            }

            .hero-title {
                font-size: 2rem;
            }

            .hero-subtitle {
                font-size: 0.94rem;
            }
        }
    </style>
    """
)


# ---------------------------------------------------------------------
# Index initialization
# ---------------------------------------------------------------------

@st.cache_resource(
    show_spinner=(
        "Preparing the search index. "
        "The first startup can take approximately three minutes."
    )
)
def ensure_index() -> bool:
    """Build the Chroma index when it does not already exist."""
    import chromadb

    index_path = ROOT_DIR / "data" / "chroma"
    client = chromadb.PersistentClient(path=str(index_path))

    try:
        collection = client.get_collection("chunks")

        if collection.count() > 0:
            return True
    except Exception:
        pass

    import index

    index.main()
    return True


ensure_index()


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚡ Codebase Q&A")

    st.caption(
        "Evaluation-driven retrieval over FastAPI source code, "
        "documentation and resolved issues."
    )

    st.divider()

    st.markdown("### Retrieval")

    mode = st.selectbox(
        "Search strategy",
        options=["dense", "dense_rw", "hybrid"],
        format_func=lambda value: MODE_LABELS[value],
        help=(
            "Dense retrieval achieved the strongest overall "
            "evaluation result."
        ),
    )

    st.caption(MODE_DESCRIPTIONS[mode])

    st.divider()

    st.markdown("### Evaluation snapshot")
    st.caption("42-question hand-labelled benchmark")

    metric_col_1, metric_col_2 = st.columns(2)
    metric_col_1.metric("Recall@5", "0.91")
    metric_col_2.metric("MRR", "0.71")

    metric_col_3, metric_col_4 = st.columns(2)
    metric_col_3.metric("Correctness", "0.91")
    metric_col_4.metric("Refusals", "7 / 7")

    st.caption(
        "Every push runs unit tests and fails CI when "
        "recall@5 falls below 0.90."
    )

    st.divider()

    st.markdown("### Project")

    st.link_button(
        "⌘ View source code",
        GITHUB_URL,
        use_container_width=True,
    )

    st.link_button(
        "↗ Open live deployment",
        SPACE_URL,
        use_container_width=True,
    )

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------

render_html(
    """
    <div class="hero-card">
        <div class="hero-eyebrow">
            <span class="status-dot"></span>
            Live evaluation-driven RAG system
        </div>

        <h1 class="hero-title">
            Understand FastAPI through its actual codebase.
        </h1>

        <p class="hero-subtitle">
            Ask technical questions and receive answers grounded in FastAPI
            source code, documentation and resolved GitHub issues. Each
            response includes retrieved evidence, while unsupported questions
            are refused instead of guessed.
        </p>

        <div class="hero-pills">
            <span class="hero-pill">1,352 indexed chunks</span>
            <span class="hero-pill">AST-aware code retrieval</span>
            <span class="hero-pill">Source-linked answers</span>
            <span class="hero-pill">CI evaluation gate</span>
        </div>
    </div>
    """
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_retrieve(
    question: str,
    retrieval_mode: str,
) -> list[dict]:
    """Retrieve the top five chunks for a question."""
    return retrieve(
        question,
        k=5,
        mode=retrieval_mode,
    )


def stream_answer(
    question: str,
    hits: list[dict],
) -> Iterator[str]:
    """Stream a grounded answer from Groq."""
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        st.error(
            "The GROQ_API_KEY environment variable is not configured."
        )
        st.stop()

    client = Groq(api_key=api_key)

    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": build_prompt(question, hits),
                },
            ],
            temperature=0.1,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content

            if delta:
                yield delta

    except Exception as error:
        st.error(
            "The answer service encountered an error. "
            "Please try the question again."
        )

        yield (
            "\n\nThe answer could not be generated because the "
            "language-model request failed."
        )

        print(f"[Groq error] {error}")


def render_sources(hits: list[dict]) -> None:
    """Render retrieved evidence below an answer."""
    render_html(
        f"""
        <div class="source-heading">
            <span class="source-heading-title">
                Retrieved evidence
            </span>

            <span class="source-heading-count">
                {len(hits)} sources
            </span>
        </div>
        """
    )

    for index_number, hit in enumerate(hits, start=1):
        metadata = hit.get("meta", {})

        source_type = metadata.get("source_type", "doc")
        source_icon = SOURCE_ICONS.get(source_type, "▤")

        source_path = metadata.get("path", "Unknown source")
        source_symbol = metadata.get("symbol", "")
        source_url = metadata.get("url", "")
        source_score = hit.get("score")

        label = f"{source_icon}  {index_number}. {source_path}"

        if source_symbol:
            label += f"  ·  {source_symbol}"

        with st.expander(label):
            detail_col_1, detail_col_2 = st.columns([3, 1])

            with detail_col_1:
                st.caption(
                    f"Source type: {source_type.capitalize()}"
                )

            with detail_col_2:
                if source_score is not None:
                    st.caption(
                        f"Similarity: {source_score:.3f}"
                    )

            if source_url:
                st.markdown(
                    f"[Open original source on GitHub ↗]"
                    f"({source_url})"
                )

            code_language = (
                "python"
                if source_type == "code"
                else "text"
            )

            source_text = hit.get("text", "")

            st.code(
                source_text[:1600],
                language=code_language,
                wrap_lines=True,
            )


# ---------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------------------
# Existing conversation
# ---------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("timing"):
            st.caption(message["timing"])

        if message.get("sources"):
            render_sources(message["sources"])


# ---------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------

pending_question = None

if not st.session_state.messages:
    render_html(
        """
        <div class="section-label">
            Start exploring
        </div>

        <div class="section-title">
            Ask a question or try an example
        </div>
        """
    )

    render_html(
        """
        <div class="welcome-note">
            The assistant searches the indexed FastAPI corpus before
            answering. Questions about unrelated integrations should
            produce an honest refusal.
        </div>
        """
    )

    first_row = st.columns(2)
    second_row = st.columns(2)

    example_columns = [
        first_row[0],
        first_row[1],
        second_row[0],
        second_row[1],
    ]

    for column, example in zip(
        example_columns,
        EXAMPLES,
    ):
        with column:
            button_label = (
                f"{example['icon']}  {example['title']}\n\n"
                f"{example['question']}"
            )

            if st.button(
                button_label,
                key=f"example-{example['title']}",
                use_container_width=True,
            ):
                pending_question = example["question"]


# ---------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------

typed_prompt = st.chat_input(
    "Ask about FastAPI's code, documentation or behaviour..."
)

prompt = typed_prompt or pending_question


# ---------------------------------------------------------------------
# Generate response
# ---------------------------------------------------------------------

if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        retrieval_started = time.perf_counter()

        with st.spinner(
            "Searching 1,352 code, documentation and issue chunks..."
        ):
            hits = cached_retrieve(
                prompt,
                mode,
            )

        retrieval_seconds = (
            time.perf_counter() - retrieval_started
        )

        generation_started = time.perf_counter()

        answer = st.write_stream(
            stream_answer(
                prompt,
                hits,
            )
        )

        generation_seconds = (
            time.perf_counter() - generation_started
        )

        timing = (
            f"Retrieved in {retrieval_seconds:.2f}s"
            f" · Generated in {generation_seconds:.1f}s"
            f" · Strategy: {MODE_LABELS[mode]}"
        )

        st.caption(timing)
        render_sources(hits)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": hits,
            "timing": timing,
        }
    )


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

render_html(
    """
    <div class="app-footer">
        Built with Streamlit, ChromaDB, BGE embeddings and Groq
        · Evaluation-gated on every GitHub push
    </div>
    """
)