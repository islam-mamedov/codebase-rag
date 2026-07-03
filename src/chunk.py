"""Chunk ingested content into retrieval units.

Reads data/manifest.json + files produced by ingest.py.
Writes data/chunks.jsonl — one JSON chunk per line.

Chunk schema (same for all source types):
    id           stable unique id
    text         what gets embedded (includes a context header)
    source_type  code | doc | issue
    path         file path in the repo (or issues/N.json)
    symbol       function/class name, doc heading, or issue title
    url          link to the exact GitHub location
    start_line / end_line  (code and docs only)

Usage:
    python src/chunk.py
"""

import json
import subprocess
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

DATA_DIR = Path("data")
REPO_DIR = DATA_DIR / "repo"
MAX_CLASS_LINES = 150   # classes bigger than this get split into methods
MIN_CHUNK_CHARS = 50    # drop trivial fragments
MAX_ISSUE_BODY = 3000
MAX_COMMENT = 1000

PY_LANGUAGE = Language(tspython.language())
PARSER = Parser(PY_LANGUAGE)


# ---------------------------------------------------------------- helpers

def default_branch() -> str:
    out = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse",
                          "--abbrev-ref", "HEAD"],
                         capture_output=True, text=True)
    return out.stdout.strip() or "master"


def github_url(repo: str, branch: str, path: str,
               start: int | None = None, end: int | None = None) -> str:
    url = f"https://github.com/{repo}/blob/{branch}/{path}"
    if start is not None:
        url += f"#L{start}-L{end}"
    return url


# ---------------------------------------------------------------- code

def chunk_code_file(source: bytes, rel_path: str, repo: str,
                    branch: str) -> list[dict]:
    """AST-aware chunking: one chunk per top-level function/class.

    Large classes are split into per-method chunks. Module-level leftovers
    (imports, constants) become one '<module>' chunk.
    """
    tree = PARSER.parse(source)
    text_lines = source.decode(errors="replace").splitlines()
    chunks: list[dict] = []
    covered: set[int] = set()  # 1-based line numbers already in a chunk

    def node_text(node) -> str:
        return source[node.start_byte:node.end_byte].decode(errors="replace")

    def emit(node, symbol: str, context: str = "") -> None:
        start, end = node.start_point[0] + 1, node.end_point[0] + 1
        header = f"# File: {rel_path}"
        if context:
            header += f" | {context}"
        header += f" | Symbol: {symbol}\n"
        body = node_text(node)
        if len(body) < MIN_CHUNK_CHARS:
            covered.update(range(start, end + 1))
            return
        chunks.append({
            "id": f"{rel_path}::{symbol}::L{start}",
            "text": header + body,
            "source_type": "code",
            "path": rel_path,
            "symbol": symbol,
            "url": github_url(repo, branch, rel_path, start, end),
            "start_line": start,
            "end_line": end,
        })
        covered.update(range(start, end + 1))

    def unwrap(node):
        """decorated_definition wraps the real def; we want the decorator
        text included but the inner node's name."""
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type in ("function_definition", "class_definition"):
                    return child, node
        return node, node

    def symbol_name(def_node) -> str:
        name = def_node.child_by_field_name("name")
        return name.text.decode() if name else "?"

    for node in tree.root_node.children:
        inner, outer = unwrap(node)

        if inner.type == "function_definition":
            emit(outer, symbol_name(inner))

        elif inner.type == "class_definition":
            cls_name = symbol_name(inner)
            n_lines = inner.end_point[0] - inner.start_point[0] + 1
            if n_lines <= MAX_CLASS_LINES:
                emit(outer, cls_name)
            else:
                # Big class: one chunk per method
                body = inner.child_by_field_name("body")
                for child in (body.children if body else []):
                    m_inner, m_outer = unwrap(child)
                    if m_inner.type == "function_definition":
                        emit(m_outer, f"{cls_name}.{symbol_name(m_inner)}",
                             context=f"Class: {cls_name}")

    # Module-level leftovers: imports, constants, top-level statements
    leftover = [i for i in range(1, len(text_lines) + 1)
                if i not in covered and text_lines[i - 1].strip()]
    if leftover:
        body = "\n".join(text_lines[i - 1] for i in leftover)
        if len(body) >= MIN_CHUNK_CHARS:
            start, end = leftover[0], leftover[-1]
            chunks.append({
                "id": f"{rel_path}::<module>",
                "text": f"# File: {rel_path} | Symbol: <module>\n{body}",
                "source_type": "code",
                "path": rel_path,
                "symbol": "<module>",
                "url": github_url(repo, branch, rel_path, start, end),
                "start_line": start,
                "end_line": end,
            })
    return chunks


# ---------------------------------------------------------------- docs

def chunk_markdown_file(text: str, rel_path: str, repo: str,
                        branch: str) -> list[dict]:
    """Split on '## ' headings; each section keeps the page title as context."""
    lines = text.splitlines()
    page_title = next((l.lstrip("# ").strip() for l in lines
                       if l.startswith("# ")), rel_path)

    sections: list[tuple[str, int]] = []  # (heading, start_line 1-based)
    for i, line in enumerate(lines):
        if line.startswith("## "):
            sections.append((line.lstrip("# ").strip(), i + 1))
    if not sections or sections[0][1] > 1:
        sections.insert(0, (page_title, 1))

    chunks = []
    for idx, (heading, start) in enumerate(sections):
        end = sections[idx + 1][1] - 1 if idx + 1 < len(sections) else len(lines)
        body = "\n".join(lines[start - 1:end]).strip()
        if len(body) < MIN_CHUNK_CHARS:
            continue
        header = f"# Doc: {rel_path} | Page: {page_title} | Section: {heading}\n"
        chunks.append({
            "id": f"{rel_path}::{heading}::L{start}",
            "text": header + body,
            "source_type": "doc",
            "path": rel_path,
            "symbol": heading,
            "url": github_url(repo, branch, rel_path, start, end),
            "start_line": start,
            "end_line": end,
        })
    return chunks


# ---------------------------------------------------------------- issues

def chunk_issue(record: dict) -> dict | None:
    """One chunk per issue: title + body + top comments."""
    parts = [f"# Issue #{record['number']}: {record['title']}"]
    if record.get("labels"):
        parts.append(f"Labels: {', '.join(record['labels'])}")
    if record.get("body"):
        parts.append(record["body"][:MAX_ISSUE_BODY])
    for c in record.get("comments", []):
        parts.append(f"---\nComment: {c[:MAX_COMMENT]}")
    text = "\n".join(parts)
    if len(text) < MIN_CHUNK_CHARS:
        return None
    return {
        "id": f"issue::{record['number']}",
        "text": text,
        "source_type": "issue",
        "path": f"issues/{record['number']}.json",
        "symbol": record["title"],
        "url": record["url"],
        "start_line": None,
        "end_line": None,
    }


# ---------------------------------------------------------------- main

def main() -> None:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    repo = manifest["repo"]
    branch = default_branch()

    all_chunks: list[dict] = []

    for rec in manifest["files"]:
        path = REPO_DIR / rec["path"]
        if rec["source_type"] == "code":
            all_chunks += chunk_code_file(path.read_bytes(), rec["path"],
                                          repo, branch)
        elif rec["source_type"] == "doc":
            all_chunks += chunk_markdown_file(
                path.read_text(errors="replace"), rec["path"], repo, branch)

    for issue_file in sorted((DATA_DIR / "issues").glob("*.json")):
        chunk = chunk_issue(json.loads(issue_file.read_text()))
        if chunk:
            all_chunks.append(chunk)

    out = DATA_DIR / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    sizes = []
    for c in all_chunks:
        counts[c["source_type"]] = counts.get(c["source_type"], 0) + 1
        sizes.append(len(c["text"]))
    print(f"[done] {len(all_chunks)} chunks -> {out}")
    print(f"[done] by type: {counts}")
    print(f"[done] chars/chunk: min={min(sizes)} "
          f"avg={sum(sizes)//len(sizes)} max={max(sizes)}")


if __name__ == "__main__":
    main()
