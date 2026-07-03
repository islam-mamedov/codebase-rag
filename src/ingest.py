
 
import argparse
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
 
from github import Auth, Github
 
DATA_DIR = Path("data")
 
# Directories that add noise, not knowledge. Extend as you inspect results.
SKIP_DIRS = {".git", ".github", "__pycache__", "node_modules", "tests", "test",
             "scripts"}
CODE_EXTS = {".py"}
DOC_EXTS = {".md"}
MAX_FILE_BYTES = 400_000   # skip generated/vendored monsters
ISSUE_LOOKBACK_DAYS = 730  # ~2 years
 
 
def clone_repo(repo: str) -> Path:
    """Shallow-clone the repo into data/repo/ (we only need current state)."""
    dest = DATA_DIR / "repo"
    if dest.exists():
        print(f"[clone] {dest} already exists, skipping clone")
        return dest
    url = f"https://github.com/{repo}.git"
    print(f"[clone] {url} -> {dest}")
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
    return dest
 
 
def collect_files(repo_dir: Path) -> list[dict]:
    """Walk the clone and collect code + doc files worth indexing."""
    records = []
    for path in sorted(repo_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_dir)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        # FastAPI-specific: keep only English docs, skip tutorial snippets
        if rel.parts[0] == "docs" and rel.parts[1] != "en":
            continue
        if rel.parts[0] == "docs_src":
            continue
        ext = path.suffix.lower()
        if ext in CODE_EXTS:
            source_type = "code"
        elif ext in DOC_EXTS:
            source_type = "doc"
        else:
            continue
        size = path.stat().st_size
        if size == 0 or size > MAX_FILE_BYTES:
            continue
        records.append({
            "source_type": source_type,
            "path": str(rel),
            "size_bytes": size,
        })
    return records
 
 
def fetch_issues(repo: str, max_issues: int) -> list[dict]:
    """Pull recent closed issues (not PRs) and save each as JSON.
 
    Resumable: issues already saved to disk are skipped (no API calls),
    so a crashed or interrupted run can simply be re-run.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("[warn] no GITHUB_TOKEN set - unauthenticated limit is 60 req/hr")
    gh = Github(auth=Auth.Token(token)) if token else Github()
    since = datetime.now(timezone.utc) - timedelta(days=ISSUE_LOOKBACK_DAYS)
    issues_dir = DATA_DIR / "issues"
    issues_dir.mkdir(parents=True, exist_ok=True)
 
    records = []
    # sort="comments" surfaces the most-discussed (usually most useful) issues first
    issues = gh.get_repo(repo).get_issues(state="closed", since=since,
                                          sort="comments", direction="desc")
    for issue in issues:
        if issue.pull_request is not None:  # PRs come through the same API; skip
            continue
        if len(records) >= max_issues:
            break
 
        # Resume support: skip issues already saved from a previous run
        out_path = issues_dir / f"{issue.number}.json"
        if out_path.exists():
            records.append({"source_type": "issue",
                            "path": f"issues/{issue.number}.json",
                            "url": issue.html_url})
            continue
 
        # Keep up to 3 comments, favoring the most 👍-reacted (often the answer)
        comments = []
        if issue.comments > 0:
            all_comments = list(issue.get_comments())
            all_comments.sort(key=lambda c: c.reactions.get("+1", 0), reverse=True)
            comments = [c.body for c in all_comments[:3] if c.body]
 
        record = {
            "number": issue.number,
            "title": issue.title,
            "body": issue.body or "",
            "labels": [label.name for label in issue.labels],
            "url": issue.html_url,
            "comments": comments,
        }
        out_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append({"source_type": "issue",
                        "path": f"issues/{issue.number}.json",
                        "url": issue.html_url})
        if len(records) % 50 == 0:
            print(f"[issues] fetched {len(records)}...")
    return records
 
 
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="e.g. fastapi/fastapi")
    parser.add_argument("--max-issues", type=int, default=300)
    parser.add_argument("--skip-issues", action="store_true",
                        help="only clone + collect files (useful while iterating)")
    args = parser.parse_args()
 
    DATA_DIR.mkdir(exist_ok=True)
    repo_dir = clone_repo(args.repo)
    file_records = collect_files(repo_dir)
    issue_records = [] if args.skip_issues else fetch_issues(args.repo,
                                                             args.max_issues)
 
    manifest = {
        "repo": args.repo,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "files": file_records,
        "issues": issue_records,
    }
    (DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
 
    n_code = sum(1 for r in file_records if r["source_type"] == "code")
    n_docs = sum(1 for r in file_records if r["source_type"] == "doc")
    print(f"\n[done] {n_code} code files, {n_docs} docs, {len(issue_records)} issues")
    print(f"[done] manifest -> {DATA_DIR / 'manifest.json'}")
 
 
if __name__ == "__main__":
    main()
 
