"""Tests for the chunkers. Run with: pytest tests/ -v"""

import sys
from pathlib import Path

# Make src/ importable when running pytest from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chunk import chunk_code_file, chunk_issue, chunk_markdown_file

SAMPLE_CODE = b'''\
import os

CONSTANT_VALUE = "something important that should end up in the module chunk"

def greet(name: str) -> str:
    """Say hello to someone by name, politely and at length."""
    message = f"Hello, {name}! Welcome to the test suite."
    return message

@property
def decorated_function(self):
    """A decorated function to test decorated_definition unwrapping."""
    return self._value

class Greeter:
    """A small class that fits in one chunk."""

    def __init__(self, name: str) -> None:
        self.name = name

    def greet(self) -> str:
        return f"Hello, {self.name}!"
'''


def test_function_becomes_chunk():
    chunks = chunk_code_file(SAMPLE_CODE, "pkg/mod.py", "owner/repo", "main")
    symbols = [c["symbol"] for c in chunks]
    assert "greet" in symbols


def test_chunk_has_context_header():
    chunks = chunk_code_file(SAMPLE_CODE, "pkg/mod.py", "owner/repo", "main")
    greet = next(c for c in chunks if c["symbol"] == "greet")
    assert greet["text"].startswith("# File: pkg/mod.py")
    assert "def greet" in greet["text"]


def test_line_numbers_are_correct():
    chunks = chunk_code_file(SAMPLE_CODE, "pkg/mod.py", "owner/repo", "main")
    greet = next(c for c in chunks if c["symbol"] == "greet")
    lines = SAMPLE_CODE.decode().splitlines()
    assert lines[greet["start_line"] - 1].startswith("def greet")


def test_class_kept_whole_when_small():
    chunks = chunk_code_file(SAMPLE_CODE, "pkg/mod.py", "owner/repo", "main")
    assert any(c["symbol"] == "Greeter" for c in chunks)


def test_module_chunk_collects_constants():
    chunks = chunk_code_file(SAMPLE_CODE, "pkg/mod.py", "owner/repo", "main")
    module = next(c for c in chunks if c["symbol"] == "<module>")
    assert "CONSTANT_VALUE" in module["text"]


def test_no_empty_chunks():
    chunks = chunk_code_file(SAMPLE_CODE, "pkg/mod.py", "owner/repo", "main")
    assert all(c["text"].strip() for c in chunks)


SAMPLE_MD = """\
# Getting Started

Some introduction text that is long enough to keep as a chunk of its own.

## Installation

Run pip install and you are ready to go. This section explains the details
of installing the package in a virtual environment.

## Usage

Import the package and call the main function to get going with the API.
"""


def test_markdown_splits_on_h2():
    chunks = chunk_markdown_file(SAMPLE_MD, "docs/en/start.md", "o/r", "main")
    sections = [c["symbol"] for c in chunks]
    assert "Installation" in sections
    assert "Usage" in sections


def test_markdown_keeps_page_title_context():
    chunks = chunk_markdown_file(SAMPLE_MD, "docs/en/start.md", "o/r", "main")
    install = next(c for c in chunks if c["symbol"] == "Installation")
    assert "Getting Started" in install["text"]


def test_issue_chunk():
    record = {
        "number": 42,
        "title": "App crashes on startup",
        "body": "When I run the app it crashes immediately with a traceback.",
        "labels": ["bug"],
        "url": "https://github.com/o/r/issues/42",
        "comments": ["Fixed by upgrading the dependency to version 2.0."],
    }
    chunk = chunk_issue(record)
    assert chunk["id"] == "issue::42"
    assert "crashes" in chunk["text"]
    assert "Fixed by upgrading" in chunk["text"]
