"""PDF text cleaner — remove headers/footers, de-hyphenate, strip tables."""
from __future__ import annotations

import re


# ── Header / footer removal ───────────────────────────────────────────────────

def remove_headers_footers(text: str, min_freq: int = 3) -> str:
    """Remove lines that repeat across pages (likely headers/footers).

    Strategy: split into pages on form-feed \\x0c, collect the first and last
    line of each page, count frequencies, and drop lines that appear ≥ min_freq.
    """
    pages = text.split("\x0c")
    if len(pages) < 2:
        return text

    candidate_lines: dict[str, int] = {}
    for page in pages:
        lines = page.strip().splitlines()
        if not lines:
            continue
        for line in [lines[0], lines[-1]]:
            stripped = line.strip()
            if stripped:
                candidate_lines[stripped] = candidate_lines.get(stripped, 0) + 1

    boilerplate = {ln for ln, cnt in candidate_lines.items() if cnt >= min_freq}

    cleaned_pages = []
    for page in pages:
        lines = [
            ln for ln in page.splitlines()
            if ln.strip() not in boilerplate
        ]
        cleaned_pages.append("\n".join(lines))
    return "\x0c".join(cleaned_pages)


# ── De-hyphenation ────────────────────────────────────────────────────────────

_HYPHEN_EOL = re.compile(r"(\w)-\n(\w)")


def dehyphenate(text: str) -> str:
    """Join words split across lines with a hyphen (e.g. 'organ-\\nization')."""
    return _HYPHEN_EOL.sub(r"\1\2", text)


# ── Table / figure stripping ──────────────────────────────────────────────────

_TABLE_ROW = re.compile(r"^(\s*\|.+\||\s*[-+]+)\s*$", re.MULTILINE)
_MANY_SPACES = re.compile(r"  {3,}")  # 3+ consecutive spaces often indicate tables


def strip_tables(text: str) -> str:
    """Remove obvious table rows (pipe-delimited or separator lines)."""
    text = _TABLE_ROW.sub("", text)
    # Collapse lines that are mostly whitespace/numbers (simple heuristic)
    lines = []
    for line in text.splitlines():
        # Drop lines that are >60% digits/spaces (numeric table rows)
        non_alpha = sum(1 for c in line if not c.isalpha())
        if line.strip() and non_alpha / max(len(line), 1) > 0.6:
            continue
        lines.append(line)
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def clean_pdf_text(text: str) -> str:
    """Apply all cleaning steps in sequence."""
    text = remove_headers_footers(text)
    text = dehyphenate(text)
    text = strip_tables(text)
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text
