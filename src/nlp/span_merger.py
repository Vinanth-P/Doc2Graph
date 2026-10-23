"""Span merger — greedy non-overlapping selection with priority + length tiebreak.

Priority: regex(0) > gazetteer(1) > spaCy model(2).
Within the same priority, the longest span wins.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class _Scored:
    start: int
    end: int
    label: str
    priority: int
    length: int
    original: Any   # the original Entity object


def merge_spans(spans_by_source: dict[str, list]) -> list:
    """Select a non-overlapping set of spans from multiple sources.

    Args:
        spans_by_source: OrderedDict-like mapping of source name → list of
            Entity objects (must have .start_char and .end_char attributes).
            Dict insertion order determines priority (first = highest).

    Returns:
        A list of Entity objects with no overlapping spans.
    """
    candidates: list[_Scored] = []

    for priority, (source, spans) in enumerate(spans_by_source.items()):
        for span in spans:
            candidates.append(_Scored(
                start=span.start_char,
                end=span.end_char,
                label=getattr(span, "type", ""),
                priority=priority,
                length=span.end_char - span.start_char,
                original=span,
            ))

    # Sort: lower priority number first, then longer spans first on ties
    candidates.sort(key=lambda s: (s.priority, -s.length))

    selected: list[_Scored] = []
    for cand in candidates:
        # Check for overlap with already-selected spans
        overlap = any(
            cand.start < sel.end and cand.end > sel.start
            for sel in selected
        )
        if not overlap:
            selected.append(cand)

    return [s.original for s in selected]
