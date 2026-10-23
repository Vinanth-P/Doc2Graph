"""Tests for span_merger — verifies the v4 bug fix (candidates built inside loop)."""
import pytest
from src.nlp.span_merger import merge_spans
from src.nlp.ner import Entity


def _ent(start, end, label, source, chunk_id="chunk:0"):
    return Entity(
        entity_id=f"{chunk_id}:{start}:{end}",
        text=f"text_{start}",
        canonical_name=f"Text {start}",
        type=label,
        start_char=start,
        end_char=end,
        source=source,
        chunk_id=chunk_id,
    )


def test_merge_returns_non_overlapping():
    spans = {
        "regex":     [_ent(0, 10, "DATE", "regex")],
        "gazetteer": [_ent(5, 15, "ORG", "gazetteer")],  # overlaps with regex
        "spacy":     [_ent(20, 30, "PERSON", "spacy")],
    }
    result = merge_spans(spans)
    starts = [e.start_char for e in result]
    ends   = [e.end_char   for e in result]
    # No overlapping pairs
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            assert ends[i] <= starts[j] or ends[j] <= starts[i]


def test_regex_takes_priority_over_spacy():
    spans = {
        "regex": [_ent(0, 10, "DATE", "regex")],
        "spacy": [_ent(0, 10, "ORG",  "spacy")],  # same span, lower priority
    }
    result = merge_spans(spans)
    assert len(result) == 1
    assert result[0].source == "regex"


def test_longer_span_wins_same_priority():
    spans = {
        "spacy": [
            _ent(0, 5, "ORG", "spacy"),
            _ent(0, 12, "ORG", "spacy"),  # longer, should win
        ],
    }
    result = merge_spans(spans)
    assert len(result) == 1
    assert result[0].end_char == 12


def test_candidates_populated_correctly():
    """Regression: merge_spans returned empty list when candidates not built inside loop."""
    spans = {
        "spacy": [_ent(0, 5, "PERSON", "spacy"), _ent(10, 20, "ORG", "spacy")],
    }
    result = merge_spans(spans)
    assert len(result) == 2   # would be 0 with the v4 bug
