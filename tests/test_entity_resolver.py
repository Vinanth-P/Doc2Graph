"""Tests for entity resolver — blocking, scoring, initials key."""
import pytest
from src.graph.entity_resolver import (
    compute_initials_key, _token_overlap_ratio,
    _abbreviation_score, build_candidate_pairs,
)
from src.nlp.ner import Entity


def _ent(name, etype="ORG"):
    return Entity(
        entity_id=name, text=name, canonical_name=name,
        type=etype, start_char=0, end_char=len(name),
        source="spacy", chunk_id="c:0",
    )


def test_initials_key():
    assert compute_initials_key("International Business Machines") == "IBM"
    assert compute_initials_key("Elon Musk") == "EM"
    assert compute_initials_key("Tesla") == "T"


def test_abbreviation_score_ibm():
    score = _abbreviation_score("IBM", "International Business Machines")
    assert score >= 0.8


def test_abbreviation_score_identical_initials():
    score = _abbreviation_score("AT&T", "AT&T")
    assert score >= 0.8


def test_token_overlap_ratio():
    assert _token_overlap_ratio("Tesla Motors", "Tesla") > 0.0
    assert _token_overlap_ratio("Apple", "Google") == 0.0


def test_blocking_finds_abbreviation_pair():
    entities = [
        _ent("IBM"),
        _ent("International Business Machines"),
        _ent("Apple"),
    ]
    pairs = build_candidate_pairs(entities)
    pair_names = {(a.canonical_name, b.canonical_name) for a, b in pairs}
    assert ("IBM", "International Business Machines") in pair_names or \
           ("International Business Machines", "IBM") in pair_names


def test_blocking_bucket_cap():
    """Bucket larger than cap should be skipped."""
    # All same type, all share token "Co" — would form one huge bucket
    entities = [_ent(f"Company {i} Co") for i in range(200)]
    pairs = build_candidate_pairs(entities, bucket_cap=100)
    # Should not explode: 200*199/2 = 19900 pairs; with cap they're skipped
    assert len(pairs) < 10000


def test_same_type_guard():
    """Cross-type pairs must not be returned."""
    entities = [_ent("Tesla", "ORG"), _ent("Tesla", "PERSON")]
    pairs = build_candidate_pairs(entities)
    assert len(pairs) == 0   # different types
