"""Tests for rule-based coreference resolution."""
import pytest
from src.ingestion.preprocessor import SentenceChunk
from src.nlp.coref_resolver import RuleBasedCorefResolver
from src.nlp.ner import Entity


def test_coref_pronoun_person():
    chunks = [
        SentenceChunk(
            chunk_id="c:0", doc_id="d1", sent_idx=0,
            text="Elon Musk visited the factory.", start_char=0, end_char=30
        ),
        SentenceChunk(
            chunk_id="c:1", doc_id="d1", sent_idx=1,
            text="He praised the engineering team.", start_char=31, end_char=63
        ),
    ]
    musk_ent = Entity(
        entity_id="c:0:0:9", text="Elon Musk", canonical_name="Elon Musk",
        type="PERSON", start_char=0, end_char=9, source="spacy", chunk_id="c:0"
    )
    entities_by_chunk = {
        "c:0": [musk_ent],
        "c:1": [],
    }

    resolver = RuleBasedCorefResolver()
    res = resolver.resolve(chunks, entities_by_chunk)

    assert "c:1:he" in res
    assert res["c:1:he"] == "c:0:0:9"


def test_coref_pronoun_org():
    chunks = [
        SentenceChunk(
            chunk_id="c:0", doc_id="d1", sent_idx=0,
            text="Tesla opened a new gigafactory.", start_char=0, end_char=31
        ),
        SentenceChunk(
            chunk_id="c:1", doc_id="d1", sent_idx=1,
            text="It increased production significantly.", start_char=32, end_char=70
        ),
    ]
    tesla_ent = Entity(
        entity_id="c:0:0:5", text="Tesla", canonical_name="Tesla",
        type="ORG", start_char=0, end_char=5, source="spacy", chunk_id="c:0"
    )
    entities_by_chunk = {
        "c:0": [tesla_ent],
        "c:1": [],
    }

    resolver = RuleBasedCorefResolver()
    res = resolver.resolve(chunks, entities_by_chunk)

    assert "c:1:it" in res
    assert res["c:1:it"] == "c:0:0:5"


def test_coref_definite_np():
    chunks = [
        SentenceChunk(
            chunk_id="c:0", doc_id="d1", sent_idx=0,
            text="Apple reported quarterly earnings.", start_char=0, end_char=34
        ),
        SentenceChunk(
            chunk_id="c:1", doc_id="d1", sent_idx=1,
            text="The company exceeded Wall Street expectations.", start_char=35, end_char=81
        ),
    ]
    apple_ent = Entity(
        entity_id="c:0:0:5", text="Apple", canonical_name="Apple",
        type="ORG", start_char=0, end_char=5, source="spacy", chunk_id="c:0"
    )
    entities_by_chunk = {
        "c:0": [apple_ent],
        "c:1": [],
    }

    resolver = RuleBasedCorefResolver()
    res = resolver.resolve(chunks, entities_by_chunk)

    assert "c:1:the company" in res
    assert res["c:1:the company"] == "c:0:0:5"


def test_coref_definite_institute():
    chunks = [
        SentenceChunk("c:0", "d1", 0, "AIRI opened a laboratory.", 0, 27),
        SentenceChunk("c:1", "d1", 1, "The institute hired researchers.", 28, 61),
    ]
    airi = Entity(
        "c:0:0:4", "AIRI", "AIRI", "ORG", 0, 4, "spacy", "c:0"
    )
    result = RuleBasedCorefResolver().resolve(
        chunks, {"c:0": [airi], "c:1": []}
    )
    assert result["c:1:the institute"] == airi.entity_id


def test_coref_alias():
    chunks = [
        SentenceChunk(
            chunk_id="c:0", doc_id="d1", sent_idx=0,
            text="Elon Musk visited the facility.",
            start_char=0, end_char=31
        ),
        SentenceChunk(
            chunk_id="c:1", doc_id="d1", sent_idx=1,
            text="Musk inspected the assembly line.",
            start_char=32, end_char=65
        ),
    ]
    full_ent = Entity(
        entity_id="c:0:0:9", text="Elon Musk",
        canonical_name="Elon Musk",
        type="PERSON", start_char=0, end_char=9, source="spacy", chunk_id="c:0"
    )
    alias_ent = Entity(
        entity_id="c:1:0:4", text="Musk", canonical_name="Musk",
        type="PERSON", start_char=0, end_char=4, source="spacy", chunk_id="c:1"
    )
    entities_by_chunk = {
        "c:0": [full_ent],
        "c:1": [alias_ent],
    }

    resolver = RuleBasedCorefResolver()
    res = resolver.resolve(chunks, entities_by_chunk)

    # Shorter alias mapped to full entity
    assert res.get("c:1:0:4") == "c:0:0:9"
