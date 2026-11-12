"""Contract, mapping, and boundary regression tests for Classical News backend."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import get_settings
from src.graph.graph_builder import ResolvedEdge, ResolvedEntity
from src.nlp.classical_news import (
    ClassicalNewsExtractor,
    RELATION_MAP,
    TYPE_MAP,
    VALID_TYPE_PAIRS,
)
from src.nlp.contracts_v2 import MentionRecord, RelationClaimRecord
from src.nlp.stanford_classical_client import (
    StanfordClassicalClient,
    build_utf16_to_codepoint_map,
    convert_utf16_span,
)
from src.pipeline import run_pipeline_nlp


def test_utf16_codepoint_offset_mapper_with_emoji():
    """Verify that surrogate pairs (emoji > 0xFFFF) do not desynchronize Python character slices."""
    text = "Angela Merkel visited Berlin 🇩🇪 today."
    u16_map = build_utf16_to_codepoint_map(text)

    prefix = "Angela Merkel visited Berlin 🇩🇪 "
    # Calculate exact Java UTF-16 code units for prefix
    u16_start = len(prefix.encode("utf-16-le")) // 2
    u16_end = u16_start + len("today.".encode("utf-16-le")) // 2

    mapped_start, mapped_end = convert_utf16_span(u16_map, u16_start, u16_end)
    assert text[mapped_start:mapped_end] == "today."


def test_four_native_labels_and_nr_mapping():
    """Verify exact ontology mapping from Stanford Roth native labels to graph relations."""
    assert RELATION_MAP["Work_For"] == "WORKS_FOR"
    assert RELATION_MAP["Live_In"] == "LIVES_IN"
    assert RELATION_MAP["OrgBased_In"] == "BASED_IN"
    assert RELATION_MAP["Located_In"] == "LOCATED_IN"
    assert "_NR" not in RELATION_MAP


def test_direction_and_type_enforcement():
    """Verify argument type checking: normalizes flipped order, rejects invalid types and self-links."""
    mock_client = MagicMock(spec=StanfordClassicalClient)
    mock_client.is_available.return_value = True

    # Case 1: Flipped native arguments (Org, Person) for Work_For -> should flip to (Person, Org)
    # Case 2: Incompatible types (Location, Location) for Work_For -> should be rejected!
    mock_annotation = {
        "sentences": [{"index": 0, "text": "Test sentence.", "char_start": 0, "char_end": 14}],
        "entities": [
            {
                "mention_id": "ent1",
                "type": "ORGANIZATION",
                "text": "Microsoft",
                "char_start": 0,
                "char_end": 9,
                "mapped_text": "Microsoft",
                "sentence_index": 0,
            },
            {
                "mention_id": "ent2",
                "type": "PEOPLE",
                "text": "Satya Nadella",
                "char_start": 10,
                "char_end": 23,
                "mapped_text": "Satya Nadella",
                "sentence_index": 0,
            },
            {
                "mention_id": "ent3",
                "type": "LOCATION",
                "text": "Seattle",
                "char_start": 24,
                "char_end": 31,
                "mapped_text": "Seattle",
                "sentence_index": 0,
            },
        ],
        "relations": [
            {
                "relation_id": "rel1",
                "label": "Work_For",
                "arg0": {"object_id": "ent1", "type": "ORGANIZATION", "char_start": 0, "char_end": 9, "mapped_text": "Microsoft"},
                "arg1": {"object_id": "ent2", "type": "PEOPLE", "char_start": 10, "char_end": 23, "mapped_text": "Satya Nadella"},
                "probabilities": {"Work_For": 0.85},
                "sentence_index": 0,
            },
            {
                "relation_id": "rel2",
                "label": "Work_For",  # Invalid types: Org and Location cannot Work_For
                "arg0": {"object_id": "ent1", "type": "ORGANIZATION", "char_start": 0, "char_end": 9, "mapped_text": "Microsoft"},
                "arg1": {"object_id": "ent3", "type": "LOCATION", "char_start": 24, "char_end": 31, "mapped_text": "Seattle"},
                "probabilities": {"Work_For": 0.90},
                "sentence_index": 0,
            },
        ],
    }
    mock_client.annotate.return_value = mock_annotation

    extractor = ClassicalNewsExtractor(client=mock_client)
    res = extractor.extract_document("doc1", "Microsoft Satya Nadella Seattle")

    # Only 1 claim should survive: the valid Work_For with normalized direction (Nadella -> Microsoft)
    assert len(res["claims"]) == 1
    claim = res["claims"][0]
    assert claim.relation == "WORKS_FOR"
    assert "ent2" in claim.subject_mention_id  # Person is subject!
    assert "ent1" in claim.object_mention_id   # Org is object!
    assert claim.polarity == "unknown"
    assert claim.calibrated_confidence is None
    assert claim.raw_score == 0.85


def test_honest_confidence_and_unknown_assertions():
    """Verify that classical news claims honestly report raw_score, leaving calibrated_confidence None,
    and assert unknown polarity/modality/lifecycle."""
    mock_client = MagicMock(spec=StanfordClassicalClient)
    mock_client.is_available.return_value = True
    mock_client.annotate.return_value = {
        "sentences": [{"index": 0, "text": "Barack Obama lived in Hawaii.", "char_start": 0, "char_end": 29}],
        "entities": [
            {"mention_id": "e1", "type": "PEOPLE", "text": "Barack Obama", "char_start": 0, "char_end": 12, "mapped_text": "Barack Obama", "sentence_index": 0},
            {"mention_id": "e2", "type": "LOCATION", "text": "Hawaii", "char_start": 22, "char_end": 28, "mapped_text": "Hawaii", "sentence_index": 0},
        ],
        "relations": [
            {
                "relation_id": "r1",
                "label": "Live_In",
                "arg0": {"object_id": "e1", "type": "PEOPLE", "char_start": 0, "char_end": 12, "mapped_text": "Barack Obama"},
                "arg1": {"object_id": "e2", "type": "LOCATION", "char_start": 22, "char_end": 28, "mapped_text": "Hawaii"},
                "probabilities": {"Live_In": 0.78, "_NR": 0.12},
                "sentence_index": 0,
            }
        ],
    }

    extractor = ClassicalNewsExtractor(client=mock_client)
    res = extractor.extract_document("doc_honesty", "Barack Obama lived in Hawaii.")

    assert len(res["claims"]) == 1
    claim = res["claims"][0]
    assert claim.relation == "LIVES_IN"
    assert claim.raw_score == 0.78
    assert claim.calibrated_confidence is None
    assert claim.polarity == "unknown"
    assert claim.modality == "unknown"
    assert claim.lifecycle == "unknown"

    assert len(res["resolved_edges"]) == 1
    edge = res["resolved_edges"][0]
    assert edge.relation_type == "LIVES_IN"
    assert edge.polarity == "unknown"
    assert edge.confidence == 0.78


def test_missing_runtime_fails_closed():
    """Verify that when Java runtime or CoreNLP models are missing, client fails closed."""
    fake_client = StanfordClassicalClient(java_bin=Path("C:/nonexistent/java.exe"))
    assert fake_client.is_available() is False

    extractor = ClassicalNewsExtractor(client=fake_client)
    with pytest.raises(RuntimeError) as exc_info:
        fake_client.start()
    assert "not found" in str(exc_info.value).lower()


def test_non_neural_pipeline_audit():
    """Verify that StanfordClassicalBridge java code strictly pins non-neural models and omits neural annotators."""
    bridge_src = Path("java/stanford-classical-bridge/src/StanfordClassicalBridge.java").read_text(encoding="utf-8")

    # Pinned non-neural models
    assert "english.all.3class.distsim.crf.ser.gz" in bridge_src
    assert "englishPCFG.ser.gz" in bridge_src
    assert "roth_relation_model_pipelineNER.ser" in bridge_src
    assert "english-left3words-distsim.tagger" in bridge_src

    # Forbidden neural annotators
    assert "depparse" not in bridge_src
    assert "coref" not in bridge_src
    assert "stanza" not in bridge_src.lower()
    assert "transformer" not in bridge_src.lower()


def test_pipeline_nlp_classical_news():
    """Test running run_pipeline_nlp with relation_backend='classical_news'."""
    settings = get_settings()
    orig_backend = settings.relation_backend
    settings.relation_backend = "classical_news"
    try:
        sample_text = (
            "Steve Jobs lived in Palo Alto. "
            "Apple is headquartered in Cupertino, California. "
            "Tim Cook works for Apple."
        )
        res = run_pipeline_nlp(sample_text.encode("utf-8"), "test_news.txt", "txt")

        assert "resolved_entities" in res
        assert "normalized" in res
        assert "v2_claims" in res
        assert "v2_canonical_claims" in res
        assert "resolved_edges" in res

        # Check entity names
        ent_names = {e.name for e in res["resolved_entities"]}
        assert "Jobs" in ent_names or "Steve Jobs" in ent_names or "Apple" in ent_names

        # Check that edges exist and have valid canonical types
        for edge in res["resolved_edges"]:
            assert edge.relation_type in {"WORKS_FOR", "LIVES_IN", "BASED_IN", "LOCATED_IN"}
            assert edge.polarity == "unknown"
            assert edge.modality == "unknown"
    finally:
        settings.relation_backend = orig_backend
