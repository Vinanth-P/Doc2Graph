"""Tests for GraphBuilder and resolved entity/edge transformations."""
import pytest
from src.nlp.ner import Entity
from src.nlp.predicate_normalizer import NormalizedTriple
from src.graph.graph_builder import (
    build_resolved_entities,
    build_resolved_edges,
    ResolvedEntity,
    ResolvedEdge,
    GraphBuilder,
)
from src.type_map import TYPE_TO_LABEL


def test_type_to_label_mapping():
    assert TYPE_TO_LABEL["PERSON"] == "Person"
    assert TYPE_TO_LABEL["ORG"] == "Org"
    assert TYPE_TO_LABEL["GPE"] == "Gpe"
    assert TYPE_TO_LABEL["PRODUCT"] == "Product"
    assert TYPE_TO_LABEL["EVENT"] == "Event"
    assert TYPE_TO_LABEL["LAW"] == "Law"


def test_build_resolved_entities():
    e1 = Entity(
        entity_id="c:0:0:5", text="Tesla", canonical_name="Tesla",
        type="ORG", start_char=0, end_char=5, source="spacy", chunk_id="c:0"
    )
    e2 = Entity(
        entity_id="c:1:0:12", text="Tesla Motors", canonical_name="Tesla Motors",
        type="ORG", start_char=0, end_char=12, source="spacy", chunk_id="c:1"
    )

    # e1 and e2 resolved together to e1
    id_map = {e1.entity_id: e1.entity_id, e2.entity_id: e1.entity_id}

    resolved, entity_to_chunks = build_resolved_entities([e1, e2], id_map)

    assert len(resolved) == 1
    res = resolved[0]
    assert res.canonical_id == e1.entity_id
    assert res.type == "ORG"
    assert "Tesla" in res.aliases
    assert "Tesla Motors" in res.aliases
    assert set(entity_to_chunks[e1.entity_id]) == {"c:0", "c:1"}


def test_build_resolved_edges():
    e1 = Entity(
        entity_id="c:0:0:9", text="Elon Musk", canonical_name="Elon Musk",
        type="PERSON", start_char=0, end_char=9, source="spacy", chunk_id="c:0"
    )
    e2 = Entity(
        entity_id="c:0:18:23", text="Tesla", canonical_name="Tesla",
        type="ORG", start_char=18, end_char=23, source="spacy", chunk_id="c:0"
    )

    id_map = {e1.entity_id: "person_musk", e2.entity_id: "org_tesla"}

    triple = NormalizedTriple(
        relation_type="FOUNDED",
        subject=e1,
        object=e2,
        raw_predicate="found",
        chunk_id="c:0",
        sent_idx=0,
        confidence=0.95,
    )

    edges = build_resolved_edges([triple], id_map)
    assert len(edges) == 1
    edge = edges[0]
    assert edge.relation_type == "FOUNDED"
    assert edge.source_id == "person_musk"
    assert edge.target_id == "org_tesla"
    assert edge.chunk_ids == ["c:0"]
    assert edge.confidence == 0.95
