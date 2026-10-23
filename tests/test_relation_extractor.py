"""Tests for dependency-pattern relation extraction."""
import pytest
from src.nlp.relation_extractor import DepPatternExtractor
from src.nlp.ner import Entity


def _make_entity(text, start, end, etype="ORG", chunk_id="c:0"):
    return Entity(
        entity_id=f"{chunk_id}:{start}:{end}",
        text=text,
        canonical_name=text,
        type=etype,
        start_char=start,
        end_char=end,
        source="test",
        chunk_id=chunk_id,
    )


def test_svo_pattern(nlp):
    extractor = DepPatternExtractor()
    text = "Elon Musk founded Tesla."
    doc = nlp(text)

    musk = _make_entity("Elon Musk", 0, 9, "PERSON")
    tesla = _make_entity("Tesla", 18, 23, "ORG")

    triples = extractor.extract(
        doc=doc,
        entities=[musk, tesla],
        center_chunk_id="c:0",
        center_sent_idx=0,
    )

    svo_triples = [t for t in triples if t.source == "svo"]
    assert len(svo_triples) >= 1
    t = svo_triples[0]
    assert t.subject.canonical_name == "Elon Musk"
    assert t.predicate == "found"
    assert t.object.canonical_name == "Tesla"


def test_appositive_pattern(nlp):
    extractor = DepPatternExtractor()
    text = "Tim Cook, CEO of Apple, announced the new iPhone."
    doc = nlp(text)

    cook = _make_entity("Tim Cook", 0, 8, "PERSON")
    apple = _make_entity("Apple", 17, 22, "ORG")

    triples = extractor.extract(
        doc=doc,
        entities=[cook, apple],
        center_chunk_id="c:0",
        center_sent_idx=0,
    )

    appos_triples = [t for t in triples if t.source == "appos"]
    assert len(appos_triples) >= 1
    t = appos_triples[0]
    assert t.subject.canonical_name == "Tim Cook"
    assert t.predicate == "appos"


def test_nominal_pattern(nlp):
    extractor = DepPatternExtractor()
    text = "Tesla founder Elon Musk presented the master plan."
    doc = nlp(text)

    tesla = _make_entity("Tesla", 0, 5, "ORG")
    musk = _make_entity("Elon Musk", 14, 23, "PERSON")

    triples = extractor.extract(
        doc=doc,
        entities=[tesla, musk],
        center_chunk_id="c:0",
        center_sent_idx=0,
    )

    nom_triples = [t for t in triples if t.source == "nominal"]
    assert len(nom_triples) >= 1
    assert nom_triples[0].subject.canonical_name == "Elon Musk"
    assert nom_triples[0].object.canonical_name == "Tesla"


def test_coref_substitution(nlp):
    extractor = DepPatternExtractor()
    text = "He leads SpaceX."
    doc = nlp(text)

    musk = _make_entity("Elon Musk", 0, 9, "PERSON", chunk_id="c:prev")
    spacex = _make_entity("SpaceX", 9, 15, "ORG", chunk_id="c:0")

    coref_map = {"c:0:he": musk.entity_id}
    entity_by_id = {musk.entity_id: musk, spacex.entity_id: spacex}

    triples = extractor.extract(
        doc=doc,
        entities=[spacex],
        center_chunk_id="c:0",
        center_sent_idx=1,
        coref_map=coref_map,
        entity_by_id=entity_by_id,
    )

    assert len(triples) >= 1
    assert triples[0].subject.canonical_name == "Elon Musk"
    assert triples[0].object.canonical_name == "SpaceX"

