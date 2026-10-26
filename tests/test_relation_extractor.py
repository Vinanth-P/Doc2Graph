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


def test_reduced_clause_keeps_partner_predicate(nlp):
    extractor = DepPatternExtractor()
    text = "TechNova Solutions partnered with the Indian Institute of Science."
    doc = nlp(text)
    technova = _make_entity("TechNova Solutions", 0, 20, "ORG")
    iisc = _make_entity("the Indian Institute of Science", 38, 69, "ORG")

    triples = extractor.extract(
        doc=doc,
        entities=[technova, iisc],
        center_chunk_id="c:0",
        center_sent_idx=0,
    )

    partner = [t for t in triples if t.prep == "with"]
    assert partner
    assert partner[0].predicate == "partner"
    assert partner[0].subject.text == "TechNova Solutions"


def test_location_modifier_uses_nearest_organization(nlp):
    extractor = DepPatternExtractor()
    text = "TechNova Solutions, a company headquartered in Mumbai."
    doc = nlp(text)
    technova = _make_entity("TechNova Solutions", 0, 20, "ORG")
    mumbai = _make_entity("Mumbai", 48, 54, "GPE")

    triples = extractor.extract(
        doc=doc,
        entities=[technova, mumbai],
        center_chunk_id="c:0",
        center_sent_idx=0,
    )

    location = [t for t in triples if t.object.text == "Mumbai"]
    assert location
    assert location[0].subject.text == "TechNova Solutions"


def test_supported_verb_can_extract_noun_phrase_product(nlp):
    extractor = DepPatternExtractor()
    text = "IISc provided the computing infrastructure."
    doc = nlp(text)
    iisc = _make_entity("IISc", 0, 4, "ORG")

    triples = extractor.extract(
        doc=doc,
        entities=[iisc],
        center_chunk_id="c:0",
        center_sent_idx=0,
    )

    provided = [t for t in triples if t.predicate == "provide"]
    assert provided
    assert provided[0].object.type == "PRODUCT"
    assert provided[0].object.text == "the computing infrastructure"
