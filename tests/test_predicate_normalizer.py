"""Tests for predicate normalizer — canonical mapping, swap, disambiguation."""
import pytest
from src.nlp.ner import Entity
from src.nlp.relation_extractor import RawTriple
from src.nlp.predicate_normalizer import PredicateNormalizer


def _ent(etype, text="X", chunk_id="c:0"):
    return Entity(
        entity_id=f"{chunk_id}:0:1",
        text=text, canonical_name=text, type=etype,
        start_char=0, end_char=1, source="spacy", chunk_id=chunk_id,
    )


def _triple(pred, subj_type, obj_type, prep=None, source="svo"):
    return RawTriple(
        subject=_ent(subj_type, "Alice"),
        predicate=pred,
        object=_ent(obj_type, "Acme"),
        prep=prep,
        source=source,
        sent_idx=0,
        chunk_id="c:0",
    )


@pytest.fixture(scope="module")
def normalizer():
    return PredicateNormalizer("config/relations.yaml")


def test_join_maps_to_member_of(normalizer):
    t = _triple("join", "PERSON", "ORG")
    r = normalizer.normalize(t)
    assert r.relation_type == "MEMBER_OF"
    assert r.subject.type == "PERSON"


def test_hire_maps_to_employed_by_and_swaps(normalizer):
    """'Google hired Alice' → Alice EMPLOYED_BY Google (swap)."""
    google = _ent("ORG", "Google")
    alice  = _ent("PERSON", "Alice")
    t = RawTriple(
        subject=google, predicate="hire", object=alice,
        prep=None, source="svo", sent_idx=0, chunk_id="c:0",
    )
    r = normalizer.normalize(t)
    assert r.relation_type == "EMPLOYED_BY"
    assert r.subject.type == "PERSON"   # after swap
    assert r.object.type  == "ORG"


def test_work_without_prep_falls_to_related_to(normalizer):
    """'works on a project' must NOT become MEMBER_OF (no qualifying prep)."""
    t = _triple("work", "PERSON", "ORG", prep=None)
    r = normalizer.normalize(t)
    assert r.relation_type == "RELATED_TO"


def test_work_with_correct_prep_maps_member_of(normalizer):
    t = _triple("work", "PERSON", "ORG", prep="at")
    r = normalizer.normalize(t)
    assert r.relation_type == "MEMBER_OF"


def test_found_with_org_object_maps_founded(normalizer):
    t = _triple("found", "PERSON", "ORG")
    r = normalizer.normalize(t)
    assert r.relation_type == "FOUNDED"


def test_found_with_non_org_object_falls_to_related_to(normalizer):
    """'find' past tense disambiguation: dobj not ORG → reject FOUNDED."""
    t = _triple("found", "PERSON", "PRODUCT")  # not an ORG
    r = normalizer.normalize(t)
    assert r.relation_type == "RELATED_TO"


def test_unknown_predicate_falls_to_related_to(normalizer):
    t = _triple("blorp", "PERSON", "ORG")
    r = normalizer.normalize(t)
    assert r.relation_type == "RELATED_TO"
    assert r.raw_predicate == "blorp"
