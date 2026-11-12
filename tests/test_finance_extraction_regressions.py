"""Finance regressions use unseen names and negative controls, not just edge counts."""
from dataclasses import replace
from pathlib import Path

import pytest

from src.ingestion.pdf_cleaner import clean_pdf_text
from src.ingestion.preprocessor import SentenceChunk
from src.nlp.document_entities import recover_document_entities
from src.nlp.ner import Entity, NERRunner
from src.nlp.relation_extractor import DepPatternExtractor
from src.nlp.predicate_normalizer import PredicateNormalizer
from src.graph.entity_resolver import EntityResolver
from src.graph.graph_builder import GraphBuilder, build_resolved_edges


def extract(nlp, text, pairs):
    entities = []
    for name, kind in pairs:
        start = text.index(name)
        entities.append(Entity(f"d:0:{start}:{start + len(name)}", name, name, kind,
                               start, start + len(name), "test", "d:0", text))
    raw = DepPatternExtractor().extract(nlp(text), entities, "d:0", 0)
    return PredicateNormalizer(include_extended_relations=True).normalize_batch(raw)


@pytest.mark.parametrize("text,pairs,expected", [
    ("Atlas signed an agreement to acquire Beacon.", [("Atlas", "ORG"), ("Beacon", "ORG")],
     ("Atlas", "PLANS_TO_ACQUIRE", "Beacon")),
    ("Atlas acquired Beacon and hired Maya Chen.", [("Atlas", "ORG"), ("Beacon", "ORG"), ("Maya Chen", "PERSON")],
     ("Maya Chen", "EMPLOYED_BY", "Atlas")),
    ("Beacon was acquired by Atlas.", [("Atlas", "ORG"), ("Beacon", "ORG")],
     ("Atlas", "ACQUIRED", "Beacon")),
    ("Atlas owns 22% of Beacon.", [("Atlas", "ORG"), ("Beacon", "ORG")],
     ("Atlas", "OWNS", "Beacon")),
    ("Orion Capital acted as financial adviser to Beacon.", [("Orion Capital", "ORG"), ("Beacon", "ORG")],
     ("Orion Capital", "ADVISER_TO", "Beacon")),
    ("Maya Chen reports to Noah Patel.", [("Maya Chen", "PERSON"), ("Noah Patel", "PERSON")],
     ("Maya Chen", "REPORTS_TO", "Noah Patel")),
    ("Beacon operates as a subsidiary of Atlas.", [("Atlas", "ORG"), ("Beacon", "ORG")],
     ("Beacon", "SUBSIDIARY_OF", "Atlas")),
    ("Beacon will operate as a subsidiary of Atlas.", [("Atlas", "ORG"), ("Beacon", "ORG")],
     ("Beacon", "PLANNED_SUBSIDIARY_OF", "Atlas")),
    ("Atlas plans to integrate Beacon Suite.", [("Atlas", "ORG"), ("Beacon Suite", "PRODUCT")],
     ("Atlas", "PLANS_TO_USE", "Beacon Suite")),
    ("Atlas received a loan from Orion Bank.", [("Atlas", "ORG"), ("Orion Bank", "ORG")],
     ("Atlas", "FINANCED_BY", "Orion Bank")),
    ("Atlas expects to fund the purchase through a loan from Orion Bank.", [("Atlas", "ORG"), ("Orion Bank", "ORG")],
     ("Atlas", "PLANS_TO_BORROW_FROM", "Orion Bank")),
    ("Atlas's Chief Financial Officer, Maya Chen, spoke.", [("Atlas", "ORG"), ("Maya Chen", "PERSON")],
     ("Maya Chen", "EMPLOYED_BY", "Atlas")),
])
def test_finance_relation_direction(nlp, text, pairs, expected):
    triples = extract(nlp, text, pairs)
    facts = {(t.subject.text, t.relation_type, t.object.text) for t in triples}
    assert expected in facts
    assert (expected[2], expected[1], expected[0]) not in facts
    if expected[1] == "PLANS_TO_ACQUIRE":
        assert not any(t.relation_type == "ACQUIRED" for t in triples)
    if expected[1] == "PLANNED_SUBSIDIARY_OF":
        assert not any(t.relation_type == "SUBSIDIARY_OF" for t in triples)


@pytest.mark.parametrize("text", [
    "Atlas did not acquire Beacon.",
    "Atlas does not own 22% of Beacon.",
    "Atlas is not a subsidiary of Beacon.",
    "Atlas did not act as financial adviser to Beacon.",
    "Atlas does not plan to acquire Beacon.",
])
def test_finance_negation(nlp, text):
    assert extract(nlp, text, [("Atlas", "ORG"), ("Beacon", "ORG")]) == []


def test_shareholders_are_not_the_company(nlp):
    triples = extract(nlp, "Atlas's shareholders own 70% of Beacon.", [("Atlas", "ORG"), ("Beacon", "ORG")])
    assert not any(t.relation_type == "OWNS" for t in triples)


def test_report_noun_is_not_a_reporting_line(nlp):
    triples = extract(nlp, "Atlas provided a quarterly report to Beacon.", [("Atlas", "ORG"), ("Beacon", "ORG")])
    assert not any(t.relation_type == "REPORTS_TO" for t in triples)


def test_document_aliases_preserve_spans_and_types(nlp):
    texts = ['Atlas Systems Ltd. ("Atlas") appointed Chief Financial Officer Maya Chen.',
             'Atlas advised Beacon Analytics Ltd. ("Beacon").',
             'Chen visited Beacon.']
    chunks = [SentenceChunk(f"d:{i}", "d", i, t, 0, len(t)) for i, t in enumerate(texts)]
    ner = NERRunner()
    result = recover_document_entities(chunks, {c.chunk_id: ner.run(nlp(c.text), c.chunk_id, c.text) for c in chunks})
    last = {e.text: e for e in result["d:2"]}
    assert last["Chen"].canonical_name == "Maya Chen"
    assert last["Chen"].type == "PERSON"
    assert last["Beacon"].canonical_name == "Beacon Analytics Ltd"
    for c in chunks:
        assert all(c.text[e.start_char:e.end_char] == e.text for e in result[c.chunk_id])


def test_ambiguous_surname_is_not_recovered(nlp):
    text = "Chief Financial Officer Maya Chen met Chief Technology Officer Noah Chen."
    chunks = [SentenceChunk("d:0", "d", 0, text, 0, len(text)), SentenceChunk("d:1", "d", 1, "Chen spoke.", 0, 11)]
    result = recover_document_entities(chunks, {"d:0": [], "d:1": []})
    assert result["d:1"] == []


def test_coordinated_company_names_stay_separate():
    text = "Orion Capital and Vega Ventures invested in Atlas Ltd."
    c = SentenceChunk("d:0", "d", 0, text, 0, len(text))
    result = recover_document_entities([c], {"d:0": []})
    assert {e.canonical_name for e in result["d:0"]} == {"Orion Capital", "Vega Ventures", "Atlas Ltd"}


def test_pdf_layout_does_not_join_headers_or_table_labels():
    text = ("CONFIDENTIAL CASE STUDY\nFinance report\n1. Overview\nAtlas acquired\nBeacon.\nCORPUS | Page 1\f"
            "2. Financing\nAtlas borrowed from Orion Bank.\n| Acquirer | Atlas |\nCORPUS | Page 2")
    cleaned = clean_pdf_text(text)
    assert "Atlas acquired Beacon." in cleaned
    assert "CORPUS" not in cleaned
    assert "Overview" not in cleaned
    assert "Acquirer" not in cleaned


def test_common_noun_is_not_a_gazetteer_company(nlp):
    text = "The bank's one-year benchmark determines the rate."
    assert "benchmark" not in {e.text for e in NERRunner().run(nlp(text), "d:0", text)}


def test_alias_resolution_cannot_create_self_edges(nlp):
    triples = extract(nlp, "Atlas acquired Beacon.", [("Atlas", "ORG"), ("Beacon", "ORG")])
    assert build_resolved_edges(triples, {triples[0].subject.entity_id: "same", triples[0].object.entity_id: "same"}) == []


def test_identical_canonical_names_merge_without_model():
    ent = Entity("one", "Atlas", "Atlas Systems Ltd", "ORG", 0, 5, "test", "d:0")
    other = replace(ent, entity_id="two", text="Atlas Systems Ltd")
    resolver = EntityResolver()
    resolver._lr = None
    ids = resolver.resolve_within_doc([ent, other])
    assert ids["one"] == ids["two"]


def test_relation_schema_matches_configuration():
    from src.nlp.predicate_normalizer import load_relations_config
    assert set(load_relations_config()) <= set(GraphBuilder.CANONICAL_TYPES)


def test_original_finance_pdf_production_pipeline():
    from scripts.inspect_document import inspect_document
    path = Path("Doc2Graph_Corporate_Finance_MA_Dummy.pdf")
    if not path.exists():
        pytest.skip("User-provided PDF is not part of the repository")
    report = inspect_document(path)
    facts = {(r["source_name"], r["relation_type"], r["target_name"]) for r in report["relationships"]}
    assert ("Northstar Systems Ltd", "PLANS_TO_ACQUIRE", "Meridian Analytics Pvt. Ltd") in facts
    assert ("Axis Capital", "ADVISER_TO", "Northstar Systems Ltd") in facts
    assert ("Desai Rao Associates", "ADVISER_TO", "Meridian Analytics Pvt. Ltd") in facts
    assert ("Rohan Mehta", "REPORTS_TO", "Ananya Rao") in facts
    assert ("Cedar Peak Ventures LLP", "OWNS", "Meridian Analytics Pvt. Ltd") in facts
    assert ("Priya Nair", "EMPLOYED_BY", "Northstar Systems Ltd") in facts
    assert report["summary"]["connected_entity_fraction"] >= 0.80
    assert not any(s == o for s, _, o in facts)
    assert not any(r["relation_type"] == "ACQUIRED" for r in report["relationships"])
    assert all(r["evidence"] and r["chunk_ids"] for r in report["relationships"])
    names = {e["name"] for e in report["entities"]}
    assert not names & {"FICTIONAL", "CORPUS", "Benchmark", "PERSON", "ORGANIZATION"}
