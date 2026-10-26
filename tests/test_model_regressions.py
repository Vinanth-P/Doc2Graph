"""Semantic regressions: exact relations, evidence, polarity, and persistence IDs."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.ingestion.preprocessor import SentenceChunk, build_windows
from src.nlp.ner import Entity
from src.nlp.coref_resolver import RuleBasedCorefResolver
from src.nlp.relation_extractor import DepPatternExtractor, RawTriple, _dep_distance
from src.nlp.predicate_normalizer import PredicateNormalizer
from src.nlp.relation_scorer import RelationScorer


def entity(text, name, kind, chunk_id="d:0"):
    start = text.index(name)
    return Entity(f"{chunk_id}:{start}:{start + len(name)}", name, name, kind,
                  start, start + len(name), "test", chunk_id)


def chunk(text, index=0):
    return SentenceChunk(f"d:{index}", "d", index, text, 0, len(text))


@pytest.mark.parametrize("text,subject,subject_type,obj,obj_type,relation", [
    ("Alice works for Google.", "Alice", "PERSON", "Google", "ORG", "EMPLOYED_BY"),
    ("Alice works at Google.", "Alice", "PERSON", "Google", "ORG", "EMPLOYED_BY"),
    ("Tesla is headquartered in Austin.", "Tesla", "ORG", "Austin", "GPE", "LOCATED_IN"),
    ("Google is based in London.", "Google", "ORG", "London", "GPE", "LOCATED_IN"),
    ("Google invested in Tesla.", "Google", "ORG", "Tesla", "ORG", "INVESTED_IN"),
    ("Google partnered with Microsoft.", "Google", "ORG", "Microsoft", "ORG", "PARTNER_WITH"),
    ("Microsoft was acquired by Google.", "Google", "ORG", "Microsoft", "ORG", "ACQUIRED"),
    ("Tesla was founded by Elon Musk.", "Elon Musk", "PERSON", "Tesla", "ORG", "FOUNDED"),
])
def test_exact_relation_patterns(nlp, text, subject, subject_type, obj, obj_type, relation):
    entities = [entity(text, subject, subject_type), entity(text, obj, obj_type)]
    raw = DepPatternExtractor().extract(nlp(text), entities, "d:0", 0)
    normalized = PredicateNormalizer().normalize_batch(raw)
    assert {(t.subject.text, t.relation_type, t.object.text) for t in normalized} == {
        (subject, relation, obj)
    }


def test_windows_keep_all_relations_and_actual_evidence(nlp):
    texts = ["Elon Musk founded Tesla.", "Google acquired Microsoft.", "Alice works for Google."]
    chunks = [chunk(text, i) for i, text in enumerate(texts)]
    names = [[("Elon Musk", "PERSON"), ("Tesla", "ORG")],
             [("Google", "ORG"), ("Microsoft", "ORG")],
             [("Alice", "PERSON"), ("Google", "ORG")]]
    entities = [entity(c.text, name, kind, c.chunk_id) for c, pairs in zip(chunks, names) for name, kind in pairs]
    triples = []
    for window, center in build_windows(chunks):
        triples.extend(DepPatternExtractor().extract(
            nlp(" ".join(c.text for c in window)), entities, center.chunk_id,
            center.sent_idx, window_chunks=window,
        ))
    facts = {(t.subject.text, t.predicate, t.object.text, t.chunk_id, t.sent_idx) for t in triples}
    assert facts == {
        ("Elon Musk", "found", "Tesla", "d:0", 0),
        ("Google", "acquire", "Microsoft", "d:1", 1),
        ("Alice", "work", "Google", "d:2", 2),
    }


@pytest.mark.parametrize("text,pairs", [
    ("Google did not acquire Microsoft.", [("Google", "ORG"), ("Microsoft", "ORG")]),
    ("Alice does not work for Google.", [("Alice", "PERSON"), ("Google", "ORG")]),
    ("Tesla is not headquartered in Austin.", [("Tesla", "ORG"), ("Austin", "GPE")]),
])
def test_negated_relations_are_not_asserted(nlp, text, pairs):
    raw = DepPatternExtractor().extract(nlp(text), [entity(text, n, k) for n, k in pairs], "d:0", 0)
    assert raw and all(t.has_negation for t in raw)
    assert PredicateNormalizer().normalize_batch(raw) == []
    scorer = RelationScorer()
    scorer._model = MagicMock()  # Even an overconfident model cannot assert negation.
    assert scorer.score_batch(raw) == [0.0] * len(raw)
    scorer._model.predict_proba.assert_not_called()


@pytest.mark.parametrize("second,name,kind,pronoun", [
    ("He hired Alice.", "Alice", "PERSON", "he"),
    ("It acquired Microsoft.", "Microsoft", "ORG", "it"),
    ("The company acquired Microsoft.", "Microsoft", "ORG", "the company"),
])
def test_coreference_cannot_look_forward(second, name, kind, pronoun):
    first = "Elon Musk founded Tesla."
    chunks = [chunk(first), chunk(second, 1)]
    musk, tesla = entity(first, "Elon Musk", "PERSON"), entity(first, "Tesla", "ORG")
    mapping = RuleBasedCorefResolver().resolve(chunks, {
        "d:0": [musk, tesla], "d:1": [entity(second, name, kind, "d:1")],
    })
    assert mapping[f"d:1:{pronoun}"] == (musk if kind == "PERSON" else tesla).entity_id


def test_same_sentence_recency_and_punctuation():
    text = "Alice said he thanked Bob and he left."
    alice, bob = entity(text, "Alice", "PERSON"), entity(text, "Bob", "PERSON")
    mapping = RuleBasedCorefResolver().resolve([chunk(text)], {"d:0": [bob, alice]})
    assert mapping[f"d:0:ref:{text.index('he')}"] == alice.entity_id
    assert mapping[f"d:0:ref:{text.rindex('he')}"] == bob.entity_id


def test_dependency_distance_includes_endpoints(nlp):
    doc = nlp("Alice joined Google.")
    assert _dep_distance(doc[0], doc[0]) == 0
    assert _dep_distance(doc[0], doc[1]) == 1
    assert _dep_distance(doc[0], doc[2]) == 2


def test_known_org_does_not_hide_sentence_boundary(nlp):
    from src.ingestion.preprocessor import preprocess
    text = "Elon Musk leads SpaceX. Tesla is headquartered in Austin."
    result = preprocess(text.encode(), "test.txt")
    assert [c.text for c in result.chunks] == [
        "Elon Musk leads SpaceX.", "Tesla is headquartered in Austin.",
    ]
    assert all(result.raw_text[c.start_char:c.end_char] == c.text for c in result.chunks)


@pytest.mark.parametrize("version,loaded", [(None, False), (1, False), (2, True)])
def test_scorer_artifact_version_guard(tmp_path, monkeypatch, version, loaded):
    import joblib
    import src.nlp.relation_scorer as module
    from sklearn.dummy import DummyClassifier
    model = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
    artifact = model if version is None else {"feature_version": version, "model": model}
    path = tmp_path / "scorer.joblib"
    joblib.dump(artifact, path)
    monkeypatch.setattr(module, "MODEL_PATH", path)
    assert (module.RelationScorer()._model is not None) is loaded


def test_feature_encoding_is_independent_of_encounter_order():
    import importlib
    import src.nlp.relation_scorer as module
    text = "Alice joined Google."
    triple = RawTriple(entity(text, "Alice", "PERSON"), "join", entity(text, "Google", "ORG"), None, "svo", 0, "d:0")
    first = module._feature_vector(triple, frozenset({"join"}))
    module = importlib.reload(module)
    module._feature_vector(replace(triple, subject=triple.object, object=triple.subject), frozenset({"join"}))
    assert first == module._feature_vector(triple, frozenset({"join"}))


@pytest.fixture
def pipeline_capture(monkeypatch, nlp):
    import src.pipeline as pipeline
    from src.graph.entity_resolver import EntityResolver
    builder = SimpleNamespace(**{name: AsyncMock() for name in (
        "write_document", "write_chunks", "write_entities", "write_mentions", "write_edges")})
    driver = MagicMock()
    driver.session.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
    driver.session.return_value.__aexit__ = AsyncMock(return_value=None)
    driver.close = AsyncMock()
    monkeypatch.setattr(pipeline, "_get_nlp", lambda: nlp)
    monkeypatch.setattr(pipeline, "get_settings", lambda: SimpleNamespace(
        window_size=3, window_stride=2, resolution_threshold=0.85,
        neo4j_uri="unused", neo4j_user="unused", neo4j_password="unused"))
    monkeypatch.setattr(pipeline.pipeline_lock, "acquire", lambda: None)
    monkeypatch.setattr(pipeline.pipeline_lock, "release", lambda: None)
    monkeypatch.setattr(pipeline, "create_driver", lambda *args: driver)
    monkeypatch.setattr(pipeline, "GraphBuilder", lambda: builder)
    monkeypatch.setattr(EntityResolver, "resolve_against_graph", AsyncMock(return_value=None))
    return pipeline, builder, driver


def test_cross_document_merges_preserve_mentions_and_aliases(pipeline_capture, monkeypatch):
    from src.graph.entity_resolver import EntityResolver
    pipeline, builder, driver = pipeline_capture
    async def existing(self, ent, session):
        return "existing:" + ent.name
    monkeypatch.setattr(EntityResolver, "resolve_against_graph", existing)
    result = pipeline.run_pipeline(b"Elon Musk founded Tesla.", "test.txt", "txt")
    entities = builder.write_entities.call_args.args[1]
    mentions = builder.write_mentions.call_args.args[1]
    edges = builder.write_edges.call_args.args[1]
    assert result["new_entities"] == 0
    assert {e.canonical_id for e in entities} == {"existing:Elon Musk", "existing:Tesla"}
    assert all(e.aliases for e in entities)
    assert set(mentions) == {e.canonical_id for e in entities}
    assert all(mentions.values())
    assert [(e.source_id, e.target_id) for e in edges] == [("existing:Elon Musk", "existing:Tesla")]
    driver.close.assert_awaited_once()


def test_pipeline_keeps_score_alignment_when_negation_is_removed(pipeline_capture, monkeypatch):
    pipeline, builder, _ = pipeline_capture
    monkeypatch.setattr(pipeline.RelationScorer, "score", lambda self, triple: 0.81 if triple.predicate == "found" else 0.13)
    pipeline.run_pipeline(b"Google did not acquire Microsoft. Elon Musk founded Tesla.", "test.txt", "txt")
    edges = builder.write_edges.call_args.args[1]
    assert len(edges) == 1
    assert edges[0].relation_type == "FOUNDED"
    assert edges[0].confidence == 0.81
    assert edges[0].chunk_ids[0].endswith(":1")
