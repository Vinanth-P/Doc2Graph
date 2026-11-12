"""Candidate, inference and training-split guards for the learned backend."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.ingestion.preprocessor import SentenceChunk
from src.nlp.ner import Entity
from src.nlp.learned_relations import build_candidates, LearnedRelationExtractor, RELATION_TYPES, load_learned_extractor
from scripts.train_learned_relations import exclude_overlap, tune_threshold


def sample(names=("Alice", "Acme"), sentence="{0} negotiated with {1}."):
    text = sentence.format(*names)
    c = SentenceChunk("d:0", "d", 0, text, 0, len(text))
    entities = [Entity(str(i), name, name, kind, text.index(name), text.index(name) + len(name), "test", c.chunk_id, text)
                for i, (name, kind) in enumerate(zip(names, ("PERSON", "ORG")))]
    return [c], entities, {e.entity_id: e.entity_id for e in entities}


def test_candidates_do_not_require_a_whitelisted_verb():
    chunks, entities, ids = sample(sentence="{0}; {1}.")
    candidates = build_candidates(chunks, entities, ids, {("PERSON", "ORG")})
    assert len(candidates) == 1
    assert "SUBJECT_PERSON" in candidates[0].text and "OBJECT_ORG" in candidates[0].text


def test_entity_names_are_masked_and_renaming_does_not_change_features():
    a = build_candidates(*sample(), {("PERSON", "ORG")})[0]
    b = build_candidates(*sample(("Meera", "Globex")), {("PERSON", "ORG")})[0]
    assert a.text == b.text
    assert "Alice" not in a.text and "Acme" not in a.text


def test_cross_sentence_candidates_preserve_both_evidence_chunks():
    chunks, entities, ids = sample()
    chunks = [SentenceChunk("d:0", "d", 0, "Alice spoke.", 0, 12),
              SentenceChunk("d:1", "d", 1, "Acme responded.", 13, 28)]
    entities[1].chunk_id, entities[1].start_char, entities[1].end_char = "d:1", 0, 4
    candidate = build_candidates(chunks, entities, ids, {("PERSON", "ORG")})[0]
    assert candidate.chunk_ids == ("d:0", "d:1")
    assert "DISTANCE_1" in candidate.text
    assert candidate.evidence == "Alice spoke. Acme responded."


def test_self_pairs_are_never_candidates():
    chunks, entities, ids = sample()
    ids = {e.entity_id: "one" for e in entities}
    assert build_candidates(chunks, entities, ids, {("PERSON", "ORG")}) == []


def test_long_sentence_retains_both_target_markers():
    chunks, entities, ids = sample(sentence="{0} " + "word " * 300 + "{1}.")
    candidate = build_candidates(chunks, entities, ids, {("PERSON", "ORG")})[0]
    assert "SUBJECT_PERSON" in candidate.text and "OBJECT_ORG" in candidate.text
    assert "CONTEXT_BREAK" in candidate.text


def test_missing_or_incompatible_artifact_fails_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train it"):
        load_learned_extractor(tmp_path / "missing.joblib")
    with pytest.raises(ValueError, match="feature version"):
        LearnedRelationExtractor({"feature_version": -1})
    with pytest.raises(ValueError, match="label order"):
        LearnedRelationExtractor({"feature_version": 1, "relations": ["BAD"]})


def test_train_dev_overlap_excluded_without_changing_dev():
    record = lambda title, words: {"title": title, "sents": [words.split()], "vertexSet": []}
    dev = [record("Dev", "same text")]
    training = [record("different title", "same text"), record("DEV", "other text"),
                record("Train", "training text"), record("Train", "training text")]
    kept, excluded = exclude_overlap(training, dev)
    assert len(kept) == 1 and len(excluded) == 3
    assert len(dev) == 1 and dev[0]["title"] == "Dev"


def test_no_supported_dev_threshold_disables_relation():
    assert tune_threshold(np.array([0, 1, 0]), np.array([0.1, 0.8, 0.3]), 1) > 1


def test_model_predictions_not_rule_predictions():
    class Vectorizer:
        def transform(self, values):
            return np.zeros((len(list(values)), 1))
    class Classifier:
        def __init__(self, score): self.score = score
        def predict_proba(self, values):
            return np.tile([1 - self.score, self.score], (len(values), 1))
    artifact = {"feature_version": 1, "relations": RELATION_TYPES,
                "type_pairs": {r: [["PERSON", "ORG"]] for r in RELATION_TYPES},
                "vectorizer": Vectorizer(), "classifiers": [Classifier(.9)] + [Classifier(.1)] * 4,
                "thresholds": {r: .5 for r in RELATION_TYPES}}
    triples = LearnedRelationExtractor(artifact).extract(*sample(sentence="{0}; {1}."))
    assert [t.relation_type for t in triples] == ["FOUNDED"]
    assert triples[0].extractor == "learned_pair_classifier"


def test_learned_backend_bypasses_dependency_rules(monkeypatch):
    import src.pipeline as pipeline
    import src.nlp.learned_relations as learned
    settings = SimpleNamespace(relation_backend="learned", learned_relation_model_path="unused",
                               resolution_threshold=.85)
    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    monkeypatch.setattr(pipeline.pipeline_lock, "acquire", lambda: None)
    monkeypatch.setattr(pipeline.pipeline_lock, "release", lambda: None)
    calls = []
    monkeypatch.setattr(learned, "load_learned_extractor", lambda path: SimpleNamespace(
        extract=lambda chunks, entities, ids: calls.append((chunks, entities, ids)) or []))
    monkeypatch.setattr(pipeline, "DepPatternExtractor", lambda: pytest.fail("Rule extractor ran"))
    result = pipeline.run_pipeline_nlp(b"Alice founded Google in California.", "test.txt", "txt")
    assert calls and result["normalized"] == [] and result["resolved_entities"]
