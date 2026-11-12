"""Validate the evaluator independently of model accuracy."""
from types import SimpleNamespace

from scripts.evaluate_external import reconstruct, gold_facts, metrics, summarize, graph_predictions
from src.nlp.ner import Entity
from src.nlp.predicate_normalizer import NormalizedTriple


def test_gold_directions_and_deduplication():
    record = {"labels": [{"h": 0, "t": 1, "r": "P112"},
                         {"h": 0, "t": 1, "r": "P112"},
                         {"h": 0, "t": 1, "r": "P127"},
                         {"h": 0, "t": 1, "r": "P108"},
                         {"h": 0, "t": 1, "r": "unsupported"}]}
    assert gold_facts(record) == {("1", "FOUNDED", "0"), ("1", "OWNS", "0"), ("0", "EMPLOYED_BY", "1")}


def test_token_offsets_include_sentence_boundaries():
    record = {"sents": [["Alice", "joined", "Acme", "."], ["She", "left", "."]],
              "vertexSet": [[{"sent_id": 0, "pos": [0, 1], "type": "PER"},
                             {"sent_id": 1, "pos": [0, 1], "type": "PER"}],
                            [{"sent_id": 0, "pos": [2, 3], "type": "ORG"}]]}
    text, _, _, mentions = reconstruct(record)
    assert [text[m["start"]:m["end"]] for m in mentions] == ["Alice", "She", "Acme"]
    assert mentions[0]["cluster"] == mentions[1]["cluster"]


def test_empty_predictions_are_false_negatives():
    assert metrics(set(), {("0", "FOUNDED", "1")}) == {"tp": 0, "fp": 0, "fn": 1}
    assert summarize({"tp": 1, "fp": 1, "fn": 3})["f1"] == 1 / 3


def test_unicode_normalization_precedes_gold_offsets():
    record = {"sents": [["The", "\ufb01rm", "hired", "Alice", "."]],
              "vertexSet": [[{"sent_id": 0, "pos": [3, 4], "type": "PER"}]]}
    text, _, _, mentions = reconstruct(record)
    assert "firm" in text
    assert text[mentions[0]["start"]:mentions[0]["end"]] == "Alice"


def test_endpoints_require_exact_typed_spans_and_do_not_use_gold_labels():
    a = Entity("a", "Alice", "Alice", "PERSON", 0, 5, "spacy", "c")
    b = Entity("b", "Acme", "Acme", "ORG", 13, 17, "spacy", "c")
    result = {"chunks": [SimpleNamespace(chunk_id="c", start_char=0)], "all_entities": [a, b],
              "id_map": {"a": "a", "b": "b"},
              "normalized": [NormalizedTriple("EMPLOYED_BY", a, b, "join", "c", 0)]}
    mentions = [{"start": 0, "end": 5, "type": "PERSON", "cluster": 0},
                {"start": 13, "end": 17, "type": "ORG", "cluster": 1}]
    facts, _, _, _ = graph_predictions(result, mentions)
    assert facts == {("0", "EMPLOYED_BY", "1")}
    mentions[1]["type"] = "PERSON"
    facts, _, _, _ = graph_predictions(result, mentions)
    assert facts == {("0", "EMPLOYED_BY", "unmatched:b")}


def test_ambiguous_merged_endpoint_is_not_assigned_a_favorable_cluster():
    a = Entity("a", "Alice", "Alice", "PERSON", 0, 5, "spacy", "c")
    b = Entity("b", "Acme", "Acme", "ORG", 13, 17, "spacy", "c")
    duplicate = Entity("d", "Other", "Other", "PERSON", 20, 25, "spacy", "c")
    result = {"chunks": [SimpleNamespace(chunk_id="c", start_char=0)], "all_entities": [a, b, duplicate],
              "id_map": {"a": "a", "b": "b", "d": "a"},
              "normalized": [NormalizedTriple("EMPLOYED_BY", a, b, "join", "c", 0)]}
    mentions = [{"start": 0, "end": 5, "type": "PERSON", "cluster": 0},
                {"start": 13, "end": 17, "type": "ORG", "cluster": 1},
                {"start": 20, "end": 25, "type": "PERSON", "cluster": 2}]
    facts, _, _, _ = graph_predictions(result, mentions)
    assert facts == {("unmatched:a", "EMPLOYED_BY", "1")}


def test_oracle_cluster_ids_are_not_fuzzy_merged(nlp):
    from scripts.evaluate_external import oracle_predictions
    from src.nlp.predicate_normalizer import PredicateNormalizer
    record = {"sents": [["Alice", "founded", "Acme", "."], ["Bob", "founded", "Beta", "."]],
              "vertexSet": [[{"sent_id": 0, "pos": [0, 1], "type": "PER"}],
                            [{"sent_id": 0, "pos": [2, 3], "type": "ORG"}],
                            [{"sent_id": 1, "pos": [0, 1], "type": "PER"}],
                            [{"sent_id": 1, "pos": [2, 3], "type": "ORG"}]]}
    _, sentences, starts, mentions = reconstruct(record)
    assert oracle_predictions(record, sentences, starts, mentions, nlp, PredicateNormalizer(), None) == {
        ("0", "FOUNDED", "1"), ("2", "FOUNDED", "3")}


def test_bootstrap_identical_reports_have_zero_difference():
    from scripts.compare_external_evaluation import compare
    fact = ["0", "FOUNDED", "1"]
    score = summarize({"tp": 1, "fp": 0, "fn": 0})
    report = {"dataset_sha256": "same", "documents": 1, "mapped_gold_facts": 1,
              "metrics": {mode: {scope: score for scope in ("shared_three", "all_five")}
                          for mode in ("end_to_end", "gold_entities")},
              "documents_detail": [{"index": 0, "title": "Example", "gold": [fact],
                                    "end_to_end": {"predicted": [fact]}, "gold_entities": {"predicted": [fact]}}]}
    compared = compare(report, report)
    for mode in compared["comparisons"].values():
        for result in mode.values():
            assert result["delta_f1"] == 0
            assert result["delta_f1_95pct_paired_bootstrap_interval"] == [0, 0]
