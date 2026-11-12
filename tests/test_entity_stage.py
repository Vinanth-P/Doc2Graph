"""Tests for entity stage benchmarking, B-cubed metrics, and non-destructive resolution."""
import pytest
from src.nlp.contracts_v2 import MentionRecord
from src.nlp.contextual_ner import (
    pairwise_merge_metrics,
    bcubed_metrics,
    NonDestructiveResolver,
)


def test_pairwise_merge_metrics_calculation():
    # Gold: cluster_1 has {m1, m2}, cluster_2 has {m3, m4}
    # Gold pairs: (m1, m2), (m3, m4)
    gold = {"c1": ["m1", "m2"], "c2": ["m3", "m4"]}

    # Perfect prediction
    pred_perfect = {"m1": "p1", "m2": "p1", "m3": "p2", "m4": "p2"}
    res = pairwise_merge_metrics(gold, pred_perfect)
    assert res["tp"] == 2
    assert res["fp"] == 0
    assert res["fn"] == 0
    assert res["f1"] == 1.0

    # Over-merged: merged everything into one cluster
    # Pred pairs: (m1,m2), (m1,m3), (m1,m4), (m2,m3), (m2,m4), (m3,m4) -> 6 pairs
    pred_over = {"m1": "p1", "m2": "p1", "m3": "p1", "m4": "p1"}
    res_over = pairwise_merge_metrics(gold, pred_over)
    assert res_over["tp"] == 2
    assert res_over["fp"] == 4
    assert res_over["fn"] == 0
    assert res_over["precision"] == 2 / 6

    # Under-merged: each mention is its own cluster
    pred_under = {"m1": "p1", "m2": "p2", "m3": "p3", "m4": "p4"}
    res_under = pairwise_merge_metrics(gold, pred_under)
    assert res_under["tp"] == 0
    assert res_under["fp"] == 0
    assert res_under["fn"] == 2
    assert res_under["recall"] == 0.0


def test_bcubed_metrics_calculation():
    gold = {"c1": ["m1", "m2"], "c2": ["m3", "m4"]}
    pred_perfect = {"m1": "p1", "m2": "p1", "m3": "p2", "m4": "p2"}
    res = bcubed_metrics(gold, pred_perfect)
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0


def test_nondestructive_resolver_preserves_mentions_and_rejects_conflicts():
    mentions = [
        MentionRecord(id="m1", document_id="d1", text="Apple Inc.", type="ORG", start=0, end=10),
        MentionRecord(id="m2", document_id="d1", text="Apple", type="ORG", start=20, end=25),
        MentionRecord(id="m3", document_id="d1", text="John Smith", type="PERSON", start=40, end=50),
        MentionRecord(id="m4", document_id="d1", text="Jane Smith", type="PERSON", start=60, end=70),
        MentionRecord(id="m5", document_id="d1", text="IBM", type="ORG", start=80, end=83),
        MentionRecord(id="m6", document_id="d1", text="International Business Machines", type="ORG", start=90, end=121),
    ]

    resolver = NonDestructiveResolver(threshold=0.80)
    mapping, links = resolver.cluster_mentions(mentions)

    # 1. Mention IDs preserved in mapping
    assert set(mapping.keys()) == {"m1", "m2", "m3", "m4", "m5", "m6"}

    # 2. Apple Inc. and Apple linked
    assert mapping["m1"] == mapping["m2"]

    # 3. Person surname conflict: John Smith and Jane Smith must NOT be linked
    assert mapping["m3"] != mapping["m4"]
    conflict_link = next(
        l for l in links
        if {l.source_mention_id, l.target_mention_id} == {"m3", "m4"}
    )
    assert conflict_link.decision == "unlinked"
    assert conflict_link.method == "person_surname_conflict"

    # 4. IBM and International Business Machines linked
    assert mapping["m5"] == mapping["m6"]
