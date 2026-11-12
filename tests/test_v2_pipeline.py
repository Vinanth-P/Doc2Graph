"""Integration tests for Milestone G: contextual_v2 backend and claim aggregation."""
from __future__ import annotations

import pytest

from src.config import get_settings
from src.nlp.contracts_v2 import (
    CanonicalClaimRecord,
    MentionRecord,
    RelationClaimRecord,
)
from src.nlp.claim_aggregation import aggregate_claims
from src.pipeline import run_pipeline_nlp


def test_default_backend_remains_rules():
    """Verify that settings default relation backend remains 'rules'."""
    settings = get_settings()
    assert settings.relation_backend == "rules"


def test_claim_aggregation_evidence_pooling():
    """Test aggregation of multiple mention claims into canonical claim with pooled evidence."""
    claims = [
        RelationClaimRecord(
            claim_id="c1",
            document_id="doc1",
            subject_mention_id="m1",
            object_mention_id="m2",
            relation="EMPLOYED_BY",
            polarity="affirmed",
            modality="asserted",
            lifecycle="completed",
            evidence_spans=[(0, 45)],
            evidence_text="Satya Nadella is CEO of Microsoft.",
            raw_score=0.88,
            calibrated_confidence=0.85,
        ),
        RelationClaimRecord(
            claim_id="c2",
            document_id="doc1",
            subject_mention_id="m3",  # another mention of Nadella
            object_mention_id="m2",
            relation="EMPLOYED_BY",
            polarity="affirmed",
            modality="asserted",
            lifecycle="completed",
            evidence_spans=[(100, 150)],
            evidence_text="Nadella joined Microsoft executive leadership.",
            raw_score=0.92,
            calibrated_confidence=0.90,
        ),
    ]

    id_map = {"m1": "E_NADELLA", "m3": "E_NADELLA", "m2": "E_MICROSOFT"}

    can_claims = aggregate_claims(claims, id_map)
    assert len(can_claims) == 1
    can = can_claims[0]

    assert can.canonical_subject_id == "E_NADELLA"
    assert can.canonical_object_id == "E_MICROSOFT"
    assert can.relation == "EMPLOYED_BY"
    assert can.polarity == "affirmed"
    assert len(can.member_claim_ids) == 2
    assert len(can.evidence_spans) == 2
    assert (0, 45) in can.evidence_spans
    assert (100, 150) in can.evidence_spans
    assert len(can.disagreements) == 0
    assert can.confidence >= 0.90


def test_claim_aggregation_disagreement_detection():
    """Test conflicting polarity claims produce an audited disagreement and unresolved polarity."""
    claims = [
        RelationClaimRecord(
            claim_id="c1",
            document_id="doc1",
            subject_mention_id="m1",
            object_mention_id="m2",
            relation="ACQUIRED",
            polarity="affirmed",
            evidence_spans=[(0, 30)],
            evidence_text="Company A acquired Company B.",
            raw_score=0.80,
        ),
        RelationClaimRecord(
            claim_id="c2",
            document_id="doc1",
            subject_mention_id="m1",
            object_mention_id="m2",
            relation="ACQUIRED",
            polarity="negated",
            evidence_spans=[(50, 95)],
            evidence_text="Company A later terminated acquisition of Company B.",
            raw_score=0.85,
        ),
    ]

    id_map = {"m1": "E_A", "m2": "E_B"}
    can_claims = aggregate_claims(claims, id_map)
    assert len(can_claims) == 1
    can = can_claims[0]

    assert can.polarity == "unknown"
    assert len(can.disagreements) == 1
    assert can.disagreements[0]["attribute"] == "polarity"
    assert set(can.disagreements[0]["values"]) == {"affirmed", "negated"}


def test_claim_aggregation_self_link_rejection():
    """Verify that claims where subject and object resolve to the same canonical entity are rejected."""
    claims = [
        RelationClaimRecord(
            claim_id="c1",
            document_id="doc1",
            subject_mention_id="m1",
            object_mention_id="m2",
            relation="SUBSIDIARY_OF",
            polarity="affirmed",
            evidence_spans=[(0, 20)],
            evidence_text="Acme Corp is Acme.",
            raw_score=0.75,
        )
    ]
    # m1 and m2 resolve to identical canonical entity
    id_map = {"m1": "E_ACME", "m2": "E_ACME"}
    can_claims = aggregate_claims(claims, id_map)
    assert len(can_claims) == 0


def test_pipeline_nlp_contextual_v2():
    """Test running run_pipeline_nlp with contextual_v2 backend."""
    settings = get_settings()
    orig_backend = settings.relation_backend
    orig_override = settings.contextual_v2_allow_unverified
    settings.relation_backend = "contextual_v2"
    settings.contextual_v2_allow_unverified = True
    try:
        sample_text = (
            "Microsoft acquired GitHub in 2018. "
            "Satya Nadella serves as chief executive officer of Microsoft. "
            "Goldman Sachs advised Microsoft on the transaction."
        )
        res = run_pipeline_nlp(sample_text.encode("utf-8"), "test_finance.txt", "txt")

        assert "resolved_entities" in res
        assert "normalized" in res
        assert "v2_claims" in res
        assert "v2_canonical_claims" in res
        assert res["resolver"].__class__.__name__ == "NonDestructiveResolver"
    finally:
        settings.relation_backend = orig_backend
        settings.contextual_v2_allow_unverified = orig_override
