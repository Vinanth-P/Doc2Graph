"""Tests for v2 data contracts, schema validation, and immutability."""
import pytest
from pydantic import ValidationError
from src.nlp.contracts_v2 import (
    DocumentRecord,
    MentionRecord,
    IdentityLinkRecord,
    RelationCandidateRecord,
    RelationClaimRecord,
    CanonicalClaimRecord,
    DocumentAnnotationPayload,
)


def test_missing_status_is_unknown_not_a_completed_assertion():
    claim = RelationClaimRecord(claim_id='c', document_id='d', subject_mention_id='a',
                                object_mention_id='b', relation='ACQUIRED')
    assert (claim.polarity, claim.modality, claim.lifecycle) == ('unknown', 'unknown', 'unknown')


@pytest.mark.parametrize('score', [-.1, 1.1, float('nan')])
def test_mention_scores_are_valid_probabilities(score):
    with pytest.raises(ValidationError):
        MentionRecord(id='a', document_id='d', text='A', type='ORG', start=0, end=1, score=score)


def test_payload_rejects_dangling_claim_and_misaligned_mention():
    mention = dict(id='a', document_id='d', text='A', type='ORG', start=0, end=1)
    claim = dict(claim_id='c', document_id='d', subject_mention_id='a', object_mention_id='missing', relation='OWNS')
    with pytest.raises(ValidationError, match='unknown mention'):
        DocumentAnnotationPayload(document_id='d', text='A', mentions=[mention], claims=[claim])
    with pytest.raises(ValidationError, match='text does not match'):
        DocumentAnnotationPayload(document_id='d', text='B', mentions=[mention], claims=[])


def test_document_record_hash_computation():
    raw = "Atlas acquired Beacon in 2024."
    doc = DocumentRecord(
        document_id="doc-1",
        source_name="test.pdf",
        raw_text=raw,
        normalized_text=raw,
    )
    assert doc.content_hash != ""
    assert doc.extraction_version == "2.0"
    assert doc.language == "en"
    
    # Verify immutability
    with pytest.raises(ValidationError):
        doc.raw_text = "new text"  # type: ignore


def test_mention_record_validation():
    mention = MentionRecord(
        id="m1",
        document_id="doc-1",
        text="Atlas",
        type="ORG",
        start=0,
        end=5,
        score=0.95,
    )
    assert mention.id == "m1"
    assert mention.end == 5
    
    # Negative start offset should fail
    with pytest.raises(ValidationError):
        MentionRecord(
            id="m2",
            document_id="doc-1",
            text="Invalid",
            type="ORG",
            start=-1,
            end=5,
        )

    # End before start should fail
    with pytest.raises(ValidationError):
        MentionRecord(
            id="m3",
            document_id="doc-1",
            text="Invalid",
            type="ORG",
            start=10,
            end=5,
        )


def test_relation_claim_assertion_attributes():
    claim = RelationClaimRecord(
        claim_id="c1",
        document_id="doc-1",
        subject_mention_id="m1",
        object_mention_id="m2",
        relation="ACQUIRED",
        polarity="affirmed",
        modality="planned",
        lifecycle="announced",
        evidence_spans=[(0, 30)],
        evidence_text="Atlas announced plans to acquire Beacon.",
        raw_score=0.88,
        calibrated_confidence=0.84,
    )
    assert claim.relation == "ACQUIRED"
    assert claim.modality == "planned"
    assert claim.lifecycle == "announced"
    assert claim.calibrated_confidence == 0.84
    
    # Invalid polarity should fail validation
    with pytest.raises(ValidationError):
        RelationClaimRecord(
            claim_id="c2",
            document_id="doc-1",
            subject_mention_id="m1",
            object_mention_id="m2",
            relation="ACQUIRED",
            polarity="invalid_polarity",  # type: ignore
        )


def test_annotation_payload_round_trip():
    payload_dict = {
        "schema_version": "2",
        "document_id": "example-doc",
        "text": "Atlas acquired Beacon.",
        "mentions": [
            {"id": "m1", "document_id": "example-doc", "start": 0, "end": 5, "text": "Atlas", "type": "ORG"},
            {"id": "m2", "document_id": "example-doc", "start": 15, "end": 21, "text": "Beacon", "type": "ORG"}
        ],
        "claims": [{
            "claim_id": "c1",
            "document_id": "example-doc",
            "subject_mention_id": "m1",
            "object_mention_id": "m2",
            "relation": "ACQUIRED",
            "polarity": "affirmed",
            "modality": "asserted",
            "lifecycle": "completed",
            "evidence_spans": [[0, 22]]
        }]
    }
    payload = DocumentAnnotationPayload.model_validate(payload_dict)
    assert len(payload.mentions) == 2
    assert len(payload.claims) == 1
    assert payload.claims[0].relation == "ACQUIRED"
    assert payload.claims[0].evidence_spans == [(0, 22)]
    
    # Dump and check json compatibility
    dumped = payload.model_dump()
    assert dumped["document_id"] == "example-doc"
