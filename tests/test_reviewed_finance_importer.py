"""Tests for the reviewed finance annotation workflow, validation, and importer."""
import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from src.nlp.contracts_v2 import ReviewedDocumentPayload, ReviewedRelationAnnotation, MentionRecord
from src.nlp.deduplication import compute_content_hash, normalize_for_dedup, analyze_duplicates_and_groups
from src.nlp.dataset_v2 import finance_split
from scripts.import_reviewed_finance import import_finance_data, convert_reviewed_document
from scripts.propose_finance_annotations import propose_annotations_for_text


def sample_doc(doc_id="d1", text="Atlas acquired Beacon.", relations=None, group="deal:atlas_beacon"):
    mentions = [
        MentionRecord(id="m1", document_id=doc_id, text="Atlas", type="ORG", start=0, end=5),
        MentionRecord(id="m2", document_id=doc_id, text="Beacon", type="ORG", start=15, end=21),
    ]
    if relations is None:
        relations = [
            ReviewedRelationAnnotation(
                annotation_id="a1",
                subject_mention_id="m1",
                object_mention_id="m2",
                relation="ACQUIRED",
                decision="positive",
                polarity="affirmed",
                modality="asserted",
                lifecycle="completed",
                evidence_spans=[(0, len(text))],
                evidence_text=text,
                reviewer="rev_1",
                review_status="reviewed",
            )
        ]
    return {
        "schema_version": "2.finance.1",
        "document_id": doc_id,
        "title": "Atlas Beacon Deal",
        "text": text,
        "doc_group": group,
        "license_or_rights": "public",
        "reviewer": "rev_1",
        "review_status": "reviewed",
        "data_origin": "human_reviewed_source",
        "review_timestamp": "2026-11-11T12:00:00Z",
        "review_guidelines_version": "2.0",
        "annotation_scope": "exhaustive_corporate_finance",
        "mentions": [m.model_dump() for m in mentions],
        "relation_annotations": [r.model_dump() if hasattr(r, "model_dump") else r for r in relations],
    }


def test_positive_annotation_requires_evidence():
    invalid_rel = {
        "annotation_id": "a1",
        "subject_mention_id": "m1",
        "object_mention_id": "m2",
        "relation": "ACQUIRED",
        "decision": "positive",
        "evidence_spans": [],  # Empty evidence for positive relation must fail
    }
    with pytest.raises(ValidationError, match="require at least one valid evidence span"):
        ReviewedRelationAnnotation.model_validate(invalid_rel)


def test_contradictory_positive_and_negative_annotations_fail():
    doc_dict = sample_doc(
        relations=[
            ReviewedRelationAnnotation(
                annotation_id="a1",
                subject_mention_id="m1",
                object_mention_id="m2",
                relation="ACQUIRED",
                decision="positive",
                evidence_spans=[(0, 22)],
            ),
            ReviewedRelationAnnotation(
                annotation_id="a2",
                subject_mention_id="m1",
                object_mention_id="m2",
                relation="ACQUIRED",
                decision="explicit_negative",
            ),
        ]
    )
    with pytest.raises(ValidationError, match="Contradictory positive and negative"):
        ReviewedDocumentPayload.model_validate(doc_dict)


def test_deduplication_clusters_near_duplicates_into_same_group():
    doc1 = sample_doc("doc_1", "Microsoft acquired GitHub for $7.5 billion in an all-stock transaction.", group="deal:g1")
    doc2_near = sample_doc("doc_2", "Microsoft completed the acquisition of GitHub for $7.5 billion in an all-stock transaction.", group="deal:g2")
    doc3_exact = sample_doc("doc_3", "Microsoft acquired GitHub for $7.5 billion in an all-stock transaction.", group="deal:g3")

    kept, exclusions, group_map = analyze_duplicates_and_groups([doc1, doc2_near, doc3_exact], near_threshold=0.60)
    
    assert len(kept) == 2
    assert len(exclusions) == 1
    assert exclusions[0]["document_id"] == "doc_3"
    # Near duplicates doc_1 and doc_2 must share the exact same cluster root group
    assert group_map["doc_1"] == group_map["doc_2"]


def test_import_finance_pipeline_run(tmp_path):
    pilot_input = Path("data/finance_pilot/pilot_reviewed_annotations.json")
    ontology_path = Path("config/ontology_v2.yaml")
    output_dir = tmp_path / "prepared_finance"

    manifest = import_finance_data(
        input_path=pilot_input,
        ontology_path=ontology_path,
        output_dir=output_dir,
        near_dedup_threshold=0.80,
        allow_fixtures=True,
    )

    assert manifest["locked_test_sealed"] is False
    assert manifest['release_holdout_available'] is False
    assert manifest['contains_test_fixtures'] is True
    # Same text with different annotation payloads must not be discarded.
    assert len(manifest["exclusions"]) == 0

    # Check near-duplicate grouping: fin_doc_001 and fin_doc_005_near_dup must be in same split
    assignments = {a["document_id"]: a["split"] for a in manifest["assignments"]}
    assert assignments["fin_doc_001"] == assignments["fin_doc_005_near_dup"]

    # Verify output JSONL records and loss masking
    all_rows = []
    for split in ("train", "validation", "calibration", "locked_test"):
        split_file = output_dir / f"{split}.jsonl"
        assert split_file.exists()
        for line in split_file.read_text(encoding="utf-8").splitlines():
            all_rows.append(json.loads(line))

    assert len(all_rows) == 6
    
    # Verify supervision masks: must have positives, explicit negatives, and unreviewed pairs
    has_positive = False
    has_negative = False
    has_unknown = False

    for row in all_rows:
        assert row["schema_version"] == "2.dataset.2"
        for candidate in row["candidates"]:
            for target, mask in zip(candidate["targets"], candidate["loss_mask"]):
                if target == 1:
                    has_positive = True
                    assert mask is True
                elif target == 0:
                    has_negative = True
                    assert mask is True
                elif target is None:
                    has_unknown = True
                    assert mask is False

    assert has_positive, "Dataset must contain explicit positive targets"
    assert has_negative, "Dataset must contain explicit negative targets"
    assert has_unknown, "Dataset must contain unreviewed unknown targets"

    # Check overwrite protection
    with pytest.raises(FileExistsError):
        import_finance_data(pilot_input, ontology_path, output_dir)


def test_propose_annotations_flags_pending_human_review():
    text = "Salesforce acquired Slack to integrate real-time collaboration. Marc Benioff leads Salesforce."
    ontology = {
        "relations": {
            "ACQUIRED": {"entity_type_pairs": [["ORG", "ORG"]]},
            "EMPLOYED_BY": {"entity_type_pairs": [["PERSON", "ORG"]]},
        }
    }
    proposal = propose_annotations_for_text("prop_doc_1", "Test Deal", text, "deal:salesforce_slack", ontology)
    assert proposal["schema_version"] == "2.finance.1"
    assert len(proposal["mentions"]) >= 2
    for ann in proposal["relation_annotations"]:
        assert ann["review_status"] == "pending_human_review"
        assert ann["reviewer"] == "proposal_engine_unreviewed"
