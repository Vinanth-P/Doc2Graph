"""Regression tests reproducing and verifying blockers identified in CONTEXTUAL_V2_COMPLETION_REVIEW.md."""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from src.nlp.contracts_v2 import MentionRecord, RelationClaimRecord
from src.nlp.contextual_relations import (
    RELATION_TYPES,
    POLARITY_TYPES,
    MODALITY_TYPES,
    LIFECYCLE_TYPES,
    LIFECYCLE_TO_IDX,
    SimpleVocabulary,
    ContextualRelationModel,
    ContextualRelationExtractor,
    build_marked_context,
)
from scripts.train_contextual_relations import (
    extract_samples_from_records,
    load_dataset_records,
)


def test_regression_trainer_never_relabels_unknown_as_negative():
    """Blocker 1: Trainer must NEVER turn unannotated/unknown pairs into supervised negative zeros."""
    # Fixture with 1 positive relation on candidate 0, and unannotated candidates with loss_mask all False
    doc = {
        "schema_version": "2.dataset.1",
        "document_id": "test_doc_0",
        "text": "Acme Corp acquired Beta Inc in 2020. Gamma LLC also operates in New York.",
        "mentions": [
            {"id": "m1", "document_id": "test_doc_0", "text": "Acme Corp", "type": "ORG", "start": 0, "end": 9},
            {"id": "m2", "document_id": "test_doc_0", "text": "Beta Inc", "type": "ORG", "start": 19, "end": 27},
            {"id": "m3", "document_id": "test_doc_0", "text": "Gamma LLC", "type": "ORG", "start": 42, "end": 51},
        ],
        "annotated_groups": {"m1": ["m1"], "m2": ["m2"], "m3": ["m3"]},
        "gold_relations": [["m1", "ACQUIRED", "m2"]],
        "candidates": [
            {
                "subject_group": "m1",
                "object_group": "m2",
                "type_supported": True,
                "targets": [None, None, None, None, None, 1, None, None, None],
                "loss_mask": [False, False, False, False, False, True, False, False, False],
            },
            {
                "subject_group": "m1",
                "object_group": "m3",
                "type_supported": True,
                "targets": [None] * len(RELATION_TYPES),
                "loss_mask": [False] * len(RELATION_TYPES),  # Unknown! Zero explicit negatives!
            },
            {
                "subject_group": "m3",
                "object_group": "m2",
                "type_supported": True,
                "targets": [None] * len(RELATION_TYPES),
                "loss_mask": [False] * len(RELATION_TYPES),  # Unknown!
            },
        ],
    }

    samples = extract_samples_from_records([doc])

    # Must NOT emit any all-zero fully supervised negative samples from unannotated candidates!
    for s in samples:
        mask = s["loss_mask"]
        targets = s["targets"]
        # If all targets are 0.0, it must be because the input was an explicit reviewed negative,
        # never a fabricated negative from an unannotated pair!
        if all(t == 0.0 for t in targets):
            assert False, "Found fabricated all-zero negative sample from unannotated candidate!"
        # Supervised mask entries must only exist where input candidate had loss_mask == True
        assert mask[RELATION_TYPES.index("ACQUIRED")] is True
        for idx, r in enumerate(RELATION_TYPES):
            if r != "ACQUIRED":
                assert mask[idx] is False, f"Mask for unannotated relation {r} was erroneously set to True!"


def test_regression_load_dataset_records_guards():
    """Blocker 9: load_dataset_records must fail on missing files and reject test fixtures unless allowed."""
    with pytest.raises(FileNotFoundError):
        load_dataset_records([Path("non_existent_file_12345.jsonl")])

    fixture_doc = {
        "schema_version": "2.dataset.2",
        "document_id": "fix_1",
        "data_origin": "test_fixture",
        "text": "Sample text.",
        "mentions": [],
        "candidates": [],
    }
    tmp_path = Path("output/temp_fixture_test.jsonl")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(json.dumps(fixture_doc) + "\n", encoding="utf-8")

    try:
        # Default load must reject test fixture
        with pytest.raises(ValueError, match="test_fixture"):
            load_dataset_records([tmp_path], allow_fixtures=False)

        # Explicit allow_fixtures=True succeeds
        recs = load_dataset_records([tmp_path], allow_fixtures=True)
        assert len(recs) == 1
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_regression_truncation_detection_and_missing_markers():
    """Blocker 4: Contexts exceeding budget must be flagged truncated=True and abstain when markers are missing."""
    vocab = SimpleVocabulary()
    vocab.build_from_texts(["Alice", "Acme", "word"])

    # 150 intervening tokens between Alice and Acme
    intervening = " ".join(["word"] * 150)
    text = f"Alice {intervening} Acme"

    mentions = [
        MentionRecord(id="m1", document_id="doc1", text="Alice", type="PERSON", start=0, end=5),
        MentionRecord(id="m2", document_id="doc1", text="Acme", type="ORG", start=len(text) - 4, end=len(text)),
    ]

    model = ContextualRelationModel(vocab_size=len(vocab.token_to_idx), embed_dim=32, hidden_dim=64, num_layers=1)
    extractor = ContextualRelationExtractor(model, vocab, thresholds={r: 0.0 for r in RELATION_TYPES})

    # Extractor must NOT emit a claim when argument markers are missing due to truncation!
    claims = extractor.extract_from_document("doc1", text, mentions, max_tokens=64)
    assert len(claims) == 0, "Extractor emitted a relation claim even though argument markers were truncated!"


def test_regression_relation_conditioned_assertions():
    """Blocker 3: Assertions must be relation-specific and not shared across different relations."""
    # Two simultaneous relations between same entities: ACQUIRED is planned, but OWNS is completed
    cand_data = {
        "subject_group": "m1",
        "object_group": "m2",
        "type_supported": True,
        "targets": [None, None, None, 1, None, 1, None, None, None],
        "loss_mask": [False, False, False, True, False, True, False, False, False],
        "assertions_by_relation": {
            "OWNS": {
                "targets": {"polarity": "affirmed", "modality": "asserted", "lifecycle": "completed"},
                "loss_mask": {"polarity": True, "modality": True, "lifecycle": True},
            },
            "ACQUIRED": {
                "targets": {"polarity": "affirmed", "modality": "planned", "lifecycle": "announced"},
                "loss_mask": {"polarity": True, "modality": True, "lifecycle": True},
            },
        },
    }

    doc = {
        "schema_version": "2.dataset.2",
        "document_id": "doc_rel_assert",
        "text": "Acme Corp announced plans to acquire Beta Inc. Acme Corp currently owns 15% of Beta Inc.",
        "mentions": [
            {"id": "m1", "document_id": "doc_rel_assert", "text": "Acme Corp", "type": "ORG", "start": 0, "end": 9},
            {"id": "m2", "document_id": "doc_rel_assert", "text": "Beta Inc", "type": "ORG", "start": 36, "end": 44},
        ],
        "annotated_groups": {"m1": ["m1"], "m2": ["m2"]},
        "gold_relations": [["m1", "OWNS", "m2"], ["m1", "ACQUIRED", "m2"]],
        "candidates": [cand_data],
    }

    samples = extract_samples_from_records([doc])
    # Must preserve per-relation assertion targets and masks without dropping ACQUIRED's planned status
    assert len(samples) >= 1
    sample = samples[0]
    assert "assertions_by_relation" in sample, "Sample does not contain relation-conditioned assertions!"
    assert sample["assertions_by_relation"]["ACQUIRED"]["targets"]["lifecycle"] == LIFECYCLE_TO_IDX["announced"]
    assert sample["assertions_by_relation"]["OWNS"]["targets"]["lifecycle"] == LIFECYCLE_TO_IDX["completed"]


def test_regression_claim_aggregation_transaction_identity_and_confidence_zero():
    """Blocker 6: Aggregator must preserve distinct deals/dates, handle confidence=0.0 without falsy fallback,
    and avoid unvalidated confidence bonuses."""
    from src.nlp.claim_aggregation import aggregate_claims

    # Two distinct transactions between the same companies at different dates
    claims = [
        RelationClaimRecord(
            claim_id="c_2018",
            document_id="doc1",
            subject_mention_id="m1",
            object_mention_id="m2",
            relation="ACQUIRED",
            polarity="affirmed",
            time_qualifiers={"year": "2018", "deal_id": "deal_1"},
            evidence_spans=[(0, 30)],
            evidence_text="Company A acquired Company B in 2018.",
            raw_score=0.85,
            calibrated_confidence=0.80,
        ),
        RelationClaimRecord(
            claim_id="c_2024",
            document_id="doc1",
            subject_mention_id="m1",
            object_mention_id="m2",
            relation="ACQUIRED",
            polarity="affirmed",
            time_qualifiers={"year": "2024", "deal_id": "deal_2"},
            evidence_spans=[(100, 140)],
            evidence_text="Company A acquired remaining assets in 2024.",
            raw_score=0.90,
            calibrated_confidence=0.0,  # Legitimate calibrated 0.0!
        ),
    ]

    id_map = {"m1": "E_A", "m2": "E_B"}
    can_claims = aggregate_claims(claims, id_map)

    # 1. Distinct transactions with distinct qualifiers must NOT be collapsed into one single timeless claim
    assert len(can_claims) == 2, "Distinct deals with conflicting year/deal_id were collapsed into one!"

    # 2. Check confidence=0.0 handling: calibrated_confidence of 0.0 must NOT be treated as falsy and replaced with raw_score
    c_2024_can = next(c for c in can_claims if c.time_qualifiers.get("year") == "2024")
    assert c_2024_can.confidence == 0.0, f"Calibrated 0.0 confidence was falsely replaced with {c_2024_can.confidence}!"


def test_regression_calibration_precision_gate_and_unknown_mask(tmp_path):
    """Blocker 2: Calibration must not label unknown pairs as zero, must not claim 'calibrated'
    when precision target fails, and must enforce abstention (disabled=True) for unsupported/failed relations."""
    from scripts.calibrate_v2 import run_calibration

    # Create dummy model dir
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    vocab = SimpleVocabulary()
    vocab.build_from_texts(["Acme", "Corp", "Beta", "Inc"])
    (model_dir / "vocab.json").write_text(json.dumps(vocab.to_dict()), encoding="utf-8")
    (model_dir / "training_manifest.json").write_text(
        json.dumps({"version": "2.0.0", "embed_dim": 16, "hidden_dim": 32, "num_layers": 1, "split": "training"}),
        encoding="utf-8",
    )

    model = ContextualRelationModel(
        vocab_size=len(vocab.token_to_idx),
        embed_dim=16,
        hidden_dim=32,
        num_layers=1,
    )
    torch.save(model.state_dict(), model_dir / "model.pt")

    # Document with 1 positive for FOUNDED, 1 UNANNOTATED pair (loss_mask False), 0 explicit negatives
    # For ACQUIRED, 0 positives and 0 negatives
    cal_doc = {
        "schema_version": "2.dataset.2",
        "document_id": "doc_cal_1",
        "split": "calibration",
        "text": "Acme Corp founded Beta Inc.",
        "mentions": [
            {"id": "m1", "document_id": "doc_cal_1", "text": "Acme Corp", "type": "ORG", "start": 0, "end": 9},
            {"id": "m2", "document_id": "doc_cal_1", "text": "Beta Inc", "type": "ORG", "start": 18, "end": 26},
        ],
        "annotated_groups": {"0": ["m1"], "1": ["m2"]},
        "gold_relations": [["0", "FOUNDED", "1"]],
        "candidates": [
            {
                "subject_group": "0",
                "object_group": "1",
                "type_supported": True,
                "targets": [1, None, None, None, None, None, None, None, None],
                "loss_mask": [True, False, False, False, False, False, False, False, False],
            }
        ],
    }
    cal_file = tmp_path / "calibration.jsonl"
    cal_file.write_text(json.dumps(cal_doc) + "\n", encoding="utf-8")

    out_json = tmp_path / "thresholds.json"
    result = run_calibration(
        model_dir=model_dir,
        calibration_file=cal_file,
        output_path=out_json,
        target_precision=0.70,
        min_support=1,
    )

    rels = result["relations"]

    # 1. ACQUIRED has 0 support -> MUST be disabled, threshold 1.01, NOT 0.80!
    assert rels["ACQUIRED"]["status"] == "unsupported_disabled"
    assert rels["ACQUIRED"]["disabled"] is True
    assert rels["ACQUIRED"]["threshold"] >= 1.0
    assert rels["ACQUIRED"]["abstention_policy"] == "enforced_abstention"

    # 2. Dataset sha256 and model sha256 must be recorded for custody
    assert "model_sha256" in result
    assert "calibration_file_sha256" in result

    # 3. ContextualRelationExtractor loading this thresholds dict must mark disabled relations
    extractor = ContextualRelationExtractor(
        model=model,
        vocab=vocab,
        thresholds={r: info["threshold"] for r, info in rels.items()},
        disabled_relations={r for r, info in rels.items() if info.get("disabled", False)},
    )
    assert "ACQUIRED" in extractor.disabled_relations

    # 4. Now test that if precision target fails, relation is marked uncalibrated_failed_target and disabled!
    # Doc with 1 positive and 3 explicit negatives for FOUNDED
    cal_doc_fail = {
        "schema_version": "2.dataset.2",
        "document_id": "doc_cal_fail",
        "split": "calibration",
        "text": "Acme Corp founded Beta Inc. Gamma LLC also operates here.",
        "mentions": [
            {"id": "m1", "document_id": "doc_cal_fail", "text": "Acme Corp", "type": "ORG", "start": 0, "end": 9},
            {"id": "m2", "document_id": "doc_cal_fail", "text": "Beta Inc", "type": "ORG", "start": 18, "end": 26},
            {"id": "m3", "document_id": "doc_cal_fail", "text": "Gamma LLC", "type": "ORG", "start": 28, "end": 37},
        ],
        "annotated_groups": {"0": ["m1"], "1": ["m2"], "2": ["m3"]},
        "gold_relations": [["0", "FOUNDED", "1"]],
        "candidates": [
            {
                "subject_group": "0",
                "object_group": "1",
                "type_supported": True,
                "targets": [1] + [None] * 8,
                "loss_mask": [True] + [False] * 8,
            },
            {
                "subject_group": "0",
                "object_group": "2",
                "type_supported": True,
                "targets": [0] + [None] * 8,  # explicit negative!
                "loss_mask": [True] + [False] * 8,
            },
            {
                "subject_group": "2",
                "object_group": "1",
                "type_supported": True,
                "targets": [0] + [None] * 8,  # explicit negative!
                "loss_mask": [True] + [False] * 8,
            },
        ],
    }
    cal_file_fail = tmp_path / "calibration_fail.jsonl"
    cal_file_fail.write_text(json.dumps(cal_doc_fail) + "\n", encoding="utf-8")

    out_json_fail = tmp_path / "thresholds_fail.json"
    result_fail = run_calibration(
        model_dir=model_dir,
        calibration_file=cal_file_fail,
        output_path=out_json_fail,
        target_precision=1.01,  # Impossible precision target
        min_support=1,
        min_negative_support=2,
    )
    founded_info = result_fail["relations"]["FOUNDED"]
    # Must NOT claim calibrated at failed precision!
    assert founded_info["status"] == "uncalibrated_failed_target"
    assert founded_info["disabled"] is True
    assert founded_info["threshold"] >= 1.0
    assert founded_info["abstention_policy"] == "abstain_precision_not_met"


def test_regression_evaluator_modes_and_dynamic_provenance():
    """Blocker 7: Evaluator must implement advertised modes (oracle, predicted resolution),
    dynamically check provenance for finance support (not hardcode 5 from test fixtures),
    and support matched label scope."""
    from scripts.evaluate_v2 import (
        evaluate_document_oracle,
        evaluate_document_predicted_resolution,
        audit_acceptance_gates,
        REDOCRED_RELATIONS,
    )
    from src.nlp.contextual_ner import NonDestructiveResolver

    # Test fixture doc with data_origin='test_fixture'
    doc = {
        "schema_version": "2.dataset.2",
        "document_id": "eval_test_doc",
        "data_origin": "test_fixture",
        "text": "Acme Corp founded Beta Inc.",
        "mentions": [
            {"id": "m1", "document_id": "eval_test_doc", "text": "Acme Corp", "type": "ORG", "start": 0, "end": 9},
            {"id": "m2", "document_id": "eval_test_doc", "text": "Beta Inc", "type": "ORG", "start": 18, "end": 26},
        ],
        "annotated_groups": {"0": ["m1"], "1": ["m2"]},
        "gold_relations": [["0", "FOUNDED", "1"]],
    }

    # Verify audit_acceptance_gates counts finance support dynamically from records:
    # A fixture doc MUST result in representative_finance_support == 0, never hard-coded 5!
    gate_audit = audit_acceptance_gates(
        results={"metrics": {"precision": 0.5, "recall": 0.2}, "bootstrap_95ci": {"precision": [0.4, 0.6]}},
        records=[doc],
        locked_test_sealed=False,
    )
    assert gate_audit["finance_support"] == 0, f"Expected 0 verified finance support, got {gate_audit['finance_support']}"
    assert gate_audit["research_gate"]["passed"] is False
    assert gate_audit["release_decision"] == "KEEP_DEFAULT_RULES"

    # Verify evaluate_document_oracle and evaluate_document_predicted_resolution execute with matched scope
    vocab = SimpleVocabulary()
    vocab.build_from_texts(["Acme", "Corp", "Beta", "Inc"])
    model = ContextualRelationModel(vocab_size=len(vocab.token_to_idx), embed_dim=16, hidden_dim=32, num_layers=1)
    extractor = ContextualRelationExtractor(model, vocab, thresholds={r: 0.99 for r in RELATION_TYPES})
    resolver = NonDestructiveResolver()

    res_oracle = evaluate_document_oracle(doc, extractor, evaluated_relations=REDOCRED_RELATIONS)
    assert "tp" in res_oracle and "fp" in res_oracle and "fn" in res_oracle
    assert res_oracle["gold_count"] == 1

    res_pred_res = evaluate_document_predicted_resolution(doc, extractor, resolver, evaluated_relations=REDOCRED_RELATIONS)
    assert "tp" in res_pred_res and "fp" in res_pred_res and "fn" in res_pred_res


@pytest.mark.asyncio
async def test_regression_database_persistence_and_distinct_transactions():
    """Task 1: Neo4j write_edges must write polarity, modality, lifecycle, time_qualifiers,
    claim_id, and deal_key, maintaining distinct transactions at the database boundary."""
    from unittest.mock import AsyncMock
    from src.graph.graph_builder import GraphBuilder, build_resolved_edges
    from src.nlp.predicate_normalizer import NormalizedTriple
    from src.nlp.ner import Entity
    from src.api.schemas import RelationResponse

    ent1 = Entity("m1", "Acme", "Acme", "ORG", 0, 4, "ner", "chunk1")
    ent2 = Entity("m2", "Beta", "Beta", "ORG", 10, 14, "ner", "chunk1")
    id_map = {"m1": "E_ACME", "m2": "E_BETA"}

    # Two distinct deals between same entities at different dates
    t1 = NormalizedTriple(
        relation_type="ACQUIRED",
        subject=ent1,
        object=ent2,
        confidence=0.9,
        raw_predicate="acquired 10%",
        evidence="Acme acquired 10% in 2018.",
        chunk_id="chunk1",
        sent_idx=0,
        polarity="affirmed",
        modality="asserted",
        lifecycle="completed",
        time_qualifiers={"year": "2018", "deal_id": "deal_1"},
        claim_id="claim_2018",
    )
    t2 = NormalizedTriple(
        relation_type="ACQUIRED",
        subject=ent1,
        object=ent2,
        confidence=0.8,
        raw_predicate="plans to acquire 90%",
        evidence="Acme plans to acquire remaining 90% in 2024.",
        chunk_id="chunk2",
        sent_idx=0,
        polarity="affirmed",
        modality="planned",
        lifecycle="announced",
        time_qualifiers={"year": "2024", "date": "2024-06-01"},
        claim_id="claim_2024",
    )

    resolved_edges = build_resolved_edges([t1, t2], id_map)
    # Distinct transactions must NOT collapse in memory
    assert len(resolved_edges) == 2, "Distinct deals collapsed in memory!"

    # Contradiction test: t3 merges with t1 with conflicting polarity
    t3 = NormalizedTriple(
        relation_type="ACQUIRED",
        subject=ent1,
        object=ent2,
        confidence=0.85,
        raw_predicate="acquired 10%",
        evidence="Acme did not acquire 10% in 2018.",
        chunk_id="chunk3",
        sent_idx=0,
        polarity="negated",  # Contradicting t1!
        modality="asserted",
        lifecycle="completed",
        time_qualifiers={"year": "2018", "deal_id": "deal_1"},
        claim_id="claim_2018",
    )
    merged_edges = build_resolved_edges([t1, t3], id_map)
    assert len(merged_edges) == 1
    assert merged_edges[0].polarity == "unknown", "Conflicting polarities on same deal were not set to 'unknown'!"

    # Verify Neo4j write_edges persists all assertion and event fields
    mock_session = AsyncMock()
    builder = GraphBuilder()
    await builder.write_edges(mock_session, resolved_edges)

    assert mock_session.run.called
    cypher_call = mock_session.run.call_args
    query = cypher_call[0][0]
    params = cypher_call[1]["edges"]

    # Verify Cypher query merges on deal_key / claim_id and writes assertion fields
    assert "deal_key: e.deal_key" in query
    assert "r.polarity" in query
    assert "r.modality" in query
    assert "r.lifecycle" in query
    assert "r.time_qualifiers" in query

    # Verify edge dicts passed to Cypher have all required fields
    assert len(params) == 2
    p1 = next(p for p in params if p["claim_id"] == "claim_2018")
    assert p1["polarity"] == "affirmed"
    assert p1["modality"] == "asserted"
    assert p1["lifecycle"] == "completed"
    assert "2018" in p1["time_qualifiers"]
    assert p1["deal_key"] == "deal_1"

    # Verify API schema round-trip
    resp = RelationResponse(
        type=p1["relation_type"],
        target_id=p1["target_id"],
        target_name="Beta",
        confidence=p1["confidence"],
        evidence_count=1,
        polarity=p1["polarity"],
        modality=p1["modality"],
        lifecycle=p1["lifecycle"],
        time_qualifiers=json.loads(p1["time_qualifiers"]),
        claim_id=p1["claim_id"],
        deal_key=p1["deal_key"],
    )
    assert resp.polarity == "affirmed"
    assert resp.time_qualifiers.get("year") == "2018"


def test_regression_artifact_validator_fails_closed_without_override():
    """Task 2: Artifact validator and pipeline loader must fail closed on unverified/rejected artifacts."""
    from src.nlp.artifact_guard import validate_model_artifacts
    from src.pipeline import run_pipeline_nlp
    from src.config import get_settings

    model_dir = Path("models/contextual_v2")

    # 1. validate_model_artifacts directly fails closed without research override
    with pytest.raises(ValueError, match="Artifact validation failed"):
        validate_model_artifacts(model_dir, allow_research_override=False)

    # 2. validate_model_artifacts succeeds with research override and flags provisional
    res = validate_model_artifacts(model_dir, allow_research_override=True)
    assert res.is_provisional is True
    assert res.artifact_status == "rejected_research_baseline"
    assert "FOUNDED" in res.disabled_relations

    # 3. Pipeline loader fails closed when contextual_v2 is selected without allow_unverified
    settings = get_settings()
    orig_backend = settings.relation_backend
    orig_override = settings.contextual_v2_allow_unverified
    settings.relation_backend = "contextual_v2"
    settings.contextual_v2_allow_unverified = False
    try:
        sample_text = "Acme Corp acquired Beta Inc in 2020."
        with pytest.raises(ValueError, match="Artifact validation failed"):
            run_pipeline_nlp(sample_text.encode("utf-8"), "test.txt", "txt")
    finally:
        settings.relation_backend = orig_backend
        settings.contextual_v2_allow_unverified = orig_override


def test_regression_calibration_rejects_positive_only_qualification(tmp_path):
    """Task 3: Calibration must require reviewed negative support and must not qualify precision 1.0 on positive-only data."""
    from scripts.calibrate_v2 import run_calibration

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    vocab = SimpleVocabulary()
    vocab.build_from_texts(["Acme", "Corp", "Beta", "Inc"])
    (model_dir / "vocab.json").write_text(json.dumps(vocab.to_dict()), encoding="utf-8")
    (model_dir / "training_manifest.json").write_text(
        json.dumps({"version": "2.0.0", "embed_dim": 16, "hidden_dim": 32, "num_layers": 1, "split": "training"}),
        encoding="utf-8",
    )
    model = ContextualRelationModel(vocab_size=len(vocab.token_to_idx), embed_dim=16, hidden_dim=32, num_layers=1)
    torch.save(model.state_dict(), model_dir / "model.pt")

    # 10 positive facts, ZERO reviewed negatives
    pos_only_docs = []
    for i in range(10):
        pos_only_docs.append({
            "schema_version": "2.dataset.2",
            "document_id": f"doc_{i}",
            "split": "calibration",
            "text": "Acme Corp founded Beta Inc.",
            "mentions": [
                {"id": f"m_{i}_1", "document_id": f"doc_{i}", "text": "Acme Corp", "type": "ORG", "start": 0, "end": 9},
                {"id": f"m_{i}_2", "document_id": f"doc_{i}", "text": "Beta Inc", "type": "ORG", "start": 18, "end": 26},
            ],
            "annotated_groups": {"0": [f"m_{i}_1"], "1": [f"m_{i}_2"]},
            "gold_relations": [["0", "FOUNDED", "1"]],
            "candidates": [
                {
                    "subject_group": "0",
                    "object_group": "1",
                    "type_supported": True,
                    "targets": [1] + [None] * 8,
                    "loss_mask": [True] + [False] * 8,
                }
            ],
        })

    cal_file = tmp_path / "pos_only_calibration.jsonl"
    cal_file.write_text("\n".join(json.dumps(d) for d in pos_only_docs) + "\n", encoding="utf-8")

    out_json = tmp_path / "pos_only_thresholds.json"
    result = run_calibration(
        model_dir=model_dir,
        calibration_file=cal_file,
        output_path=out_json,
        target_precision=0.70,
        min_support=5,
        min_negative_support=5,
    )

    founded_info = result["relations"]["FOUNDED"]
    # Must NOT claim calibrated at precision 1.0 without reviewed negatives!
    assert founded_info["status"] == "uncalibrated_insufficient_negatives"
    assert founded_info["disabled"] is True
    assert founded_info["threshold"] >= 1.0
    assert founded_info["abstention_policy"] == "abstain_insufficient_negatives"


def test_regression_assertion_support_awareness_and_honest_confidence():
    """Task 4: Unsupported assertion heads must emit 'unknown' status, and raw probabilities must not be stored as calibrated_confidence."""
    vocab = SimpleVocabulary()
    vocab.build_from_texts(["Acme", "Corp", "Beta", "Inc"])
    model = ContextualRelationModel(vocab_size=len(vocab.token_to_idx), embed_dim=16, hidden_dim=32, num_layers=1)

    # Extractor with assertion_support specifying that ACQUIRED has polarity support, but FOUNDED has NO assertion support!
    assertion_support = {
        "ACQUIRED": {"polarity": True, "modality": True, "lifecycle": True},
        "FOUNDED": {"polarity": False, "modality": False, "lifecycle": False},
    }

    extractor = ContextualRelationExtractor(
        model=model,
        vocab=vocab,
        thresholds={"FOUNDED": 0.0, "ACQUIRED": 0.0},
        assertion_support=assertion_support,
    )

    mentions = [
        MentionRecord(id="m1", document_id="doc1", text="Acme Corp", type="ORG", start=0, end=9),
        MentionRecord(id="m2", document_id="doc1", text="Beta Inc", type="ORG", start=18, end=26),
    ]
    claims = extractor.extract_from_document("doc1", "Acme Corp founded Beta Inc.", mentions)

    # Find FOUNDED claim
    founded_claim = next((c for c in claims if c.relation == "FOUNDED"), None)
    if founded_claim:
        # Unsupported assertion heads must remain 'unknown'!
        assert founded_claim.polarity == "unknown"
        assert founded_claim.modality == "unknown"
        assert founded_claim.lifecycle == "unknown"
        # Raw score must be preserved, and calibrated_confidence must be None!
        assert founded_claim.raw_score is not None
        assert founded_claim.calibrated_confidence is None


def test_regression_split_custody_and_group_disjointness(tmp_path):
    """Task 5: Split custody must reject protected test records (even if renamed),
    and reject cross-split document/group overlap."""
    from src.nlp.split_custody import verify_split_custody, verify_cross_split_disjointness

    # 1. Reject protected test record placed into a file named calibration.jsonl
    test_record = {
        "document_id": "test_leak_1",
        "split": "test",  # Protected!
        "group_key": "deal_group_a",
        "text": "Acme Corp founded Beta Inc.",
        "mentions": [],
        "candidates": [],
    }
    with pytest.raises(ValueError, match="Protected split custody violation"):
        verify_split_custody([test_record], expected_split="calibration")

    # 2. Reject cross-split document overlap
    doc_a = {"document_id": "shared_doc", "group_key": "grp1"}
    doc_b = {"document_id": "shared_doc", "group_key": "grp2"}
    with pytest.raises(ValueError, match="Cross-split document leakage"):
        verify_cross_split_disjointness({"train": [doc_a], "val": [doc_b]})

    # 3. Reject cross-split group overlap (prevent leakage of related documents)
    doc_c = {"document_id": "doc_1", "group_key": "same_deal_group"}
    doc_d = {"document_id": "doc_2", "group_key": "same_deal_group"}
    with pytest.raises(ValueError, match="Cross-split group leakage"):
        verify_cross_split_disjointness({"train": [doc_c], "val": [doc_d]})




