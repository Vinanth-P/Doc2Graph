"""Tests for contextual relation modeling and masked loss in Milestone E."""
from __future__ import annotations

import pytest
import torch

from src.nlp.contextual_relations import (
    RELATION_TYPES,
    POLARITY_TYPES,
    MODALITY_TYPES,
    LIFECYCLE_TYPES,
    SimpleVocabulary,
    ContextualRelationModel,
    ContextualRelationExtractor,
    compute_masked_loss,
    generate_directional_candidates,
    build_marked_context,
)
from src.nlp.contracts_v2 import MentionRecord, RelationClaimRecord


def test_build_marked_context_offsets():
    text = "Microsoft acquired GitHub for $7.5B in 2018."
    s_span = (0, 9)     # Microsoft
    o_span = (19, 25)   # GitHub

    marked, (c_start, c_end), truncated = build_marked_context(text, s_span, o_span)
    assert not truncated
    assert "[SUBJ_START]" in marked
    assert "[SUBJ_END]" in marked
    assert "[OBJ_START]" in marked
    assert "[OBJ_END]" in marked
    assert "Microsoft" in marked
    assert "GitHub" in marked
    assert c_start <= s_span[0]
    assert c_end >= o_span[1]


def test_vocabulary_encoding():
    vocab = SimpleVocabulary()
    texts = [
        "Microsoft acquired GitHub in San Francisco.",
        "Satya Nadella is CEO of Microsoft.",
    ]
    vocab.build_from_texts(texts)

    marked_text = "[SUBJ_START] Microsoft [SUBJ_END] acquired [OBJ_START] GitHub [OBJ_END]"
    tokens = vocab.tokenize(marked_text)
    ids, s_s, s_e, o_s, o_e, truncated = vocab.encode(tokens, max_len=32)
    assert not truncated

    assert len(ids) == 32
    assert s_s != -1 and s_e != -1 and o_s != -1 and o_e != -1
    assert s_s < s_e < o_s < o_e


def test_masked_loss_zero_gradient_on_unknown():
    """Verify that unannotated/unknown relations (loss_mask=False) contribute exactly 0 gradient."""
    batch_size = 3
    num_relations = len(RELATION_TYPES)

    logits = torch.randn(batch_size, num_relations, requires_grad=True)
    targets = torch.zeros(batch_size, num_relations)
    loss_mask = torch.zeros(batch_size, num_relations, dtype=torch.bool)

    # Candidate 0: positive relation 0
    targets[0, 0] = 1.0
    loss_mask[0, 0] = True

    # Candidate 1: explicit negative relation 1
    targets[1, 1] = 0.0
    loss_mask[1, 1] = True

    # Candidate 2: completely unannotated (all loss_mask = False)
    # Target values at candidate 2 are dummy, but must contribute 0 loss and 0 gradient.

    loss, metrics = compute_masked_loss(logits, targets, loss_mask)
    loss.backward()

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert metrics["supervised_targets"] == 2.0

    # Logit gradients on candidate 2 (all unknown) must be strictly 0.0
    assert torch.all(logits.grad[2] == 0.0)

    # Logit gradient on candidate 0 relation 1 (unknown) must be strictly 0.0
    assert logits.grad[0, 1] == 0.0

    # Logit gradients on supervised positions must be non-zero
    assert logits.grad[0, 0] != 0.0
    assert logits.grad[1, 1] != 0.0


def test_masked_loss_all_masked_division_guard():
    """Verify zero-division guard when an entire batch contains only unknown labels."""
    logits = torch.randn(2, len(RELATION_TYPES), requires_grad=True)
    targets = torch.zeros(2, len(RELATION_TYPES))
    loss_mask = torch.zeros(2, len(RELATION_TYPES), dtype=torch.bool)

    loss, metrics = compute_masked_loss(logits, targets, loss_mask)
    assert float(loss.item()) == 0.0
    assert metrics["supervised_targets"] == 0.0

    loss.backward()
    assert torch.all(logits.grad == 0.0)


def test_directionality_asymmetry():
    """Swapping subject and object yields distinct candidate representations."""
    vocab = SimpleVocabulary()
    vocab.build_from_texts(["Acme Corporation acquired Beta Analytics."])

    model = ContextualRelationModel(vocab_size=len(vocab.token_to_idx), embed_dim=32, hidden_dim=64, num_layers=1)
    model.eval()

    # Forward candidate: Acme -> Beta
    toks_fwd = "[SUBJ_START] Acme Corporation [SUBJ_END] acquired [OBJ_START] Beta Analytics [OBJ_END]".split()
    ids_fwd, s_s, s_e, o_s, o_e, _ = vocab.encode(toks_fwd, max_len=32)

    # Reversed candidate: Beta -> Acme
    toks_rev = "[SUBJ_START] Beta Analytics [SUBJ_END] acquired [OBJ_START] Acme Corporation [OBJ_END]".split()
    ids_rev, s_s_r, s_e_r, o_s_r, o_e_r, _ = vocab.encode(toks_rev, max_len=32)

    inp = torch.tensor([ids_fwd, ids_rev], dtype=torch.long)
    pos = torch.tensor([[s_s, s_e, o_s, o_e], [s_s_r, s_e_r, o_s_r, o_e_r]], dtype=torch.long)

    with torch.no_grad():
        out = model(inp, pos)

    rep_fwd = out["pair_rep"][0]
    rep_rev = out["pair_rep"][1]
    # Representations and logits must differ due to directional marker pooling and subtraction
    assert not torch.allclose(rep_fwd, rep_rev)
    assert not torch.allclose(out["relation_logits"][0], out["relation_logits"][1])


def test_candidate_generation_with_ontology_types():
    text = "Satya Nadella is CEO of Microsoft. Microsoft acquired GitHub in San Francisco."
    mentions = [
        MentionRecord(id="m1", document_id="doc1", text="Satya Nadella", type="PERSON", start=0, end=13),
        MentionRecord(id="m2", document_id="doc1", text="Microsoft", type="ORG", start=24, end=33),
        MentionRecord(id="m3", document_id="doc1", text="GitHub", type="ORG", start=53, end=59),
        MentionRecord(id="m4", document_id="doc1", text="San Francisco", type="GPE", start=63, end=76),
    ]

    candidates = generate_directional_candidates("doc1", text, mentions)
    candidate_pairs = {(c.subject_mention_id, c.object_mention_id) for c in candidates}

    # Allowed by ontology: (PERSON, ORG), (ORG, ORG)
    assert ("m1", "m2") in candidate_pairs  # PERSON -> ORG (EMPLOYED_BY)
    assert ("m2", "m3") in candidate_pairs  # ORG -> ORG (ACQUIRED)
    assert ("m3", "m2") in candidate_pairs  # ORG -> ORG (reversed)

    # Pruned by ontology: GPE is neither subject nor object in target finance/general relations
    for c in candidates:
        assert c.subject_mention_id != "m4"
        assert c.object_mention_id != "m4"


def test_extractor_output_contract():
    vocab = SimpleVocabulary()
    vocab.build_from_texts(["Satya Nadella is CEO of Microsoft."])

    model = ContextualRelationModel(vocab_size=len(vocab.token_to_idx), embed_dim=32, hidden_dim=64, num_layers=1)
    extractor = ContextualRelationExtractor(model, vocab, thresholds={r: 0.0 for r in RELATION_TYPES})

    text = "Satya Nadella is CEO of Microsoft."
    mentions = [
        MentionRecord(id="m1", document_id="doc1", text="Satya Nadella", type="PERSON", start=0, end=13),
        MentionRecord(id="m2", document_id="doc1", text="Microsoft", type="ORG", start=24, end=33),
    ]

    claims = extractor.extract_from_document("doc1", text, mentions)
    assert len(claims) > 0
    for claim in claims:
        assert isinstance(claim, RelationClaimRecord)
        assert claim.relation in RELATION_TYPES
        assert claim.polarity in POLARITY_TYPES
        assert claim.modality in MODALITY_TYPES
        assert claim.lifecycle in LIFECYCLE_TYPES
        assert len(claim.evidence_spans) == 1
        s, e = claim.evidence_spans[0]
        assert 0 <= s < e <= len(text)
        assert claim.raw_score >= 0.0
