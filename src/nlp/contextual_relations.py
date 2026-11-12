"""Contextual relation extraction and assertion modeling for Milestone E.

Implements directional ordered-pair candidate generation, argument-preserving contextual
encoding, sequence padding masks, per-relation assertion heads (polarity, modality, lifecycle),
and strict masked loss that never trains unknown labels as supervised negative zeros.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nlp.contracts_v2 import (
    LifecycleType,
    MentionRecord,
    ModalityType,
    PolarityType,
    RelationCandidateRecord,
    RelationClaimRecord,
)

RELATION_TYPES: List[str] = [
    "FOUNDED",
    "EMPLOYED_BY",
    "MEMBER_OF",
    "OWNS",
    "SUBSIDIARY_OF",
    "ACQUIRED",
    "INVESTED_IN",
    "ADVISED",
    "FINANCED",
]

POLARITY_TYPES: List[PolarityType] = ["affirmed", "negated", "unknown"]
MODALITY_TYPES: List[ModalityType] = ["asserted", "planned", "conditional", "rumored", "unknown"]
LIFECYCLE_TYPES: List[LifecycleType] = ["announced", "signed", "completed", "cancelled", "unknown"]

RELATION_TO_IDX: Dict[str, int] = {r: i for i, r in enumerate(RELATION_TYPES)}
IDX_TO_RELATION: Dict[int, str] = {i: r for i, r in enumerate(RELATION_TYPES)}

POLARITY_TO_IDX: Dict[str, int] = {p: i for i, p in enumerate(POLARITY_TYPES)}
MODALITY_TO_IDX: Dict[str, int] = {m: i for i, m in enumerate(MODALITY_TYPES)}
LIFECYCLE_TO_IDX: Dict[str, int] = {l: i for i, l in enumerate(LIFECYCLE_TYPES)}

SUBJ_START_TOKEN = "[SUBJ_START]"
SUBJ_END_TOKEN = "[SUBJ_END]"
OBJ_START_TOKEN = "[OBJ_START]"
OBJ_END_TOKEN = "[OBJ_END]"
SPECIAL_TOKENS = [SUBJ_START_TOKEN, SUBJ_END_TOKEN, OBJ_START_TOKEN, OBJ_END_TOKEN]


def load_ontology_type_pairs(ontology_path: Optional[Path] = None) -> Dict[str, List[Tuple[str, str]]]:
    """Load allowed entity type pairs per relation from ontology_v2.yaml."""
    if ontology_path is None:
        ontology_path = Path(__file__).resolve().parents[2] / "config" / "ontology_v2.yaml"
    import yaml

    data = yaml.safe_load(ontology_path.read_text(encoding="utf-8"))
    type_pairs: Dict[str, List[Tuple[str, str]]] = {}
    for rel_name, rel_cfg in data.get("relations", {}).items():
        pairs = [tuple(p) for p in rel_cfg.get("entity_type_pairs", [])]
        type_pairs[rel_name] = pairs
    return type_pairs


class SimpleVocabulary:
    """Lightweight reproducible vocabulary for offline contextual neural encoding."""

    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    CLS_TOKEN = "[CLS]"

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}
        self._frozen = False

        for tok in [self.PAD_TOKEN, self.UNK_TOKEN, self.CLS_TOKEN] + SPECIAL_TOKENS:
            self._add_token(tok)

    def _add_token(self, token: str) -> int:
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
            return idx
        return self.token_to_idx[token]

    def build_from_texts(self, texts: List[str], max_vocab_size: int = 20000) -> None:
        from collections import Counter

        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(text.lower().split())

        for word, count in counts.most_common(max_vocab_size):
            if count >= self.min_freq:
                self._add_token(word)
        self._frozen = True

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def encode(self, tokens: List[str], max_len: int = 128) -> Tuple[List[int], int, int, int, int, bool]:
        """Convert tokens to ids and return positions of subject and object markers.

        Returns:
            (input_ids, s_start, s_end, o_start, o_end, truncated)
        """
        truncated = len(tokens) > (max_len - 1)
        sub_tokens = tokens[: max_len - 1]

        ids = [self.token_to_idx[self.CLS_TOKEN]]
        s_start, s_end, o_start, o_end = -1, -1, -1, -1

        for tok in sub_tokens:
            curr_pos = len(ids)
            if tok == SUBJ_START_TOKEN:
                s_start = curr_pos
            elif tok == SUBJ_END_TOKEN:
                s_end = curr_pos
            elif tok == OBJ_START_TOKEN:
                o_start = curr_pos
            elif tok == OBJ_END_TOKEN:
                o_end = curr_pos

            tid = self.token_to_idx.get(tok, self.token_to_idx.get(tok.lower(), self.token_to_idx[self.UNK_TOKEN]))
            ids.append(tid)

        pad_id = self.token_to_idx[self.PAD_TOKEN]
        while len(ids) < max_len:
            ids.append(pad_id)

        # If any marker token was truncated past max_len, flag incomplete arguments
        if (
            SUBJ_START_TOKEN in tokens and s_start == -1
            or SUBJ_END_TOKEN in tokens and s_end == -1
            or OBJ_START_TOKEN in tokens and o_start == -1
            or OBJ_END_TOKEN in tokens and o_end == -1
        ):
            truncated = True

        return ids[:max_len], s_start, s_end, o_start, o_end, truncated

    def to_dict(self) -> Dict[str, Any]:
        return {"token_to_idx": self.token_to_idx}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SimpleVocabulary:
        vocab = cls()
        vocab.token_to_idx = dict(data["token_to_idx"])
        vocab.idx_to_token = {int(v): k for k, v in vocab.token_to_idx.items()}
        vocab._frozen = True
        return vocab


def build_marked_context(
    text: str,
    subject_span: Tuple[int, int],
    object_span: Tuple[int, int],
    window_padding: int = 120,
    max_char_span: int = 1200,
) -> Tuple[str, Tuple[int, int], bool]:
    """Build marked context window string inserting entity markers at exact character boundaries.

    Returns:
        marked_text: context with [SUBJ_START] ... [SUBJ_END] and [OBJ_START] ... [OBJ_END].
        context_span: (context_start_char, context_end_char) in the original document text.
        truncated: True if arguments could not both fit within max_char_span.
    """
    s_start, s_end = subject_span
    o_start, o_end = object_span

    # Check if distance between mentions exceeds allowable window
    arg_span_len = max(s_end, o_end) - min(s_start, o_start)
    truncated = arg_span_len > max_char_span

    start_char = max(0, min(s_start, o_start) - window_padding)
    end_char = min(len(text), max(s_end, o_end) + window_padding)

    while start_char > 0 and not text[start_char].isspace():
        start_char -= 1
    while end_char < len(text) and not text[end_char].isspace():
        end_char += 1

    cuts: List[Tuple[int, str]] = []
    cuts.append((s_start, f" {SUBJ_START_TOKEN} "))
    cuts.append((s_end, f" {SUBJ_END_TOKEN} "))
    cuts.append((o_start, f" {OBJ_START_TOKEN} "))
    cuts.append((o_end, f" {OBJ_END_TOKEN} "))
    cuts.sort(key=lambda x: x[0])

    pieces = []
    curr = start_char
    for cut_pos, tag in cuts:
        if cut_pos > curr:
            pieces.append(text[curr:cut_pos])
        pieces.append(tag)
        curr = cut_pos
    if curr < end_char:
        pieces.append(text[curr:end_char])

    marked_text = " ".join("".join(pieces).split())
    return marked_text, (start_char, end_char), truncated


def generate_directional_candidates(
    document_id: str,
    text: str,
    mentions: List[MentionRecord],
    ontology_type_pairs: Optional[Dict[str, List[Tuple[str, str]]]] = None,
    max_char_distance: int = 1200,
) -> List[RelationCandidateRecord]:
    """Generate ordered candidate pairs with typed ontology filtering without hand-written relation verbs."""
    if ontology_type_pairs is None:
        ontology_type_pairs = load_ontology_type_pairs()

    all_allowed_pairs: Set[Tuple[str, str]] = set()
    for pairs in ontology_type_pairs.values():
        all_allowed_pairs.update(pairs)

    candidates: List[RelationCandidateRecord] = []
    num_mentions = len(mentions)

    for i in range(num_mentions):
        for j in range(num_mentions):
            if i == j:
                continue
            m_subj = mentions[i]
            m_obj = mentions[j]

            if (m_subj.type, m_obj.type) not in all_allowed_pairs:
                continue

            dist = abs(m_subj.start - m_obj.start)
            if dist > max_char_distance:
                continue

            marked_text, context_span, is_truncated = build_marked_context(
                text, (m_subj.start, m_subj.end), (m_obj.start, m_obj.end)
            )

            candidate_id = f"{document_id}:{m_subj.id}:{m_obj.id}"
            candidates.append(
                RelationCandidateRecord(
                    candidate_id=candidate_id,
                    document_id=document_id,
                    subject_mention_id=m_subj.id,
                    object_mention_id=m_obj.id,
                    context_ids=[f"span:{context_span[0]}-{context_span[1]}"],
                    context_text=marked_text,
                    retrieval_reason="contextual_cooccurrence",
                    truncated=is_truncated,
                )
            )
    return candidates


class ContextualRelationModel(nn.Module):
    """Contextual neural classifier with sequence padding masks and per-relation assertion heads."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(512, embed_dim)
        self.encoder = nn.GRU(
            embed_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        pair_dim = hidden_dim * 5

        self.relation_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(RELATION_TYPES)),
        )

        # Relation-conditioned assertion heads: each relation has independent assertion projections
        self.polarity_heads = nn.ModuleDict({
            r: nn.Sequential(nn.Linear(pair_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, len(POLARITY_TYPES)))
            for r in RELATION_TYPES
        })
        self.modality_heads = nn.ModuleDict({
            r: nn.Sequential(nn.Linear(pair_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, len(MODALITY_TYPES)))
            for r in RELATION_TYPES
        })
        self.lifecycle_heads = nn.ModuleDict({
            r: nn.Sequential(nn.Linear(pair_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, len(LIFECYCLE_TYPES)))
            for r in RELATION_TYPES
        })

    def load_state_dict(self, state_dict: Any, strict: bool = True, assign: bool = False):
        """Support loading legacy single-assertion head checkpoints by replicating weights to per-relation heads."""
        adapted = dict(state_dict)
        if "polarity_head.0.weight" in adapted and not any(k.startswith("polarity_heads.") for k in adapted):
            for r in RELATION_TYPES:
                adapted[f"polarity_heads.{r}.0.weight"] = adapted["polarity_head.0.weight"].clone()
                adapted[f"polarity_heads.{r}.0.bias"] = adapted["polarity_head.0.bias"].clone()
                adapted[f"polarity_heads.{r}.2.weight"] = adapted["polarity_head.2.weight"].clone()
                adapted[f"polarity_heads.{r}.2.bias"] = adapted["polarity_head.2.bias"].clone()

                adapted[f"modality_heads.{r}.0.weight"] = adapted["modality_head.0.weight"].clone()
                adapted[f"modality_heads.{r}.0.bias"] = adapted["modality_head.0.bias"].clone()
                adapted[f"modality_heads.{r}.2.weight"] = adapted["modality_head.2.weight"].clone()
                adapted[f"modality_heads.{r}.2.bias"] = adapted["modality_head.2.bias"].clone()

                adapted[f"lifecycle_heads.{r}.0.weight"] = adapted["lifecycle_head.0.weight"].clone()
                adapted[f"lifecycle_heads.{r}.0.bias"] = adapted["lifecycle_head.0.bias"].clone()
                adapted[f"lifecycle_heads.{r}.2.weight"] = adapted["lifecycle_head.2.weight"].clone()
                adapted[f"lifecycle_heads.{r}.2.bias"] = adapted["lifecycle_head.2.bias"].clone()

            for old_k in list(adapted.keys()):
                if old_k.startswith(("polarity_head.", "modality_head.", "lifecycle_head.")):
                    del adapted[old_k]

        return super().load_state_dict(adapted, strict=strict, assign=assign)

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, L]
        marker_positions: torch.Tensor,  # [B, 4] -> [s_start, s_end, o_start, o_end]
    ) -> Dict[str, Any]:
        batch_size, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(input_ids) + self.pos_embedding(pos)
        x = self.dropout(x)

        # Padding mask: compute actual lengths to prevent GRU from processing padded positions
        lengths = (input_ids != self.pad_idx).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)

        h_cls = hidden[:, 0, :]

        h_subj_list = []
        h_obj_list = []
        valid_markers = []

        for b in range(batch_size):
            s_s, s_e, o_s, o_e = marker_positions[b].tolist()
            has_s = (0 <= s_s < seq_len) and (0 <= s_e < seq_len) and (s_s <= s_e)
            has_o = (0 <= o_s < seq_len) and (0 <= o_e < seq_len) and (o_s <= o_e)

            if has_s:
                h_s = hidden[b, s_s : s_e + 1].mean(dim=0)
            else:
                h_s = torch.zeros_like(h_cls[b])

            if has_o:
                h_o = hidden[b, o_s : o_e + 1].mean(dim=0)
            else:
                h_o = torch.zeros_like(h_cls[b])

            valid_markers.append(has_s and has_o)
            h_subj_list.append(h_s)
            h_obj_list.append(h_o)

        h_subj = torch.stack(h_subj_list, dim=0)
        h_obj = torch.stack(h_obj_list, dim=0)

        h_diff = h_subj - h_obj
        h_prod = h_subj * h_obj
        pair_rep = torch.cat([h_subj, h_obj, h_diff, h_prod, h_cls], dim=-1)

        rel_logits = self.relation_head(pair_rep)

        # Relation-conditioned assertions
        pol_logits_by_rel = {r: self.polarity_heads[r](pair_rep) for r in RELATION_TYPES}
        mod_logits_by_rel = {r: self.modality_heads[r](pair_rep) for r in RELATION_TYPES}
        life_logits_by_rel = {r: self.lifecycle_heads[r](pair_rep) for r in RELATION_TYPES}

        return {
            "relation_logits": rel_logits,
            "polarity_logits_by_rel": pol_logits_by_rel,
            "modality_logits_by_rel": mod_logits_by_rel,
            "lifecycle_logits_by_rel": life_logits_by_rel,
            "pair_rep": pair_rep,
            "valid_markers": torch.tensor(valid_markers, dtype=torch.bool, device=input_ids.device),
        }


def compute_masked_loss(
    relation_logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    assertion_logits_by_rel: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    assertion_targets_by_rel: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    assertion_masks_by_rel: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    assertion_weight: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute binary cross entropy loss strictly where loss_mask is True."""
    bce = F.binary_cross_entropy_with_logits(relation_logits, targets.float(), reduction="none")
    mask_float = loss_mask.float()
    masked_bce = (bce * mask_float).sum()
    valid_elements = mask_float.sum().clamp(min=1.0)
    rel_loss = masked_bce / valid_elements

    metrics = {"rel_loss": float(rel_loss.item()), "supervised_targets": float(mask_float.sum().item())}
    total_loss = rel_loss

    if assertion_logits_by_rel and assertion_targets_by_rel and assertion_masks_by_rel:
        for r_name in RELATION_TYPES:
            r_logits = assertion_logits_by_rel.get(r_name, {})
            r_tgts = assertion_targets_by_rel.get(r_name, {})
            r_masks = assertion_masks_by_rel.get(r_name, {})

            for attr in ["polarity", "modality", "lifecycle"]:
                l = r_logits.get(attr)
                t = r_tgts.get(attr)
                m = r_masks.get(attr)
                if l is not None and t is not None and m is not None and m.any():
                    ce = F.cross_entropy(l, t, reduction="none")
                    m_f = m.float()
                    denom = m_f.sum().clamp(min=1.0)
                    a_loss = (ce * m_f).sum() / denom
                    total_loss = total_loss + assertion_weight * a_loss
                    metrics[f"{r_name}_{attr}_loss"] = float(a_loss.item())

    metrics["total_loss"] = float(total_loss.item())
    return total_loss, metrics


class ContextualRelationExtractor:
    """Inference extractor producing immutable RelationClaimRecords with provenance."""

    def __init__(
        self,
        model: ContextualRelationModel,
        vocab: SimpleVocabulary,
        thresholds: Optional[Dict[str, float]] = None,
        disabled_relations: Optional[Set[str]] = None,
        assertion_support: Optional[Dict[str, Dict[str, bool]]] = None,
        device: str = "cpu",
        model_version: str = "contextual_v2.0",
    ):
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.device = device
        self.model_version = model_version
        self.ontology_type_pairs = load_ontology_type_pairs()
        self.disabled_relations = set(disabled_relations or [])
        self.assertion_support = assertion_support or {}

        self.thresholds = {r: 0.5 for r in RELATION_TYPES}
        if thresholds:
            self.thresholds.update(thresholds)

    def extract_from_document(
        self,
        document_id: str,
        text: str,
        mentions: List[MentionRecord],
        max_tokens: int = 128,
    ) -> List[RelationClaimRecord]:
        """Run candidate generation and classification on document mentions."""
        candidates = generate_directional_candidates(
            document_id=document_id,
            text=text,
            mentions=mentions,
            ontology_type_pairs=self.ontology_type_pairs,
        )
        if not candidates:
            return []

        mentions_by_id = {m.id: m for m in mentions}
        claims: List[RelationClaimRecord] = []

        valid_cands: List[RelationCandidateRecord] = []
        batch_ids = []
        batch_markers = []

        for cand in candidates:
            toks = self.vocab.tokenize(cand.context_text)
            ids, s_s, s_e, o_s, o_e, is_truncated = self.vocab.encode(toks, max_len=max_tokens)

            # Blocker 4: Abstain if argument markers were truncated or missing
            if s_s == -1 or s_e == -1 or o_s == -1 or o_e == -1:
                continue

            valid_cands.append(cand)
            batch_ids.append(ids)
            batch_markers.append([s_s, s_e, o_s, o_e])

        if not valid_cands:
            return []

        input_tensor = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
        markers_tensor = torch.tensor(batch_markers, dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor, markers_tensor)
            probs = torch.sigmoid(outputs["relation_logits"]).cpu().numpy()

            pol_preds = {
                r: outputs["polarity_logits_by_rel"][r].argmax(dim=-1).cpu().numpy()
                for r in RELATION_TYPES
            }
            mod_preds = {
                r: outputs["modality_logits_by_rel"][r].argmax(dim=-1).cpu().numpy()
                for r in RELATION_TYPES
            }
            life_preds = {
                r: outputs["lifecycle_logits_by_rel"][r].argmax(dim=-1).cpu().numpy()
                for r in RELATION_TYPES
            }

        for idx, cand in enumerate(valid_cands):
            m_subj = mentions_by_id[cand.subject_mention_id]
            m_obj = mentions_by_id[cand.object_mention_id]

            for rel_idx, rel_name in enumerate(RELATION_TYPES):
                # Blocker 2: Abstain from disabled / uncalibrated relations
                if rel_name in self.disabled_relations:
                    continue

                prob = float(probs[idx, rel_idx])
                thresh = self.thresholds.get(rel_name, 0.5)
                if prob >= thresh:
                    has_pol = self.assertion_support.get(rel_name, {}).get("polarity", False)
                    has_mod = self.assertion_support.get(rel_name, {}).get("modality", False)
                    has_life = self.assertion_support.get(rel_name, {}).get("lifecycle", False)

                    pol = POLARITY_TYPES[pol_preds[rel_name][idx]] if has_pol else "unknown"
                    mod = MODALITY_TYPES[mod_preds[rel_name][idx]] if has_mod else "unknown"
                    life = LIFECYCLE_TYPES[life_preds[rel_name][idx]] if has_life else "unknown"

                    ev_start = min(m_subj.start, m_obj.start)
                    ev_end = max(m_subj.end, m_obj.end)
                    ev_text = text[ev_start:ev_end]

                    claim = RelationClaimRecord(
                        claim_id=f"claim_{document_id}_{cand.subject_mention_id}_{rel_name}_{cand.object_mention_id}",
                        document_id=document_id,
                        subject_mention_id=cand.subject_mention_id,
                        object_mention_id=cand.object_mention_id,
                        relation=rel_name,
                        polarity=pol,
                        modality=mod,
                        lifecycle=life,
                        time_qualifiers={},
                        evidence_spans=[(ev_start, ev_end)],
                        evidence_text=ev_text,
                        model_version=self.model_version,
                        raw_score=prob,
                        calibrated_confidence=None,  # Honest confidence: unset without probability calibrator
                        abstention_reason=None,
                    )
                    claims.append(claim)

        return claims
