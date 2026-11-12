"""Classical English News relation extraction pipeline adapter.

Uses Stanford CRF NER and Roth Linear Relation Extractor with strict non-neural execution,
contract-conforming mention and claim records, and document-scoped entity identity.
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from src.graph.graph_builder import ResolvedEdge, ResolvedEntity
from src.graph.entity_resolver import compute_initials_key
from src.nlp.contracts_v2 import (
    CanonicalClaimRecord,
    MentionRecord,
    RelationClaimRecord,
)
from src.nlp.stanford_classical_client import (
    StanfordClassicalClient,
    get_shared_stanford_client,
)

logger = logging.getLogger(__name__)

ONTOLOGY_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "ontology_classical_news.yaml"

TYPE_MAP = {
    "PEOPLE": "PERSON",
    "PERSON": "PERSON",
    "ORGANIZATION": "ORGANIZATION",
    "LOCATION": "LOCATION",
}

RELATION_MAP = {
    "Work_For": "WORKS_FOR",
    "Live_In": "LIVES_IN",
    "OrgBased_In": "BASED_IN",
    "Located_In": "LOCATED_IN",
}

VALID_TYPE_PAIRS = {
    "WORKS_FOR": ("PERSON", "ORGANIZATION"),
    "LIVES_IN": ("PERSON", "LOCATION"),
    "BASED_IN": ("ORGANIZATION", "LOCATION"),
    "LOCATED_IN": ("LOCATION", "LOCATION"),
}


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return cleaned.upper() if cleaned else "UNKNOWN"


class ClassicalNewsExtractor:
    """Pretrained Classical ML relation extractor for English news and biography."""

    def __init__(self, client: Optional[StanfordClassicalClient] = None):
        self.client = client or get_shared_stanford_client()
        self.ontology = self._load_ontology()

    def _load_ontology(self) -> dict:
        if ONTOLOGY_PATH.exists():
            return yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))
        return {}

    def extract_document(
        self,
        document_id: str,
        text: str,
        source_map: Any = None,
        chunk_id_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process document through classical pipeline and return validated entities, claims, and edges."""
        cid_prefix = chunk_id_prefix or document_id

        # 1. Annotate text via persistent Stanford bridge
        annotation = self.client.annotate(text, req_id=document_id)

        # 2. Extract and normalize Named Entity Mentions
        mentions: List[MentionRecord] = []
        mention_by_id: Dict[str, MentionRecord] = {}

        for ent_raw in annotation.get("entities", []):
            raw_type = ent_raw.get("type", "OTHER")
            canon_type = TYPE_MAP.get(raw_type)
            if not canon_type:
                continue

            mid = f"{document_id}:{ent_raw['mention_id']}"
            m_text = ent_raw.get("mapped_text", "")
            start = ent_raw.get("char_start", 0)
            end = ent_raw.get("char_end", 0)

            if end <= start or not m_text.strip():
                continue

            # Check source map if provided
            src_start = start
            src_end = end
            if source_map and hasattr(source_map, "get_source_span"):
                span_res = source_map.get_source_span(start, end)
                if span_res:
                    src_start, src_end = span_res

            mention = MentionRecord(
                id=mid,
                document_id=document_id,
                text=m_text,
                type=canon_type,
                start=start,
                end=end,
                source_start=src_start,
                source_end=src_end,
                sentence_idx=ent_raw.get("sentence_index", 0),
                detector="stanford_crf_3class",
                score=1.0,
            )
            mentions.append(mention)
            mention_by_id[mid] = mention

        # 3. Document-scoped Entity Resolution
        # Group mentions within the document by (type, normalized_display_name)
        groups: Dict[Tuple[str, str], List[MentionRecord]] = {}
        for m in mentions:
            norm_key = (m.type, m.text.strip().lower())
            groups.setdefault(norm_key, []).append(m)

        resolved_entities: List[ResolvedEntity] = []
        mention_to_entity_id: Dict[str, str] = {}
        entity_to_chunks: Dict[str, List[str]] = {}

        for (etype, norm_name), m_list in groups.items():
            # Pick most frequent or longest surface form as primary name
            names = [m.text.strip() for m in m_list]
            primary_name = max(set(names), key=names.count)
            slug = slugify(primary_name)[:32]
            entity_id = f"E_{document_id}_{etype}_{slug}"

            aliases = sorted(list(set(names)))
            init_key = compute_initials_key(primary_name) if etype == "PERSON" else ""
            resolved_entity = ResolvedEntity(
                canonical_id=entity_id,
                name=primary_name,
                type=etype,
                aliases=aliases,
                initials_key=init_key,
            )
            resolved_entities.append(resolved_entity)

            for m in m_list:
                mention_to_entity_id[m.id] = entity_id

            entity_to_chunks[entity_id] = [f"{cid_prefix}:{m.sentence_idx or 0}"]

        # 4. Extract and validate Relation Claims
        claims: List[RelationClaimRecord] = []
        sentences = annotation.get("sentences", [])

        for rel_raw in annotation.get("relations", []):
            native_label = rel_raw.get("label", "_NR")
            graph_rel = RELATION_MAP.get(native_label)
            if not graph_rel:
                continue

            a0 = rel_raw.get("arg0", {})
            a1 = rel_raw.get("arg1", {})

            a0_mid = f"{document_id}:{a0.get('object_id', '')}"
            a1_mid = f"{document_id}:{a1.get('object_id', '')}"

            m0 = mention_by_id.get(a0_mid)
            m1 = mention_by_id.get(a1_mid)

            if not m0 or not m1:
                continue

            expected_s_type, expected_o_type = VALID_TYPE_PAIRS[graph_rel]

            # Check directionality and argument types
            if m0.type == expected_s_type and m1.type == expected_o_type:
                subj_m, obj_m = m0, m1
            elif m1.type == expected_s_type and m0.type == expected_o_type:
                # Flipped argument order in native model
                subj_m, obj_m = m1, m0
            else:
                # Type signature violation: reject edge rather than invent an invalid link
                continue

            # Reject self-links
            s_eid = mention_to_entity_id.get(subj_m.id)
            o_eid = mention_to_entity_id.get(obj_m.id)
            if s_eid == o_eid:
                continue

            probs = rel_raw.get("probabilities", {})
            raw_score = float(probs.get(native_label, 0.0))

            s_idx = rel_raw.get("sentence_index", 0)
            sent_obj = sentences[s_idx] if 0 <= s_idx < len(sentences) else {}
            sent_text = rel_raw.get("sentence_text") or sent_obj.get("text", "")
            sent_start = sent_obj.get("char_start", subj_m.start)
            sent_end = sent_obj.get("char_end", obj_m.end)

            claim_key = f"{document_id}:{subj_m.id}:{graph_rel}:{obj_m.id}"
            claim_id = f"claim_{hashlib.sha256(claim_key.encode('utf-8')).hexdigest()[:16]}"

            claim = RelationClaimRecord(
                claim_id=claim_id,
                document_id=document_id,
                subject_mention_id=subj_m.id,
                object_mention_id=obj_m.id,
                relation=graph_rel,
                polarity="unknown",
                modality="unknown",
                lifecycle="unknown",
                evidence_spans=[(sent_start, sent_end)],
                evidence_text=sent_text,
                model_version="stanford_corenlp_4.5.8_classical",
                raw_score=raw_score,
                calibrated_confidence=None,  # Honest confidence: uncalibrated score
            )
            claims.append(claim)

        # 5. In-Memory Aggregation into Canonical Claims & Resolved Edges
        # Group by (subject_entity_id, relation, object_entity_id)
        edge_groups: Dict[Tuple[str, str, str], List[RelationClaimRecord]] = {}
        for c in claims:
            s_eid = mention_to_entity_id[c.subject_mention_id]
            o_eid = mention_to_entity_id[c.object_mention_id]
            edge_groups.setdefault((s_eid, c.relation, o_eid), []).append(c)

        canonical_claims: List[CanonicalClaimRecord] = []
        resolved_edges: List[ResolvedEdge] = []

        for (s_eid, rel_type, o_eid), claim_list in edge_groups.items():
            member_ids = [c.claim_id for c in claim_list]
            all_spans = [s for c in claim_list for s in c.evidence_spans]
            evidence_texts = list({c.evidence_text for c in claim_list if c.evidence_text})
            combined_evidence = " ... ".join(evidence_texts)
            best_raw_score = max(c.raw_score for c in claim_list)

            can_claim_id = f"can_{hashlib.sha256(f'{document_id}:{s_eid}:{rel_type}:{o_eid}'.encode()).hexdigest()[:16]}"

            can_claim = CanonicalClaimRecord(
                canonical_claim_id=can_claim_id,
                canonical_subject_id=s_eid,
                canonical_object_id=o_eid,
                relation=rel_type,
                polarity="unknown",
                modality="unknown",
                lifecycle="unknown",
                time_qualifiers={},
                confidence=best_raw_score,
                member_claim_ids=member_ids,
                evidence_spans=all_spans,
                evidence_texts=evidence_texts,
                disagreements=[],
            )
            canonical_claims.append(can_claim)

            # Build chunk_ids covering the evidence sentences
            chunk_ids = list({f"{cid_prefix}:{mention_by_id[c.subject_mention_id].sentence_idx or 0}" for c in claim_list})

            edge = ResolvedEdge(
                relation_type=rel_type,
                source_id=s_eid,
                target_id=o_eid,
                chunk_ids=chunk_ids,
                extractors=["stanford_classical_roth"],
                confidence=best_raw_score,
                raw_predicate=rel_type.lower(),
                evidence=combined_evidence,
                polarity="unknown",
                modality="unknown",
                lifecycle="unknown",
                time_qualifiers={},
                claim_id=can_claim_id,
            )
            resolved_edges.append(edge)

        return {
            "document_id": document_id,
            "resolved_entities": resolved_entities,
            "mentions": mentions,
            "mention_to_entity_id": mention_to_entity_id,
            "entity_to_chunks": entity_to_chunks,
            "claims": claims,
            "canonical_claims": canonical_claims,
            "resolved_edges": resolved_edges,
            "sentences": sentences,
        }
