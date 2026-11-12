"""Aggregation of mention-level relation claims into canonical claims for Milestone G.

Preserves distinct transactions/claims, tracks evidence spans, records assertion
disagreements (e.g. affirmed vs negated), and avoids naive independence assumptions
in confidence aggregation.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from src.nlp.contracts_v2 import (
    CanonicalClaimRecord,
    LifecycleType,
    ModalityType,
    PolarityType,
    RelationClaimRecord,
)


def aggregate_claims(
    mention_claims: List[RelationClaimRecord],
    entity_id_map: Dict[str, str],
) -> List[CanonicalClaimRecord]:
    """Aggregate mention-level claims into canonical entity claims.

    Args:
        mention_claims: List of RelationClaimRecord objects.
        entity_id_map: Mapping from mention_id/entity_id to canonical_entity_id.

    Returns:
        List of CanonicalClaimRecord objects with preserved evidence and disagreement audit.
    """
    # Blocker 6: Group by (subject, relation, object, deal_key) to preserve distinct deals/events
    grouped_claims: Dict[Tuple[str, str, str, str], List[RelationClaimRecord]] = defaultdict(list)

    for claim in mention_claims:
        can_subj = entity_id_map.get(claim.subject_mention_id, claim.subject_mention_id)
        can_obj = entity_id_map.get(claim.object_mention_id, claim.object_mention_id)

        # Integrity constraint: no canonical self-edges
        if can_subj == can_obj:
            continue

        deal_key = (
            claim.time_qualifiers.get("deal_id")
            or claim.time_qualifiers.get("year")
            or claim.time_qualifiers.get("date")
            or "default"
        )

        key = (can_subj, claim.relation, can_obj, str(deal_key))
        grouped_claims[key].append(claim)

    canonical_records: List[CanonicalClaimRecord] = []

    for (can_subj, relation, can_obj, deal_key), claims in sorted(grouped_claims.items()):
        member_ids = [c.claim_id for c in claims]

        seen_spans: Set[Tuple[int, int]] = set()
        unique_spans: List[Tuple[int, int]] = []
        unique_texts: List[str] = []

        for c in claims:
            for span in c.evidence_spans:
                span_tuple = (span[0], span[1])
                if span_tuple not in seen_spans:
                    seen_spans.add(span_tuple)
                    unique_spans.append(span_tuple)
            if c.evidence_text and c.evidence_text not in unique_texts:
                unique_texts.append(c.evidence_text)

        # Blocker 6: Do not treat calibrated_confidence=0.0 as falsy!
        # Do NOT apply arbitrary confidence bonus without empirical validation.
        conf_values = [
            c.calibrated_confidence if c.calibrated_confidence is not None else c.raw_score
            for c in claims
        ]
        aggregated_conf = max(conf_values) if conf_values else 0.0

        # Assertion consensus and disagreement checking
        polarities = {c.polarity for c in claims if c.polarity != "unknown"}
        modalities = {c.modality for c in claims if c.modality != "unknown"}
        lifecycles = {c.lifecycle for c in claims if c.lifecycle != "unknown"}

        disagreements: List[Dict[str, Any]] = []

        if len(polarities) > 1:
            canonical_polarity: PolarityType = "unknown"
            disagreements.append(
                {
                    "attribute": "polarity",
                    "values": sorted(list(polarities)),
                    "reason": "Conflicting polarity assertions across mentions",
                }
            )
        elif len(polarities) == 1:
            canonical_polarity = next(iter(polarities))
        else:
            canonical_polarity = "unknown"

        if len(modalities) > 1:
            canonical_modality: ModalityType = "unknown"
            disagreements.append(
                {
                    "attribute": "modality",
                    "values": sorted(list(modalities)),
                    "reason": "Conflicting modality assertions across mentions",
                }
            )
        elif len(modalities) == 1:
            canonical_modality = next(iter(modalities))
        else:
            canonical_modality = "unknown"

        if len(lifecycles) > 1:
            canonical_lifecycle: LifecycleType = "unknown"
            disagreements.append(
                {
                    "attribute": "lifecycle",
                    "values": sorted(list(lifecycles)),
                    "reason": "Conflicting lifecycle assertions across mentions",
                }
            )
        elif len(lifecycles) == 1:
            canonical_lifecycle = next(iter(lifecycles))
        else:
            canonical_lifecycle = "unknown"

        # Time qualifiers aggregation
        time_quals: Dict[str, Any] = {}
        for c in claims:
            time_quals.update(c.time_qualifiers)

        suffix = f"_{deal_key}" if deal_key != "default" else ""
        can_claim_id = f"can_claim_{can_subj}_{relation}_{can_obj}{suffix}"
        canonical_records.append(
            CanonicalClaimRecord(
                canonical_claim_id=can_claim_id,
                canonical_subject_id=can_subj,
                canonical_object_id=can_obj,
                relation=relation,
                polarity=canonical_polarity,
                modality=canonical_modality,
                lifecycle=canonical_lifecycle,
                time_qualifiers=time_quals,
                member_claim_ids=member_ids,
                evidence_spans=unique_spans,
                evidence_texts=unique_texts,
                confidence=round(float(aggregated_conf), 4),
                disagreements=disagreements,
                schema_version="2",
            )
        )

    return canonical_records
