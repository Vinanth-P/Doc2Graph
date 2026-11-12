"""Non-destructive entity resolution adapter and clustering metrics for Milestone D.

Preserves mention identities and offsets. Produces inspectable soft clusters and identity
links without collapsing distinct mentions into single nodes before relation extraction.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Any, Optional

from src.graph.entity_resolver import (
    EntityResolver,
    compute_initials_key,
    _tokenize,
    _abbreviation_score,
    _is_person_name_conflict,
)
from src.nlp.contracts_v2 import MentionRecord, IdentityLinkRecord


def pairwise_merge_metrics(
    gold_clusters: Dict[str, List[str]],
    predicted_clusters: Dict[str, str],
) -> Dict[str, float]:
    """Calculate pairwise merge precision, recall, and F1 over mention pairs."""
    # Build gold pairs
    gold_pairs: Set[Tuple[str, str]] = set()
    all_mentions: Set[str] = set()
    for cluster_id, m_ids in gold_clusters.items():
        sorted_m = sorted(m_ids)
        for i in range(len(sorted_m)):
            all_mentions.add(sorted_m[i])
            for j in range(i + 1, len(sorted_m)):
                gold_pairs.add((sorted_m[i], sorted_m[j]))

    # Group mentions by predicted cluster
    pred_groups: Dict[str, List[str]] = defaultdict(list)
    for m_id, c_id in predicted_clusters.items():
        if m_id in all_mentions:
            pred_groups[c_id].append(m_id)

    pred_pairs: Set[Tuple[str, str]] = set()
    for c_id, m_ids in pred_groups.items():
        sorted_m = sorted(m_ids)
        for i in range(len(sorted_m)):
            for j in range(i + 1, len(sorted_m)):
                pred_pairs.add((sorted_m[i], sorted_m[j]))

    tp = len(gold_pairs & pred_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def bcubed_metrics(
    gold_clusters: Dict[str, List[str]],
    predicted_clusters: Dict[str, str],
) -> Dict[str, float]:
    """Calculate B-cubed cluster precision, recall, and F1."""
    # Map mention to its gold cluster members
    mention_to_gold: Dict[str, Set[str]] = {}
    for cluster_id, m_ids in gold_clusters.items():
        m_set = set(m_ids)
        for m_id in m_ids:
            mention_to_gold[m_id] = m_set

    # Map mention to its predicted cluster members
    pred_groups: Dict[str, Set[str]] = defaultdict(set)
    for m_id, c_id in predicted_clusters.items():
        pred_groups[c_id].add(m_id)

    mention_to_pred: Dict[str, Set[str]] = {
        m_id: pred_groups[c_id] for m_id, c_id in predicted_clusters.items()
    }

    eval_mentions = [m for m in mention_to_gold if m in mention_to_pred]
    if not eval_mentions:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    precisions = []
    recalls = []

    for m in eval_mentions:
        gold_set = mention_to_gold[m]
        pred_set = mention_to_pred[m]
        intersection = len(gold_set & pred_set)
        precisions.append(intersection / len(pred_set))
        recalls.append(intersection / len(gold_set))

    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {"precision": p, "recall": r, "f1": f1}


class NonDestructiveResolver:
    """Soft-clustering adapter preserving mention integrity."""

    def __init__(self, threshold: float = 0.85, base_threshold: Optional[float] = None, conservative_abbr: bool = True):
        self.threshold = base_threshold if base_threshold is not None else threshold
        self.conservative_abbr = conservative_abbr
        self._inner_resolver = EntityResolver(threshold=self.threshold)

    def resolve_within_doc(self, entities: List[Any]) -> Dict[str, str]:
        """Resolve entity clusters within document, accepting either Entities or MentionRecords."""
        if not entities:
            return {}
        if not isinstance(entities[0], MentionRecord):
            mentions = [
                MentionRecord(
                    id=e.entity_id,
                    document_id=getattr(e, "chunk_id", "doc"),
                    text=e.text,
                    type=e.type,
                    start=max(0, getattr(e, "start_char", 0)),
                    end=max(1, getattr(e, "end_char", 1)),
                    sentence_idx=getattr(e, "sent_idx", 0),
                    detector=getattr(e, "source", "ner"),
                )
                for e in entities
            ]
        else:
            mentions = entities
        mapping, _ = self.cluster_mentions(mentions)
        return mapping

    def cluster_mentions(
        self,
        mentions: List[MentionRecord],
    ) -> Tuple[Dict[str, str], List[IdentityLinkRecord]]:
        """Cluster mentions non-destructively.

        Returns:
            mapping: {mention_id: cluster_id}
            links: List of inspected IdentityLinkRecords
        """
        if not mentions:
            return {}, []

        # Convert MentionRecords to pseudo-entities for inner scoring logic
        from src.nlp.ner import Entity
        entities = [
            Entity(
                entity_id=m.id,
                text=m.text,
                canonical_name=m.text.strip(),
                type=m.type,
                start_char=m.start,
                end_char=m.end,
                source="non_destructive_adapter",
                chunk_id=m.document_id,
            )
            for m in mentions
        ]

        parent: Dict[str, str] = {m.id: m.id for m in mentions}
        links: List[IdentityLinkRecord] = []

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                # Prefer lexicographically smaller ID for determinism
                if ra < rb:
                    parent[rb] = ra
                else:
                    parent[ra] = rb

        # Compare all pairs within the same document sharing entity type
        by_type: Dict[str, List[Entity]] = defaultdict(list)
        for ent in entities:
            by_type[ent.type].append(ent)

        for etype, group in by_type.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    ea = group[i]
                    eb = group[j]

                    # 1. Exact normalized text match (case-insensitive)
                    if ea.canonical_name.casefold() == eb.canonical_name.casefold():
                        score = 1.0
                        decision = "linked"
                        union(ea.entity_id, eb.entity_id)
                        links.append(IdentityLinkRecord(
                            source_mention_id=ea.entity_id,
                            target_mention_id=eb.entity_id,
                            decision=decision,
                            score=score,
                            method="exact_normalized_match",
                            evidence=f"Exact match between '{ea.text}' and '{eb.text}'",
                        ))
                        continue

                    # 2. Strict Person Conflict Check
                    if ea.type == "PERSON" and eb.type == "PERSON":
                        toks_a = _tokenize(ea.canonical_name)
                        toks_b = _tokenize(eb.canonical_name)
                        if len(toks_a) >= 2 and len(toks_b) >= 2 and toks_a[-1] == toks_b[-1]:
                            first_a, first_b = toks_a[0], toks_b[0]
                            if first_a != first_b and len(first_a) > 1 and len(first_b) > 1:
                                links.append(IdentityLinkRecord(
                                    source_mention_id=ea.entity_id,
                                    target_mention_id=eb.entity_id,
                                    decision="unlinked",
                                    score=0.0,
                                    method="person_surname_conflict",
                                    evidence=f"Different given names ('{first_a}' vs '{first_b}') with shared surname '{toks_a[-1]}'",
                                ))
                                continue

                    # 3. Abbreviation / Acronym match
                    abbr = _abbreviation_score(ea.canonical_name, eb.canonical_name)
                    if abbr >= 0.9:
                        score = abbr
                        decision = "linked" if score >= self.threshold else "uncertain"
                        if decision == "linked":
                            union(ea.entity_id, eb.entity_id)
                        links.append(IdentityLinkRecord(
                            source_mention_id=ea.entity_id,
                            target_mention_id=eb.entity_id,
                            decision=decision,
                            score=score,
                            method="initials_acronym_match",
                            evidence=f"Initials/acronym match score {abbr:.2f}",
                        ))
                        continue

                    # 4. Scored match via model / feature vector
                    score = self._inner_resolver._score(ea, eb)
                    decision = "linked" if score >= self.threshold else "unlinked"
                    if decision == "linked":
                        union(ea.entity_id, eb.entity_id)
                    links.append(IdentityLinkRecord(
                        source_mention_id=ea.entity_id,
                        target_mention_id=eb.entity_id,
                        decision=decision,
                        score=score,
                        method="feature_vector_scoring",
                        evidence=f"Similarity score {score:.3f}",
                    ))

        mapping = {m.id: find(m.id) for m in mentions}
        return mapping, links
