"""Predicate normalizer — maps raw dep-pattern predicates to canonical relation types.

Single-path normalization:
  verb lemma + optional prep context → canonical type from relations.yaml

Disambiguation:
  - "found" (past of "find") vs "found" (past of "found/establish"):
    check that the direct object is typed ORG before accepting FOUNDED.
  - "work" requires a qualifying preposition (for/at) to become MEMBER_OF.

Swap flag:
  lemma_swap=true means the dep-pattern subject is actually the object in the
  canonical relation (e.g., "Google hired Alice" → subject=Google, but the
  edge should read Alice EMPLOYED_BY Google → swap).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from src.nlp.ner import Entity
from src.nlp.relation_extractor import RawTriple
from src.type_map import NODE_TYPES


# ── Output type ───────────────────────────────────────────────────────────────

@dataclass
class NormalizedTriple:
    relation_type: str     # canonical name from whitelist, or "RELATED_TO"
    subject: Entity
    object: Entity
    raw_predicate: str     # original predicate for RELATED_TO triples
    chunk_id: str
    sent_idx: int
    confidence: float = 0.0    # filled in by relation_scorer


# ── Loader ────────────────────────────────────────────────────────────────────

def load_relations_config(path: str = "config/relations.yaml") -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("canonical_relations", {})


# ── Normalizer ────────────────────────────────────────────────────────────────

class PredicateNormalizer:
    def __init__(self, config_path: str = "config/relations.yaml"):
        self._config = load_relations_config(config_path)

    def normalize(self, triple: RawTriple) -> NormalizedTriple:
        for rel_name, conf in self._config.items():
            for entry in conf.get("lemmas", []):
                if triple.predicate != entry["lemma"]:
                    continue

                # Preposition context guard
                req_prep = entry.get("requires_prep")
                if req_prep and triple.prep not in req_prep:
                    continue

                # "found" disambiguation: reject if object is not ORG
                if triple.predicate == "found" and rel_name == "FOUNDED":
                    if triple.object.type != "ORG":
                        continue

                # Apply swap if needed first, then check type pair
                subj, obj = triple.subject, triple.object
                if entry.get("lemma_swap", False):
                    subj, obj = obj, subj

                # Entity type pair guard (checked AFTER swap)
                type_pairs = conf.get("entity_type_pairs", [])
                if type_pairs:
                    pair = [subj.type, obj.type]
                    if pair not in type_pairs:
                        continue

                return NormalizedTriple(
                    relation_type=rel_name,
                    subject=subj,
                    object=obj,
                    raw_predicate=triple.predicate,
                    chunk_id=triple.chunk_id,
                    sent_idx=triple.sent_idx,
                )

        # No match — fall to catch-all
        return NormalizedTriple(
            relation_type="RELATED_TO",
            subject=triple.subject,
            object=triple.object,
            raw_predicate=triple.predicate,
            chunk_id=triple.chunk_id,
            sent_idx=triple.sent_idx,
        )

    def normalize_batch(self, triples: list[RawTriple]) -> list[NormalizedTriple]:
        return [self.normalize(t) for t in triples]
