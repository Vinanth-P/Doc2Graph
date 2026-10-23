"""Rule-based coreference resolver.

Three tiers (applied in order per sentence):
  1. Personal pronouns   → nearest PERSON entity in preceding context
  2. Organizational pronouns (it/its) → nearest ORG entity
  3. Definite NPs ("the company") → nearest ORG / PERSON heuristic
  4. Alias detection     → Jaro-Winkler similarity between surface forms

Output: a ClusterMap mapping canonical entity IDs to lists of coreferent spans.
The original text is NEVER rewritten. Triple construction uses this map to
substitute pronoun subjects/objects with their cluster-head entity ID.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz.distance import JaroWinkler

from src.ingestion.preprocessor import SentenceChunk
from src.nlp.ner import Entity


# ── Types ─────────────────────────────────────────────────────────────────────

ClusterMap = dict[str, list[str]]   # entity_id → [chunk_ids where it is mentioned]


# ── Constants ─────────────────────────────────────────────────────────────────

PRONOUNS_PERSON = frozenset({
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "they", "them", "their", "theirs", "themselves",
})
PRONOUNS_ORG = frozenset({"it", "its", "itself"})

DEF_NP_ORG = frozenset({
    "the company", "the firm", "the organization", "the group",
    "the bank", "the agency", "the institution", "the corporation",
    "the startup", "the venture", "the business",
})
DEF_NP_PERSON = frozenset({
    "the man", "the woman", "the person", "the official",
    "the executive", "the director", "the ceo", "the president",
    "the founder", "the chairman", "the chief",
})

ALIAS_THRESHOLD = 0.88   # Jaro-Winkler score above which two names are aliases


# ── Resolver ──────────────────────────────────────────────────────────────────

class RuleBasedCorefResolver:
    """Lightweight rule-based coreference for English business/news text."""

    def resolve(
        self,
        chunks: list[SentenceChunk],
        entities_by_chunk: dict[str, list[Entity]],
    ) -> dict[str, str]:
        """Return a pronoun/alias → canonical_entity_id mapping.

        Args:
            chunks: ordered sentence chunks.
            entities_by_chunk: chunk_id → list of NER entities in that chunk.

        Returns:
            A dict mapping chunk-local pronoun token text (lowered) to
            a canonical entity_id. The caller substitutes at triple time.
        """
        resolution: dict[str, str] = {}   # surface_in_chunk → entity_id
        recent_person: Optional[Entity] = None
        recent_org:    Optional[Entity] = None
        all_entities: list[Entity] = []

        for chunk in chunks:
            ents = entities_by_chunk.get(chunk.chunk_id, [])

            # Update recency pointers
            for ent in ents:
                if ent.type == "PERSON":
                    recent_person = ent
                elif ent.type == "ORG":
                    recent_org = ent
                all_entities.append(ent)

            # Tier 1 & 2: pronoun resolution by recency
            words = chunk.text.lower().split()
            for word in words:
                if word in PRONOUNS_PERSON and recent_person:
                    resolution[f"{chunk.chunk_id}:{word}"] = recent_person.entity_id
                elif word in PRONOUNS_ORG and recent_org:
                    resolution[f"{chunk.chunk_id}:{word}"] = recent_org.entity_id

            # Tier 3: definite NP resolution
            text_low = chunk.text.lower()
            for np in DEF_NP_ORG:
                if np in text_low and recent_org:
                    resolution[f"{chunk.chunk_id}:{np}"] = recent_org.entity_id
            for np in DEF_NP_PERSON:
                if np in text_low and recent_person:
                    resolution[f"{chunk.chunk_id}:{np}"] = recent_person.entity_id

            # Tier 4: alias detection — link short forms to the known full name
            for ent in ents:
                for known in all_entities:
                    if ent.entity_id == known.entity_id:
                        continue
                    if ent.type != known.type:
                        continue
                    if self._is_alias(ent.canonical_name, known.canonical_name):
                        # Map shorter alias to the longer canonical name's entity
                        if len(ent.canonical_name) < len(known.canonical_name):
                            resolution[ent.entity_id] = known.entity_id

        return resolution

    @staticmethod
    def _is_alias(a: str, b: str) -> bool:
        """True if one name is contained in the other or their JW similarity is high."""
        al, bl = a.lower(), b.lower()
        if al in bl or bl in al:
            return True
        return JaroWinkler.similarity(a, b) >= ALIAS_THRESHOLD
