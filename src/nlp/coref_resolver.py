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
import re

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
    "the startup", "the venture", "the business", "the institute",
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

            # Process mentions in reading order. A later object must not become
            # the antecedent of an earlier pronoun ("He hired Alice").
            phrases = PRONOUNS_PERSON | PRONOUNS_ORG | DEF_NP_PERSON | DEF_NP_ORG
            pattern = r"\b(?:" + "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True)) + r")\b"
            events = [(e.end_char, 0, e) for e in ents]
            events += [
                (m.start(), 1, m) for m in re.finditer(pattern, chunk.text, re.IGNORECASE)
                if not any(e.start_char <= m.start() < e.end_char for e in ents)
            ]
            for _, kind, mention in sorted(events, key=lambda event: (event[0], event[1])):
                if kind == 0:
                    if mention.type == "PERSON":
                        recent_person = mention
                    elif mention.type == "ORG":
                        recent_org = mention
                    continue
                phrase = mention.group().lower()
                antecedent = recent_person if phrase in PRONOUNS_PERSON | DEF_NP_PERSON else recent_org
                # Position keys distinguish repeated pronouns in one sentence.
                resolution[f"{chunk.chunk_id}:ref:{mention.start()}"] = antecedent.entity_id if antecedent else ""
                if antecedent:
                    resolution.setdefault(f"{chunk.chunk_id}:{phrase}", antecedent.entity_id)

            all_entities.extend(ents)

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
