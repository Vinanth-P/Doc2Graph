"""Dependency-pattern relation extractor.

Extracts (subject, predicate, object) triples from spaCy dependency parses
using four pattern types:
  - SVO:        subject–verb–object
  - Appositive: token with dep_="appos"
  - Prepositional: verb/noun + prep + pobj
  - Nominal:    role-noun ("founder", "CEO") with nmod/poss child
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import spacy
from spacy.tokens import Doc, Token, Span

from src.nlp.ner import Entity


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RawTriple:
    """An unscored, unresolved (subject, predicate, object) triple."""
    subject: Entity
    predicate: str          # verb lemma or pattern key (e.g. "join", "appos", "found")
    object: Entity
    prep: Optional[str]     # preposition token if applicable, else None
    source: str             # "svo" | "appos" | "prep" | "nominal"
    sent_idx: int           # sentence index in the document
    chunk_id: str           # center chunk ID (for evidence attribution)

    # The dep-tree distance between subject and object tokens
    dep_distance: int = 0
    # Whether a negation is in the governor's subtree
    has_negation: bool = False


# ── Helpers ───────────────────────────────────────────────────────────────────

ROLE_NOUNS = frozenset({
    "founder", "ceo", "director", "chairman", "president",
    "head", "chief", "owner", "partner", "investor",
    "manager", "officer", "executive",
})


def _build_span_map(entities: list[Entity]) -> dict[tuple[int, int], Entity]:
    """Map (start_char, end_char) → Entity for fast lookup."""
    return {(e.start_char, e.end_char): e for e in entities}


def _token_to_entity(
    token: Token,
    span_map: dict[tuple[int, int], Entity],
) -> Optional[Entity]:
    """Try to find an NER entity covering this token (or its head noun phrase)."""
    # Try exact token span
    key = (token.idx, token.idx + len(token.text))
    if key in span_map:
        return span_map[key]
    # Try the full noun chunk containing this token
    for chunk in token.doc.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            ck = (chunk.start_char, chunk.end_char)
            if ck in span_map:
                return span_map[ck]
    return None


def _dep_distance(a: Token, b: Token) -> int:
    """Approximate dependency-tree distance between two tokens."""
    a_ancestors = {t.i for t in a.ancestors}
    b_ancestors = {t.i for t in b.ancestors}
    common = a_ancestors & b_ancestors
    if not common:
        return 999
    lca = min(common, key=lambda i: a.doc[i].dep_)
    return (
        len([t for t in a.ancestors if t.i not in b_ancestors]) +
        len([t for t in b.ancestors if t.i not in a_ancestors])
    )


def _has_negation(token: Token) -> bool:
    return any(c.dep_ == "neg" for c in token.subtree)


# ── Extractor ─────────────────────────────────────────────────────────────────

class DepPatternExtractor:
    """Extract relation triples from spaCy dependency parses."""

    def extract(
        self,
        doc: Doc,
        entities: list[Entity],
        center_chunk_id: str,
        center_sent_idx: int,
    ) -> list[RawTriple]:
        span_map = _build_span_map(entities)
        triples: list[RawTriple] = []
        for sent in doc.sents:
            triples += self._svo(sent, span_map, center_chunk_id, center_sent_idx)
            triples += self._appositive(sent, span_map, center_chunk_id, center_sent_idx)
            triples += self._prepositional(sent, span_map, center_chunk_id, center_sent_idx)
            triples += self._nominal(sent, span_map, center_chunk_id, center_sent_idx)
        return triples

    # ── Pattern implementations ───────────────────────────────────────────────

    def _svo(self, sent: Span, span_map, chunk_id, sent_idx) -> list[RawTriple]:
        """SVO: 'Musk founded Tesla', 'Google hired Alice'."""
        triples = []
        for token in sent:
            if token.pos_ != "VERB":
                continue
            # Find subject
            subj_tok = next(
                (c for c in token.children if c.dep_ in {"nsubj", "nsubjpass"}),
                None,
            )
            # Find object
            obj_tok = next(
                (c for c in token.children if c.dep_ in {"dobj", "attr", "oprd"}),
                None,
            )
            if subj_tok is None or obj_tok is None:
                continue
            subj_ent = _token_to_entity(subj_tok, span_map)
            obj_ent  = _token_to_entity(obj_tok,  span_map)
            if subj_ent is None or obj_ent is None:
                continue
            prep = next(
                (c.text for c in token.children if c.dep_ == "prep"), None
            )
            triples.append(RawTriple(
                subject=subj_ent,
                predicate=token.lemma_,
                object=obj_ent,
                prep=prep,
                source="svo",
                sent_idx=sent_idx,
                chunk_id=chunk_id,
                dep_distance=_dep_distance(subj_tok, obj_tok),
                has_negation=_has_negation(token),
            ))
        return triples

    def _appositive(self, sent: Span, span_map, chunk_id, sent_idx) -> list[RawTriple]:
        """Appositive: 'Elon Musk, CEO of Tesla' → (Musk, appos, Tesla)."""
        triples = []
        for token in sent:
            if token.dep_ != "appos":
                continue
            head_ent  = _token_to_entity(token.head, span_map)
            appos_ent = _token_to_entity(token, span_map)
            if head_ent and appos_ent:
                triples.append(RawTriple(
                    subject=head_ent,
                    predicate="appos",
                    object=appos_ent,
                    prep=None,
                    source="appos",
                    sent_idx=sent_idx,
                    chunk_id=chunk_id,
                    dep_distance=1,
                ))
        return triples

    def _prepositional(self, sent: Span, span_map, chunk_id, sent_idx) -> list[RawTriple]:
        """Prepositional: 'director of Tesla', 'based in California'."""
        triples = []
        for token in sent:
            if token.dep_ != "prep":
                continue
            pobj = next((c for c in token.children if c.dep_ == "pobj"), None)
            if pobj is None:
                continue
            head_ent = _token_to_entity(token.head, span_map)
            obj_ent  = _token_to_entity(pobj, span_map)
            if head_ent and obj_ent:
                predicate = f"{token.head.lemma_}_{token.lemma_}"
                triples.append(RawTriple(
                    subject=head_ent,
                    predicate=predicate,
                    object=obj_ent,
                    prep=token.text,
                    source="prep",
                    sent_idx=sent_idx,
                    chunk_id=chunk_id,
                    dep_distance=_dep_distance(token.head, pobj),
                ))
        return triples

    def _nominal(self, sent: Span, span_map, chunk_id, sent_idx) -> list[RawTriple]:
        """Nominal: 'Tesla founder Musk' or 'Musk, founder of Tesla'."""
        triples = []
        for token in sent:
            if token.lemma_.lower() not in ROLE_NOUNS:
                continue
            # Possessive or nmod child = the entity being described
            org_tok = next(
                (c for c in token.children if c.dep_ in {"nmod", "poss", "prep"}),
                None,
            )
            # The head noun phrase = the person
            person_tok = token.head
            if org_tok is None:
                continue
            org_ent    = _token_to_entity(org_tok, span_map)
            person_ent = _token_to_entity(person_tok, span_map)
            if org_ent and person_ent:
                triples.append(RawTriple(
                    subject=person_ent,
                    predicate=token.lemma_,
                    object=org_ent,
                    prep=None,
                    source="nominal",
                    sent_idx=sent_idx,
                    chunk_id=chunk_id,
                    dep_distance=_dep_distance(person_tok, org_tok),
                ))
        return triples
