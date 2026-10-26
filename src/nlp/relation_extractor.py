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
from src.ingestion.preprocessor import SentenceChunk


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
    chunk_id: str           # actual source chunk ID (for evidence attribution)

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
    chunk_id: str = "",
    coref_map: dict[str, str] | None = None,
    entity_by_id: dict[str, Entity] | None = None,
) -> Optional[Entity]:
    """Try to find an NER entity covering this token (or its head noun phrase or via coreference)."""
    # 1. Direct or stripped span match
    clean_text = token.text.rstrip(".,;:!?\"')}]")
    t_start, t_end = token.idx, token.idx + len(clean_text)

    exact_matches = [
        ent for ent in span_map.values()
        if ent.text.rstrip(".,;:!?\"')}]").casefold() == clean_text.casefold()
    ]
    if not exact_matches and entity_by_id:
        exact_matches = [
            ent for ent in entity_by_id.values()
            if ent.text.rstrip(".,;:!?\"')}]").casefold() == clean_text.casefold()
        ]
    if exact_matches:
        ent = min(exact_matches, key=lambda candidate: abs(candidate.start_char - t_start))
        if coref_map and entity_by_id and ent.entity_id in coref_map:
            return entity_by_id.get(coref_map[ent.entity_id], ent)
        return ent

    for (s, e), ent in span_map.items():
        if (s == token.idx and e == token.idx + len(token.text)) or (s == t_start and e == t_end):
            if coref_map and entity_by_id and ent.entity_id in coref_map:
                return entity_by_id.get(coref_map[ent.entity_id], ent)
            return ent
        # Overlapping span
        if not (e <= t_start or s >= t_end):
            if coref_map and entity_by_id and ent.entity_id in coref_map:
                return entity_by_id.get(coref_map[ent.entity_id], ent)
            return ent

    # Try noun chunks
    for chunk in token.doc.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            for (s, e), ent in span_map.items():
                if not (e <= chunk.start_char or s >= chunk.end_char):
                    if coref_map and entity_by_id and ent.entity_id in coref_map:
                        return entity_by_id.get(coref_map[ent.entity_id], ent)
                    return ent

    # Coreference resolution for pronouns and definite NPs
    if coref_map and entity_by_id:
        tok_low = clean_text.lower()
        position_key = f"{chunk_id}:ref:{token.idx}"
        target_id = coref_map.get(position_key, coref_map.get(f"{chunk_id}:{tok_low}"))
        if not target_id:
            for chunk in token.doc.noun_chunks:
                if chunk.start <= token.i < chunk.end:
                    target_id = coref_map.get(
                        f"{chunk_id}:ref:{chunk.start_char}",
                        coref_map.get(f"{chunk_id}:{chunk.text.lower()}"),
                    )
                    if target_id:
                        break
        if target_id and target_id in entity_by_id:
            return entity_by_id[target_id]

    return None


def _dep_distance(a: Token, b: Token) -> int:
    """Shortest path through the common ancestor, including both endpoints."""
    distances = {token.i: depth for depth, token in enumerate([a, *a.ancestors])}
    return min(
        (depth + distances[token.i] for depth, token in enumerate([b, *b.ancestors])
         if token.i in distances),
        default=999,
    )


def _has_negation(token: Token) -> bool:
    return any(c.dep_ == "neg" for c in token.subtree)


def _noun_phrase_product(token: Token, chunk_id: str) -> Optional[Entity]:
    """Create a product/concept entity for a supported verb's named object."""
    for noun_chunk in token.doc.noun_chunks:
        if noun_chunk.root.i != token.i:
            continue
        text = noun_chunk.text.strip()
        if len(text.split()) < 2:
            return None
        return Entity(
            entity_id=f"{chunk_id}:concept:{noun_chunk.start_char}:{noun_chunk.end_char}",
            text=text,
            canonical_name=" ".join(text.split()).strip(".,;:"),
            type="PRODUCT",
            start_char=noun_chunk.start_char,
            end_char=noun_chunk.end_char,
            source="noun_phrase",
            chunk_id=chunk_id,
            sentence=token.doc.text,
        )
    return None


# ── Extractor ─────────────────────────────────────────────────────────────────

class DepPatternExtractor:
    """Extract relation triples from spaCy dependency parses."""

    def extract(
        self,
        doc: Doc,
        entities: list[Entity],
        center_chunk_id: str,
        center_sent_idx: int,
        coref_map: dict[str, str] | None = None,
        entity_by_id: dict[str, Entity] | None = None,
        window_chunks: list[SentenceChunk] | None = None,
    ) -> list[RawTriple]:
        # Entity offsets and coreference keys belong to individual chunks.
        # Slice the parsed window back into those chunks before lookup; this
        # also attributes each relation to its actual evidence sentence.
        if window_chunks is not None:
            if doc.text != " ".join(chunk.text for chunk in window_chunks):
                raise ValueError("Parsed window text does not match its chunks")
            triples = []
            offset = 0
            for chunk in window_chunks:
                span = doc.char_span(offset, offset + len(chunk.text))
                if span is None:
                    raise ValueError("Chunk boundaries do not align with parsed tokens")
                triples.extend(self.extract(
                    span.as_doc(), [e for e in entities if e.chunk_id == chunk.chunk_id],
                    chunk.chunk_id, chunk.sent_idx, coref_map, entity_by_id,
                ))
                offset += len(chunk.text) + 1
            return triples

        span_map = _build_span_map(entities)
        triples: list[RawTriple] = []
        for sent in doc.sents:
            triples += self._svo(sent, span_map, center_chunk_id, center_sent_idx, coref_map, entity_by_id)
            triples += self._appositive(sent, span_map, center_chunk_id, center_sent_idx, coref_map, entity_by_id)
            triples += self._prepositional(sent, span_map, center_chunk_id, center_sent_idx, coref_map, entity_by_id)
            triples += self._nominal(sent, span_map, center_chunk_id, center_sent_idx, coref_map, entity_by_id)
        return triples

    # ── Pattern implementations ───────────────────────────────────────────────

    def _svo(
        self, sent: Span, span_map, chunk_id, sent_idx,
        coref_map: dict[str, str] | None = None,
        entity_by_id: dict[str, Entity] | None = None,
    ) -> list[RawTriple]:
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
            # Convert passive voice to actor -> patient before normalization.
            if subj_tok is not None and subj_tok.dep_ == "nsubjpass":
                agent = next((c for c in token.children if c.dep_ == "agent"), None)
                actor = next((c for c in agent.children if c.dep_ == "pobj"), None) if agent is not None else None
                subj_tok, obj_tok = actor, subj_tok
            if subj_tok is None or obj_tok is None:
                continue
            subj_ent = _token_to_entity(subj_tok, span_map, chunk_id, coref_map, entity_by_id)
            obj_ent = _token_to_entity(
                obj_tok, span_map, chunk_id, coref_map, entity_by_id
            )
            prep = next(
                (child for child in token.children
                 if child.dep_ == "prep" and child.text.lower() == "in"),
                None,
            )
            if token.lemma_ == "store" and prep is not None:
                storage_obj = next(
                    (child for child in prep.children if child.dep_ == "pobj"),
                    None,
                )
                storage_ent = (
                    _token_to_entity(
                        storage_obj, span_map, chunk_id, coref_map, entity_by_id
                    )
                    if storage_obj is not None else None
                )
                if storage_ent is not None:
                    obj_ent = storage_ent
            if obj_ent is None and token.lemma_ in {
                "provide", "supply", "store", "use", "develop", "lead"
            }:
                obj_ent = _noun_phrase_product(obj_tok, chunk_id)
            if obj_ent is None and token.lemma_ in {"monitor"}:
                obj_ent = _token_to_entity(
                    obj_tok, span_map, chunk_id, coref_map, entity_by_id
                )
            if (
                token.lemma_ == "store"
                and subj_ent is not None
                and obj_ent is not None
                and subj_ent.entity_id == obj_ent.entity_id
            ):
                continue
            if subj_ent is None or obj_ent is None:
                continue
            prep_text = prep.text if prep is not None else next(
                (c.text for c in token.children if c.dep_ == "prep"), None
            )
            triples.append(RawTriple(
                subject=subj_ent,
                predicate=token.lemma_,
                object=obj_ent,
                prep=prep_text,
                source="svo",
                sent_idx=sent_idx,
                chunk_id=chunk_id,
                dep_distance=_dep_distance(subj_tok, obj_tok),
                has_negation=_has_negation(token),
            ))
        return triples

    def _appositive(
        self, sent: Span, span_map, chunk_id, sent_idx,
        coref_map: dict[str, str] | None = None,
        entity_by_id: dict[str, Entity] | None = None,
    ) -> list[RawTriple]:
        """Appositive: 'Elon Musk, CEO of Tesla' → (Musk, appos, Tesla)."""
        triples = []
        for token in sent:
            if token.dep_ != "appos":
                continue
            head_ent  = _token_to_entity(token.head, span_map, chunk_id, coref_map, entity_by_id)
            appos_ent = _token_to_entity(token, span_map, chunk_id, coref_map, entity_by_id)
            if appos_ent is None:
                # e.g. "Tim Cook, CEO of Apple" -> CEO is appos, Apple is prep.pobj
                for child in token.children:
                    if child.dep_ == "prep":
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                appos_ent = _token_to_entity(
                                    grandchild, span_map, chunk_id, coref_map, entity_by_id
                                )
                                break
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
                    has_negation=_has_negation(token),
                ))
        return triples

    def _prepositional(
        self, sent: Span, span_map, chunk_id, sent_idx,
        coref_map: dict[str, str] | None = None,
        entity_by_id: dict[str, Entity] | None = None,
    ) -> list[RawTriple]:
        """Prepositional: 'director of Tesla', 'based in California'."""
        triples = []
        for token in sent:
            if token.dep_ != "prep":
                continue
            pobj = next((c for c in token.children if c.dep_ == "pobj"), None)
            if pobj is None:
                continue
            governor = token.head
            subject_token = None
            subject_ent_override = None
            # Reduced clauses such as "TechNova Solutions partnered with
            # IISc" are attached to the subject noun by spaCy. Use that
            # noun as the subject before walking to an unrelated ancestor.
            if (
                governor.pos_ in {"VERB", "AUX"}
                and governor.dep_ in {"acl", "relcl"}
            ):
                subject_token = governor.head

            # In "AIRI in Bengaluru" and "IISc, located in Bengaluru",
            # the location preposition can attach to a nearby verb/adjective.
            # A preceding organization is the subject of the location fact.
            if token.text.lower() in {"in", "at"} and pobj is not None:
                preceding_orgs = [
                    entity for entity in span_map.values()
                    if entity.type in {"ORG", "PERSON"}
                    and entity.end_char <= token.idx
                ]
                if preceding_orgs:
                    nearest = max(preceding_orgs, key=lambda entity: entity.end_char)
                    if _token_to_entity(pobj, span_map, chunk_id, coref_map, entity_by_id):
                        subject_ent_override = nearest

            if governor.pos_ in {"VERB", "AUX", "ADJ"}:
                if subject_token is None:
                    subject_token = next(
                        (c for c in governor.children if c.dep_ in {"nsubj", "nsubjpass"}),
                        None,
                    )
                if subject_token is None and governor.pos_ == "ADJ":
                    subject_token = governor.head
            else:
                subject_token = governor
            if subject_token is None:
                # Nominal modifiers such as "funded the phase of DocSense"
                # attach the preposition to a noun, not to the governing verb.
                for ancestor in governor.ancestors:
                    if ancestor.pos_ == "VERB":
                        subject_token = next(
                            (c for c in ancestor.children
                             if c.dep_ in {"nsubj", "nsubjpass"}),
                            None,
                        )
                        if subject_token is not None:
                            governor = ancestor
                            break
            if subject_token is None:
                # SpaCy may attach a reduced clause ("Solutions partnered")
                # or a participle ("IISc, located") to the preceding noun.
                # Use the nearest preceding entity as the grammatical subject
                # only when the dependency tree supplies no subject.
                preceding = [
                    entity for entity in span_map.values()
                    if entity.end_char <= token.idx
                ]
                if preceding:
                    subject_ent_override = max(
                        preceding, key=lambda entity: entity.end_char
                    )
            if (
                governor.lemma_ in {"join", "partner"}
                and token.text.lower() in {"in", "at"}
                and pobj is not None
            ):
                verb_subject = next(
                    (child for child in governor.children
                     if child.dep_ in {"nsubj", "nsubjpass"}),
                    None,
                )
                verb_object = next(
                    (child for child in governor.children
                     if child.dep_ in {"dobj", "attr", "oprd"}),
                    None,
                )
                joined_entity = _token_to_entity(
                    verb_object, span_map, chunk_id, coref_map, entity_by_id
                ) if verb_object is not None else None
                if joined_entity is not None:
                    subject_ent_override = joined_entity
                    predicate = "locate"
                else:
                    predicate = governor.lemma_.lower()
            else:
                predicate = governor.lemma_.lower()

            if subject_ent_override is not None:
                head_ent = subject_ent_override
            elif subject_token is None:
                head_ent = subject_ent_override
            else:
                head_ent = _token_to_entity(subject_token, span_map, chunk_id, coref_map, entity_by_id)
            if head_ent is None:
                for ancestor in (governor, *governor.ancestors):
                    if ancestor.pos_ != "VERB":
                        continue
                    candidate = next(
                        (child for child in ancestor.children
                         if child.dep_ in {"nsubj", "nsubjpass"}),
                        None,
                    )
                    if candidate is not None:
                        candidate_ent = _token_to_entity(
                            candidate, span_map, chunk_id, coref_map, entity_by_id
                        )
                        if candidate_ent is not None:
                            head_ent = candidate_ent
                            subject_token = candidate
                            governor = ancestor
                            break
            obj_ent  = _token_to_entity(pobj, span_map, chunk_id, coref_map, entity_by_id)
            if head_ent and obj_ent:
                triples.append(RawTriple(
                    subject=head_ent,
                    predicate=predicate,
                    object=obj_ent,
                    prep=token.text,
                    source="prep",
                    sent_idx=sent_idx,
                    chunk_id=chunk_id,
                    dep_distance=_dep_distance(subject_token or governor, pobj),
                    has_negation=_has_negation(governor),
                ))
        return triples

    def _nominal(
        self, sent: Span, span_map, chunk_id, sent_idx,
        coref_map: dict[str, str] | None = None,
        entity_by_id: dict[str, Entity] | None = None,
    ) -> list[RawTriple]:
        """Nominal: 'Tesla founder Musk' or 'Musk, founder of Tesla'."""
        triples = []
        for token in sent:
            if token.lemma_.lower() not in ROLE_NOUNS:
                continue
            # Possessive or nmod child = the entity being described
            org_tok = next(
                (c for c in token.children if c.dep_ in {"compound", "nmod", "poss", "prep"}),
                None,
            )
            # The head noun phrase = the person
            person_tok = token.head
            if org_tok is None:
                continue
            org_ent    = _token_to_entity(org_tok, span_map, chunk_id, coref_map, entity_by_id)
            person_ent = _token_to_entity(person_tok, span_map, chunk_id, coref_map, entity_by_id)
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
                    has_negation=_has_negation(token),
                ))
        return triples
