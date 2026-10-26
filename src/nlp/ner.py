"""Named Entity Recognition — three-layer pipeline with span merge policy.

Layers (in priority order):
  1. Regex patterns  — dates, emails, codes, money, IDs
  2. Gazetteer       — exact + fuzzy match against domain wordlists
  3. spaCy CRF NER   — en_core_web_sm fallback

Spans from all three layers are merged by the span_merger module.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import spacy
from spacy.tokens import Doc

from src.type_map import SPACY_TO_INTERNAL, NODE_TYPES


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Entity:
    """A recognized named entity in a sentence."""
    entity_id: str           # "<chunk_id>:<start>:<end>"
    text: str                # surface form
    canonical_name: str      # normalized for resolution (title-cased, stripped)
    type: str                # internal type (PERSON, ORG, GPE, …)
    start_char: int          # character offset in the chunk text
    end_char: int
    source: str              # "regex" | "gazetteer" | "spacy"
    chunk_id: str
    sentence: str = ""       # original sentence context, if available

    def __hash__(self):
        return hash(self.entity_id)

    def __eq__(self, other):
        return isinstance(other, Entity) and self.entity_id == other.entity_id


# ── Regex patterns ────────────────────────────────────────────────────────────

_REGEX_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("DATE",  re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|"
        r"\d{4})\b"
    )),
    ("MONEY", re.compile(
        r"\b(?:USD|EUR|GBP|INR)?\s?\$?\d[\d,]*(?:\.\d+)?\s*(?:million|billion|trillion|M|B|K)?\b"
    )),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
]

_ORG_SUFFIXES = re.compile(
    r"\b(?:Solutions|Institute|Institutes|University|Corporation|Corp\.?|"
    r"Company|Systems|Technologies|Foundation|Bank|Labs?)\b$",
    re.IGNORECASE,
)
_NON_PERSON_STARTS = frozenset({
    "in", "on", "at", "by", "from", "to", "for", "the", "this", "that",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
})
_GENERIC_ACRONYMS = frozenset({"AI"})
_KNOWN_TECH_PRODUCTS = frozenset({"postgresql", "mongodb", "neo4j"})
_ACRONYM_DEFINITION = re.compile(
    r"\b(?:[A-Z][A-Za-z]+|of|the|and)(?:\s+(?:[A-Z][A-Za-z]+|of|the|and)){1,8}\s*"
    r"\(([A-Z]{2,}[a-z]?)\)"
)
_GPE_NAME_SUFFIXES = re.compile(r"\b(?:river|lake|mountain|sea)\b", re.IGNORECASE)


def _canonical_name(text: str) -> str:
    value = re.sub(r"\s+", " ", text.strip(" \t\r\n.,;:"))
    value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.IGNORECASE)
    return " ".join(
        token if (
            (token.isupper() and len(token) > 1)
            or any(char.isupper() for char in token[1:])
            or any(char.isdigit() for char in token)
        ) else token.title()
        for token in value.split()
    )


def regex_entities(text: str, chunk_id: str) -> list[Entity]:
    entities: list[Entity] = []
    for etype, pattern in _REGEX_PATTERNS:
        if etype not in NODE_TYPES and etype not in ("DATE", "MONEY", "EMAIL"):
            continue
        for m in pattern.finditer(text):
            eid = f"{chunk_id}:{m.start()}:{m.end()}"
            entities.append(Entity(
                entity_id=eid,
                text=m.group(),
                canonical_name=_canonical_name(m.group()),
                type=etype,
                start_char=m.start(),
                end_char=m.end(),
                source="regex",
                chunk_id=chunk_id,
            ))
    for match in re.finditer(
        r"\b(?:[A-Z]{2,}[a-z]?|[A-Z][a-z]+\d+[A-Za-z]*)\b", text
    ):
        if (
            match.group().casefold() in _NON_PERSON_STARTS
            or match.group() in _GENERIC_ACRONYMS
        ):
            continue
        entities.append(Entity(
            entity_id=f"{chunk_id}:{match.start()}:{match.end()}",
            text=match.group(),
            canonical_name=_canonical_name(match.group()),
            type="PRODUCT" if any(char.isdigit() for char in match.group()) else "ORG",
            start_char=match.start(),
            end_char=match.end(),
            source="regex",
            chunk_id=chunk_id,
        ))
    for match in re.finditer(r"\b[A-Z][A-Za-z]*[A-Z][A-Za-z]*\b", text):
        if match.group().casefold() not in _KNOWN_TECH_PRODUCTS:
            continue
        entities.append(Entity(
            entity_id=f"{chunk_id}:{match.start()}:{match.end()}",
            text=match.group(),
            canonical_name=_canonical_name(match.group()),
            type="PRODUCT",
            start_char=match.start(),
            end_char=match.end(),
            source="regex",
            chunk_id=chunk_id,
        ))
    for match in _ACRONYM_DEFINITION.finditer(text):
        full_name = match.group(0)[:match.group(0).rfind("(")].strip()
        entities.append(Entity(
            entity_id=f"{chunk_id}:{match.start()}:{match.start() + len(full_name)}",
            text=full_name,
            canonical_name=_canonical_name(full_name),
            type="ORG",
            start_char=match.start(),
            end_char=match.start() + len(full_name),
            source="regex",
            chunk_id=chunk_id,
        ))
    return entities


# ── Gazetteer ─────────────────────────────────────────────────────────────────

class GazetteerMatcher:
    """Fuzzy gazetteer lookup using rapidfuzz."""

    def __init__(self, gazetteer_dir: str = "data/gazetteers"):
        from rapidfuzz import process as rf_process
        self._rf = rf_process
        self._entries: dict[str, list[str]] = {}   # type → list of names
        base = Path(gazetteer_dir)
        if not base.is_absolute() and not base.exists():
            alt = Path(__file__).resolve().parents[2] / base
            if alt.exists():
                base = alt
        type_files = {
            "PERSON": "persons.txt",
            "ORG": "organizations.txt",
            "GPE": "locations.txt",
        }
        for etype, fname in type_files.items():
            fpath = base / fname
            if fpath.exists():
                names = [ln.strip() for ln in fpath.read_text().splitlines() if ln.strip()]
                self._entries[etype] = names

    def match(self, text: str, chunk_id: str, threshold: int = 90) -> list[Entity]:
        from rapidfuzz.fuzz import ratio

        # Compare dictionary names with actual token spans, not the entire
        # sentence. This finds multiple/repeated names and preserves offsets
        # even for case differences or a fuzzy spelling match.
        tokens = list(re.finditer(r"\w+(?:[&.'-]\w+)*", text))
        entities: list[Entity] = []
        for etype, names in self._entries.items():
            if not names:
                continue
            by_length: dict[int, list[str]] = {}
            for name in names:
                length = len(re.findall(r"\w+(?:[&.'-]\w+)*", name))
                if length:
                    by_length.setdefault(length, []).append(name)
            for length, candidates in by_length.items():
                for index in range(len(tokens) - length + 1):
                    start, end = tokens[index].start(), tokens[index + length - 1].end()
                    surface = text[start:end]
                    result = self._rf.extractOne(
                        surface, candidates, scorer=ratio, processor=str.casefold,
                        score_cutoff=threshold,
                    )
                    if result:
                        matched, _, _ = result
                        entities.append(Entity(
                            entity_id=f"{chunk_id}:{start}:{end}",
                            text=surface,
                            canonical_name=" ".join(matched.strip().split()).title(),
                            type=etype, start_char=start, end_char=end,
                            source="gazetteer", chunk_id=chunk_id,
                        ))
        return entities


# ── spaCy NER ─────────────────────────────────────────────────────────────────

def spacy_entities(doc: Doc, chunk_id: str) -> list[Entity]:
    entities: list[Entity] = []
    for ent in doc.ents:
        itype = SPACY_TO_INTERNAL.get(ent.label_)
        if itype is None:
            continue
        eid = f"{chunk_id}:{ent.start_char}:{ent.end_char}"
        entities.append(Entity(
            entity_id=eid,
            text=ent.text,
            canonical_name=_canonical_name(ent.text),
            type=itype,
            start_char=ent.start_char,
            end_char=ent.end_char,
            source="spacy",
            chunk_id=chunk_id,
        ))
    return entities


# ── NER Runner ────────────────────────────────────────────────────────────────

class NERRunner:
    """Three-layer NER: regex > gazetteer > spaCy CRF."""

    def __init__(self, gazetteer_dir: str = "data/gazetteers"):
        from src.nlp.span_merger import merge_spans
        self._merge = merge_spans
        self._gazetteers = GazetteerMatcher(gazetteer_dir)

    def run(self, doc: Doc, chunk_id: str, raw_text: str) -> list[Entity]:
        regex_ents    = regex_entities(raw_text, chunk_id)
        gazetteer_ents = self._gazetteers.match(raw_text, chunk_id)
        spacy_ents    = spacy_entities(doc, chunk_id)
        known_person_tokens = {
            token.casefold()
            for entity in spacy_ents
            if entity.type == "PERSON"
            for token in re.findall(r"[A-Za-z]+", entity.text)
        }
        pattern_ents: list[Entity] = []
        for match in re.finditer(
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", raw_text
        ):
            words = re.findall(r"[A-Za-z]+", match.group())
            if (
                len(words) == 2
                and not _ORG_SUFFIXES.search(match.group())
                and not _GPE_NAME_SUFFIXES.search(match.group())
                and words[0].casefold() not in _NON_PERSON_STARTS
                and (
                    any(word.casefold() in known_person_tokens for word in words)
                    or all(word[0].isupper() for word in words)
                )
            ):
                pattern_ents.append(Entity(
                    entity_id=f"{chunk_id}:{match.start()}:{match.end()}",
                    text=match.group(),
                    canonical_name=_canonical_name(match.group()),
                    type="PERSON",
                    start_char=match.start(),
                    end_char=match.end(),
                    source="pattern",
                    chunk_id=chunk_id,
                ))

        # Small, deterministic repair for common model confusions. It is
        # restricted to organization-name suffixes rather than arbitrary NPs.
        for entity in spacy_ents:
            if entity.type in {"PRODUCT", "ORG"} and _ORG_SUFFIXES.search(entity.text):
                entity.type = "ORG"
            if entity.text.casefold().endswith(" river"):
                entity.type = "GPE"

        merged = self._merge({
            "regex":     regex_ents,
            "gazetteer": gazetteer_ents,
            "pattern":   pattern_ents,
            "spacy":      spacy_ents,
        })
        # Only return node-types for graph construction
        # DATE/MONEY/EMAIL are retained by the lower-level extractors as
        # metadata, but are not graph nodes.
        return [
            Entity(**{**e.__dict__, "sentence": raw_text})
            for e in merged
            if e.type in NODE_TYPES
        ]
