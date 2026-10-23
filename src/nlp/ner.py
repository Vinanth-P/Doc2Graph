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
                canonical_name=m.group().strip().title(),
                type=etype,
                start_char=m.start(),
                end_char=m.end(),
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
                            canonical_name=matched.strip().title(),
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
            canonical_name=ent.text.strip().title(),
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

        merged = self._merge({
            "regex":     regex_ents,
            "gazetteer": gazetteer_ents,
            "spacy":     spacy_ents,
        })
        # Only return node-types for graph construction
        return [e for e in merged if e.type in NODE_TYPES]
