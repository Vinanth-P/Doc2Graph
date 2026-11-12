"""Document-local name recovery using explicit definitions and role context.

No company/person names are baked into these rules. Definitions discovered in
the document repair missed mentions and inconsistent small-model NER labels.
Offsets always refer to the original chunk; canonical names carry the aliases.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import re

from src.nlp.ner import Entity, _canonical_name
from src.nlp.span_merger import merge_spans

_WORD = r"[A-Z][A-Za-z&]*(?:[.-][A-Za-z]+)*(?:;(?=\s))?"
_NAME = rf"{_WORD}(?:\s+(?:{_WORD}|of|and|the)){{0,7}}"
_ORG_NAME = rf"{_WORD}(?:\s+(?:{_WORD}|of|the)){{0,7}}"
_ORG = re.compile(
    rf"\b{_ORG_NAME}\s+(?:Pvt\.\s*Ltd\.?|Ltd\.?|LLP|LLC|Inc\.?|"
    r"Capital|Ventures|Partners|Associates|Bank|Mutual Fund)\b\.?"
)
_ROLE = (r"Chief\s+(?:Executive|Financial|Technology|Operating)\s+Officer|"
         r"(?:engineering\s+)?head|(?:independent\s+|product\s+)?director|"
         r"founder(?:\s+and\s+CEO)?|CEO|CFO|CTO")
_ROLE_PERSON = re.compile(rf"\b(?:{_ROLE}),?\s+(?P<name>{_WORD}(?:\s+{_WORD}){{1,2}})\b")
_PRODUCT = re.compile(rf"\b(?:platform|product|division|software),\s+(?P<name>{_NAME})(?=[,.])")
_INSTITUTION = re.compile(rf"\b(?:{_WORD}\s+){{0,3}}(?:Stock Exchange|Commission)\s+of\s+{_WORD}\b")
_LEGAL_SUFFIX = re.compile(r"\s+(?:(?:Pvt|Private)\.?\s+)?(?:Ltd|Limited|LLP|LLC|Inc)\.?$", re.I)
_FISCAL = re.compile(r"FY\d{2,4}(?:\s+and\s+FY\d{2,4})?", re.I)


def recover_document_entities(chunks, entities_by_chunk):
    """Return repaired mentions, including unambiguous explicit aliases."""
    repaired = {}
    definitions = defaultdict(set)

    def make(chunk, start, end, kind):
        surface = chunk.text[start:end]
        return Entity(f"{chunk.chunk_id}:{start}:{end}", surface,
                      _canonical_name(surface.replace(";", "")), kind, start, end,
                      "document_context", chunk.chunk_id, chunk.text)

    for chunk in chunks:
        if re.match(r"(?:Suggested\s+)?extraction\s+targets\s*:", chunk.text, re.I):
            repaired[chunk.chunk_id] = []
            continue
        context = []
        for pattern, kind in ((_ORG, "ORG"), (_INSTITUTION, "ORG"), (_ROLE_PERSON, "PERSON"), (_PRODUCT, "PRODUCT")):
            for match in pattern.finditer(chunk.text):
                start, end = match.span("name") if "name" in match.re.groupindex else match.span()
                context.append(make(chunk, start, end, kind))
        # Location modifiers provide stronger evidence than a guessed ORG label.
        for match in re.finditer(rf"\b({_WORD}(?:\s+{_WORD})?)-based\b", chunk.text):
            context.append(make(chunk, *match.span(1), "GPE"))
        for match in re.finditer(rf"\b(?:site|centres?|centers?|facilities)\s+in\s+({_WORD})\b", chunk.text):
            context.append(make(chunk, *match.span(1), "GPE"))
        candidates = [e for e in entities_by_chunk.get(chunk.chunk_id, [])
                      if not _FISCAL.fullmatch(e.text)
                      and not re.fullmatch(r"[A-Z]+(?:_[A-Z]+)+", e.text)
                      and not (e.source == "spacy" and e.type == "ORG" and e.text.islower())]
        repaired[chunk.chunk_id] = merge_spans({"context": context, "ner": candidates})
        for ent in repaired[chunk.chunk_id]:
            if ent.source == "document_context":
                definitions[ent.text.casefold()].add((ent.canonical_name, ent.type))
            # Parenthetical quoted aliases: Acme Analytics Ltd. ("Acme").
            tail = chunk.text[ent.end_char:]
            alias = re.match(r'\s*\(["\u201c\u2018\']([^"\u201d\u2019\']+)["\u201d\u2019\']\)', tail)
            if alias and ent.type == "ORG":
                definitions[alias.group(1).casefold()].add((ent.canonical_name, ent.type))
            if ent.type == "ORG" and _LEGAL_SUFFIX.search(ent.canonical_name):
                short = _LEGAL_SUFFIX.sub("", ent.canonical_name)
                definitions[short.casefold()].add((ent.canonical_name, ent.type))
            if ent.type == "PRODUCT" and len(ent.text.split()) > 1:
                short = ent.text.split()[-1]
                if any(c.isupper() for c in short[1:]):
                    definitions[short.casefold()].add((ent.canonical_name, ent.type))
        ticker = re.search(r"\blisted\b.*\bticker\s+([A-Z][A-Z0-9.]+)", chunk.text)
        if ticker:
            subjects = [e for e in repaired[chunk.chunk_id] if e.type == "ORG" and e.end_char <= ticker.start()]
            if len(subjects) == 1:
                subject = subjects[0]
                target = definitions.get(subject.text.casefold(), {(subject.canonical_name, "ORG")})
                definitions[ticker.group(1).casefold()].update(target)

    # A surname is usable only if it identifies exactly one full person name.
    people = defaultdict(set)
    for ents in repaired.values():
        for ent in ents:
            if ent.type == "PERSON" and len(ent.canonical_name.split()) >= 2:
                people[ent.canonical_name.split()[-1].casefold()].add((ent.canonical_name, "PERSON"))
    for name, targets in people.items():
        if len(targets) == 1:
            definitions[name].update(targets)

    known = {name: next(iter(targets)) for name, targets in definitions.items() if len(targets) == 1}
    # Prefer the explicit legal name when a short name also has its own seed.
    for name, targets in definitions.items():
        legal = {target for target in targets if target[1] == "ORG" and _LEGAL_SUFFIX.search(target[0])
                 and _LEGAL_SUFFIX.sub("", target[0]).casefold() == name}
        if len(legal) == 1:
            known[name] = next(iter(legal))
    # Use long names first, so an alias cannot swallow part of a product name.
    patterns = [(re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.I), value)
                for name, value in sorted(known.items(), key=lambda item: -len(item[0]))]
    for chunk in chunks:
        if re.match(r"(?:Suggested\s+)?extraction\s+targets\s*:", chunk.text, re.I):
            continue
        old = repaired[chunk.chunk_id]
        recovered = []
        for pattern, (canonical, kind) in patterns:
            for match in pattern.finditer(chunk.text):
                # Keep longer named products/orgs, but repair role-contaminated spans.
                covering = [e for e in old if e.start_char <= match.start() and e.end_char >= match.end()
                            and (e.start_char, e.end_char) != match.span()]
                if any(not re.search(r"\b(?:Chief|Officer|head|director)\b", e.text, re.I)
                       and e.text[match.end() - e.start_char:] not in {"'s", "\u2019s"}
                       for e in covering):
                    continue
                ent = make(chunk, *match.span(), kind)
                recovered.append(replace(ent, canonical_name=canonical))
        repaired[chunk.chunk_id] = merge_spans({"recovered": recovered, "existing": old})
    return repaired
