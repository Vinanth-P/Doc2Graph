"""Supervised entity-pair relation classification, independent of verb rules.

Candidate generation uses entity pairs and document context only. The fitted
TF-IDF vocabulary, logistic-regression weights and development thresholds live
in a versioned artifact. This experimental backend has five trained labels.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

import joblib
import numpy as np

from src.nlp.ner import Entity
from src.nlp.predicate_normalizer import NormalizedTriple

FEATURE_VERSION = 1
RELATION_TYPES = ("FOUNDED", "EMPLOYED_BY", "MEMBER_OF", "OWNS", "SUBSIDIARY_OF")


@dataclass
class PairCandidate:
    source_id: str
    target_id: str
    subject: Entity
    object: Entity
    text: str
    chunk_ids: tuple[str, ...]
    evidence: str


def _masked_context(chunk, mentions, id_map, source, target):
    """Mask identities, preserving entity roles, word order and context."""
    spans = []
    for ent in mentions:
        cid = id_map.get(ent.entity_id, ent.entity_id)
        role = "SUBJECT" if cid == source else "OBJECT" if cid == target else "OTHER"
        spans.append((0 if role != "OTHER" else 1, -(ent.end_char - ent.start_char),
                      ent.start_char, ent.end_char, f" {role}_{ent.type} "))
    selected = []
    for _, _, start, end, marker in sorted(spans):
        if not any(start < e and end > s for s, e, _ in selected):
            selected.append((start, end, marker))
    text, previous = "", 0
    for start, end, marker in sorted(selected):
        text += chunk.text[previous:start] + marker
        previous = end
    text += chunk.text[previous:]
    # Every target marker keeps nearby text, even in very long sentences.
    anchors = list(re.finditer(r"\b(?:SUBJECT|OBJECT)_\w+\b", text))
    intervals = []
    for anchor in anchors:
        start, end = max(0, anchor.start() - 300), min(len(text), anchor.end() + 300)
        if intervals and start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(end, intervals[-1][1]))
        else:
            intervals.append((start, end))
    return " CONTEXT_BREAK ".join(text[s:e] for s, e in intervals) if intervals else ""


def build_candidates(chunks, entities, id_map, allowed_pairs):
    """Enumerate ordered pairs; never consult relation labels or verb lists."""
    grouped, by_chunk = defaultdict(list), defaultdict(list)
    chunk_by_id = {c.chunk_id: c for c in chunks}
    order = {c.chunk_id: i for i, c in enumerate(chunks)}
    for ent in entities:
        if ent.chunk_id in order and ent.source != "noun_phrase":
            grouped[id_map.get(ent.entity_id, ent.entity_id)].append(ent)
            by_chunk[ent.chunk_id].append(ent)
    candidates = []
    for source in sorted(grouped):
        for target in sorted(grouped):
            if source == target:
                continue
            subjects, objects = grouped[source], grouped[target]
            if (subjects[0].type, objects[0].type) not in allowed_pairs:
                continue
            combinations = sorted(((s, o) for s in subjects for o in objects), key=lambda pair: (
                abs(order[pair[0].chunk_id] - order[pair[1].chunk_id]),
                abs(pair[0].start_char - pair[1].start_char) if pair[0].chunk_id == pair[1].chunk_id else 0,
                order[pair[0].chunk_id], order[pair[1].chunk_id]))
            contexts, selected_chunks, seen = [], set(), set()
            for subject, obj in combinations:
                pair_ids = tuple(sorted({subject.chunk_id, obj.chunk_id}, key=order.get))
                if pair_ids in seen:
                    continue
                seen.add(pair_ids)
                distance = abs(order[subject.chunk_id] - order[obj.chunk_id])
                direction = "SUBJECT_FIRST" if (order[subject.chunk_id], subject.start_char) < (
                    order[obj.chunk_id], obj.start_char) else "OBJECT_FIRST"
                masked = " SENTENCE_BREAK ".join(_masked_context(
                    chunk_by_id[cid], by_chunk[cid], id_map, source, target) for cid in pair_ids)
                contexts.append(f"PAIR_{subject.type}_{obj.type} DISTANCE_{min(distance, 4)} {direction} {masked}")
                selected_chunks.update(pair_ids)
                if len(contexts) == 2:
                    break
            subject, obj = combinations[0]
            ids = tuple(sorted(selected_chunks, key=order.get))
            candidates.append(PairCandidate(source, target, subject, obj, " EVIDENCE_BREAK ".join(contexts),
                                            ids, " ".join(chunk_by_id[cid].text for cid in ids)))
    return candidates


class LearnedRelationExtractor:
    def __init__(self, artifact):
        if artifact.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Incompatible learned relation artifact feature version")
        if tuple(artifact.get("relations", ())) != RELATION_TYPES:
            raise ValueError("Learned relation artifact has an incompatible label order")
        self.artifact = artifact
        self.allowed_pairs = {tuple(pair) for pairs in artifact["type_pairs"].values() for pair in pairs}

    def probabilities(self, candidates):
        if not candidates:
            return np.empty((0, len(RELATION_TYPES)))
        features = self.artifact["vectorizer"].transform(c.text for c in candidates)
        scores = np.column_stack([model.predict_proba(features)[:, 1] for model in self.artifact["classifiers"]])
        for i, relation in enumerate(RELATION_TYPES):
            pairs = {tuple(pair) for pair in self.artifact["type_pairs"][relation]}
            scores[:, i] *= np.asarray([(c.subject.type, c.object.type) in pairs for c in candidates])
        return scores

    def extract(self, chunks, entities, id_map):
        candidates = build_candidates(chunks, entities, id_map, self.allowed_pairs)
        scores = self.probabilities(candidates)
        chunk_by_id = {c.chunk_id: c for c in chunks}
        triples = []
        for candidate, probabilities in zip(candidates, scores):
            for relation, score in zip(RELATION_TYPES, probabilities):
                if score < self.artifact["thresholds"][relation]:
                    continue
                # Preserve every context chunk for cross-sentence provenance.
                for cid in candidate.chunk_ids:
                    triples.append(NormalizedTriple(
                        relation, candidate.subject, candidate.object, "learned_pair_classifier", cid,
                        chunk_by_id[cid].sent_idx, float(score), candidate.evidence, "learned_pair_classifier"))
        return triples


@lru_cache(maxsize=2)
def _load(path, modified_ns):
    # Only load locally produced trusted artifacts, never arbitrary uploads.
    return LearnedRelationExtractor(joblib.load(path))


def load_learned_extractor(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Learned relation model is missing: {path}. Train it before selecting this backend.")
    return _load(str(path), path.stat().st_mtime_ns)
