"""Exact and near-duplicate document detection and group clustering.

Ensures exact duplicate documents are tracked and excluded, and near-duplicates
(e.g. revised press releases, template variants) are clustered into the same
`group_key` so they are never split across train, validation, calibration, or test sets.
"""
from __future__ import annotations

import hashlib
import re
from typing import List, Dict, Set, Tuple, Any


def compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_for_dedup(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(cleaned.split())


def get_shingles(text: str, k: int = 4) -> Set[str]:
    norm = normalize_for_dedup(text)
    if len(norm) < k:
        return {norm} if norm else set()
    return {norm[i:i + k] for i in range(len(norm) - k + 1)}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


class DisjointSet:
    def __init__(self):
        self.parent = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            # Prefer group: prefix if present, else min for determinism
            if root_b.startswith("group:") and not root_a.startswith("group:"):
                self.parent[root_a] = root_b
            elif root_a.startswith("group:") and not root_b.startswith("group:"):
                self.parent[root_b] = root_a
            elif root_a < root_b:
                self.parent[root_b] = root_a
            else:
                self.parent[root_a] = root_b


def analyze_duplicates_and_groups(records, near_threshold=0.85):
    """Group every record before exclusion, preserving duplicate group bridges.

    Only raw-identical records with equivalent annotation/reviewer payloads are
    excluded. Differently annotated copies remain together for adjudication.
    """
    import json
    if not 0 < near_threshold <= 1:
        raise ValueError('Near-duplicate threshold must be in (0, 1]')
    ids = [r['document_id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate document IDs are not allowed')
    dsu = DisjointSet()
    for r in records:
        dsu.union('doc:' + r['document_id'], 'group:' + r.get('doc_group', r['document_id']))
    shingles = {r['document_id']: get_shingles(r['text']) for r in records}
    for i, first in enumerate(records):
        for second in records[i + 1:]:
            if (normalize_for_dedup(first['text']) == normalize_for_dedup(second['text']) or
                jaccard_similarity(shingles[first['document_id']], shingles[second['document_id']]) >= near_threshold):
                dsu.union('doc:' + first['document_id'], 'doc:' + second['document_id'])

    def annotation_signature(record):
        # Keep annotation IDs and review metadata: differing records are not
        # automatically adjudicated just because their document text matches.
        mentions = [{k: v for k, v in m.items() if k != 'document_id'} for m in record.get('mentions', [])]
        payload = {k: record.get(k) for k in ('relation_annotations', 'reviewer', 'review_status',
                   'review_guidelines_version', 'review_timestamp', 'data_origin', 'annotation_scope', 'license_or_rights')}
        payload['mentions'] = mentions
        return json.dumps(payload, sort_keys=True)

    seen, kept, exclusions = {}, [], []
    for record in sorted(records, key=lambda r: r['document_id']):
        key = compute_content_hash(record['text']), annotation_signature(record)
        if key in seen:
            exclusions.append({'document_id': record['document_id'], 'title': record.get('title', ''),
                               'reason': 'Identical content and annotation provenance', 'duplicate_of': seen[key]})
        else:
            seen[key] = record['document_id']
            kept.append(record)
    group_map = {r['document_id']: dsu.find('doc:' + r['document_id']).removeprefix('group:') for r in records}
    return kept, exclusions, group_map
