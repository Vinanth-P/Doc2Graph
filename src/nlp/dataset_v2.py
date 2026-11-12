"""Explicit supervision masks and mention-preserving dataset utilities.

Unannotated, unmatched, and ambiguous pairs are unknown, never implicit negatives.
"""
from __future__ import annotations
import hashlib


def pair_supervision(source_matches, target_matches, relations, positives, reviewed_negatives=()):
    """Return targets/masks for a directed, uniquely aligned cluster pair.

    `reviewed_negatives` must contain explicit (source, relation, target) decisions,
    not every fact absent from an incomplete corpus.
    """
    positives, negatives = set(positives), set(reviewed_negatives)
    if positives & negatives:
        raise ValueError('Contradictory positive and negative supervision')
    targets = [None] * len(relations)
    if len(source_matches) == len(target_matches) == 1:
        source, target = next(iter(source_matches)), next(iter(target_matches))
        if source != target:
            for i, relation in enumerate(relations):
                fact = source, relation, target
                if fact in positives:
                    targets[i] = 1
                elif fact in negatives:
                    targets[i] = 0
    return {'targets': targets, 'loss_mask': [value is not None for value in targets]}


def align_mentions(predicted, annotated):
    """Read-only exact typed matching; preserve every predicted mention ID.

    Multiple gold IDs at an identical span remain ambiguous. No fuzzy merge or
    boundary enlargement is performed to make training/evaluation scores better.
    """
    by_span = {}
    for mention in annotated:
        key = mention.document_id, mention.start, mention.end, mention.type
        by_span.setdefault(key, set()).add(mention.id)
    return {mention.id: frozenset(by_span.get(
        (mention.document_id, mention.start, mention.end, mention.type), set())) for mention in predicted}


def development_split(group_id, seed='doc2graph-v2-20261111'):
    """Stable group assignment, deliberately no final-test partition."""
    bucket = int(hashlib.sha256(f'{seed}:{group_id}'.encode()).hexdigest(), 16) % 100
    return 'train' if bucket < 80 else 'validation' if bucket < 90 else 'calibration'


def finance_split(group_id, seed='doc2graph-v2-finance-20261111'):
    """Deterministic 60/15/15/10 assignment; assignment does not seal test data."""
    bucket = int(hashlib.sha256(f'{seed}:{group_id}'.encode()).hexdigest(), 16) % 100
    if bucket < 60:
        return 'train'
    elif bucket < 75:
        return 'validation'
    elif bucket < 90:
        return 'calibration'
    else:
        return 'locked_test'

