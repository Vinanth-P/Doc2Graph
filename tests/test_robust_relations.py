import numpy as np
from scripts.train_robust_relations import select_threshold, align_candidates, report_scores
from src.nlp.learned_relations import RELATION_TYPES
from types import SimpleNamespace


def test_precision_constraint_does_not_choose_high_recall_low_precision():
    keys = list(range(30))
    scores = np.array([.9] * 5 + [.1] * 25)
    assert select_threshold(scores, keys, set(range(10))) == .9


def test_insufficient_correct_predictions_abstains():
    assert select_threshold([.9] * 4, list(range(4)), set(range(4))) > 1
    assert select_threshold([], [], set()) > 1


def test_duplicate_aligned_edges_do_not_inflate_support():
    assert select_threshold([.9] * 10, ['same'] * 10, {'same'}) > 1


def test_ambiguous_entities_are_not_given_positive_training_labels():
    result = {'chunks': [SimpleNamespace(chunk_id='c', start_char=0)],
              'all_entities': [SimpleNamespace(chunk_id='c', start_char=0, end_char=4,
                                               type='ORG', entity_id='e')], 'id_map': {'e': 'resolved'}}
    mentions = [dict(start=0, end=4, type='ORG', cluster=i) for i in (1, 2)]
    pair = SimpleNamespace(source_id='resolved', target_id='missing')
    assert align_candidates(result, mentions, [pair]) == [('unmatched:resolved', 'unmatched:missing')]


def test_metrics_keep_uncandidate_gold_in_recall_denominator():
    thresholds = {r: .5 for r in RELATION_TYPES}
    scores = np.array([[.9, 0, 0, 0, 0]])
    result = report_scores(scores, ['found'], [{'found', 'missed'}, set(), set(), set(), set()], thresholds)
    assert result['overall']['recall'] == .5
