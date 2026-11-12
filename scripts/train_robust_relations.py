"""Train on production candidates; calibrate and validate on separate dev halves.

Never reads the previously observed test split. Cache is local trusted joblib.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from scripts.train_learned_relations import exclude_overlap, examples
from scripts.evaluate_external import reconstruct, gold_facts, summarize
from src.config import get_settings
from src.pipeline import run_pipeline_nlp
from src.nlp.learned_relations import build_candidates, RELATION_TYPES, LearnedRelationExtractor


def align_candidates(result, mentions, pairs):
    spans, alignment = defaultdict(set), defaultdict(set)
    for m in mentions:
        if m['type']:
            spans[m['start'], m['end'], m['type']].add(str(m['cluster']))
    chunks = {c.chunk_id: c for c in result['chunks']}
    for ent in result['all_entities']:
        chunk = chunks[ent.chunk_id]
        alignment[result['id_map'].get(ent.entity_id, ent.entity_id)].update(
            spans[chunk.start_char + ent.start_char, chunk.start_char + ent.end_char, ent.type])
    def link(cid):
        values = alignment[cid]
        return next(iter(values)) if len(values) == 1 else 'unmatched:' + cid
    return [(link(p.source_id), link(p.target_id)) for p in pairs]


def collect(records, allowed, cache):
    # Include code and data identity so changed extraction cannot reuse stale rows.
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode())
    for path in sorted(Path('src').rglob('*.py')):
        digest.update(path.as_posix().encode()); digest.update(path.read_bytes())
    fingerprint = digest.hexdigest()
    if cache.exists():
        saved = joblib.load(cache)
        if saved['fingerprint'] == fingerprint:
            return saved['rows']
    rows = []
    for i, record in enumerate(records):
        text, _, _, mentions = reconstruct(record)
        try:
            result = run_pipeline_nlp(text.encode(), f'training-{i}.txt', 'txt')
        except ValueError as exc:
            if 'English only' not in str(exc):
                raise
            # Retain rejected documents and their gold denominator in evaluation.
            rows.append({'pairs': [], 'y': [], 'keys': [], 'gold': gold_facts(record),
                         'error': str(exc), 'title': record['title']})
            continue
        pairs = build_candidates(result['chunks'], result['all_entities'], result['id_map'], allowed)
        keys = align_candidates(result, mentions, pairs)
        gold = gold_facts(record)
        y = [[int((s, r, t) in gold) for r in RELATION_TYPES] for s, t in keys]
        rows.append({'pairs': pairs, 'y': y, 'keys': keys, 'gold': gold})
        if (i + 1) % 100 == 0:
            print(f'{cache.stem}: {i + 1}/{len(records)}', flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'fingerprint': fingerprint, 'rows': rows}, cache, compress=3)
    return rows


def select_threshold(scores, keys, gold, minimum_precision=.5):
    """Maximize recall subject to precision and minimum support; otherwise abstain."""
    best, best_tp = 1.01, 0
    for threshold in sorted(set(float(s) for s in scores), reverse=True):
        predictions = {k for k, s in zip(keys, scores) if s >= threshold}
        tp = len(predictions & gold)
        if tp >= 5 and tp / len(predictions) >= minimum_precision and tp > best_tp:
            best, best_tp = threshold, tp
    return best


def score_rows(extractor, rows):
    pairs = [p for row in rows for p in row['pairs']]
    keys = [(i, s, t) for i, row in enumerate(rows) for s, t in row['keys']]
    gold = [{(i, s, t) for i, row in enumerate(rows) for s, r, t in row['gold'] if r == relation}
            for relation in RELATION_TYPES]
    return extractor.probabilities(pairs), keys, gold


def report_scores(scores, keys, gold, thresholds):
    counts = Counter(); per_label = {}
    for j, relation in enumerate(RELATION_TYPES):
        predictions = {k for k, s in zip(keys, scores[:, j]) if s >= thresholds[relation]}
        c = dict(tp=len(predictions & gold[j]), fp=len(predictions - gold[j]), fn=len(gold[j] - predictions))
        counts.update(c); per_label[relation] = summarize(c)
    return {'overall': summarize(counts), 'per_label': per_label}


def main(args):
    old = joblib.load('models/learned_relations.joblib')
    settings = get_settings(); settings.relation_backend = 'learned'
    settings.learned_relation_model_path = str(Path('models/learned_relations.joblib').resolve())
    train_path = Path('data/external/redocred/train_revised.json')
    dev_path = Path('data/external/redocred/dev_revised.json')
    dev = json.loads(dev_path.read_text(encoding='utf-8'))
    train, excluded = exclude_overlap(json.loads(train_path.read_text(encoding='utf-8')), dev)
    allowed = {tuple(p) for pairs in old['type_pairs'].values() for p in pairs}
    train_rows = collect(train, allowed, Path('output/evaluation/cache/production_train.joblib'))
    dev_rows = collect(dev, allowed, Path('output/evaluation/cache/production_dev.joblib'))
    # Full production negatives, including unmatched and ambiguous entities.
    pairs = [p for row in train_rows for p in row['pairs']]
    y = np.asarray([v for row in train_rows for v in row['y']], dtype=np.int8)
    # Preserve clean positive coverage while learning actual production noise.
    clean, clean_y, _, _, _ = examples(train, allowed, False)
    pairs += clean; y = np.concatenate([y, clean_y])
    print(f'Fitting {len(pairs)} pairs; positives={y.sum(axis=0).tolist()}', flush=True)
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=150000, min_df=2, sublinear_tf=True,
                                 strip_accents='unicode', dtype=np.float64)
    x = vectorizer.fit_transform(p.text for p in pairs)
    classifiers = []
    with threadpool_limits(limits=2):
        for j, relation in enumerate(RELATION_TYPES):
            model = LogisticRegression(C=4, solver='liblinear', max_iter=500, random_state=20261110)
            model.fit(x, y[:, j]); classifiers.append(model)
    artifact = {**old, 'vectorizer': vectorizer, 'classifiers': classifiers, 'thresholds': {},
                'metadata': {'training': 'all gold and production candidates, no class balancing or negative sampling',
                             'train_documents': len(train), 'excluded_training_titles': excluded,
                             'train_sha256': hashlib.sha256(train_path.read_bytes()).hexdigest(),
                             'dev_sha256': hashlib.sha256(dev_path.read_bytes()).hexdigest(),
                             'calibration': 'even dev indices, precision >= .5, >= 5 TP, maximum recall',
                             'validation': 'odd dev indices; previously observed development data, not fresh holdout'}}
    extractor = LearnedRelationExtractor(artifact)
    calibration = score_rows(extractor, dev_rows[::2])
    for j, relation in enumerate(RELATION_TYPES):
        artifact['thresholds'][relation] = select_threshold(calibration[0][:, j], calibration[1], calibration[2][j])
    validation = score_rows(extractor, dev_rows[1::2])
    report = {'metadata': artifact['metadata'], 'thresholds': artifact['thresholds'], 'training_pairs': len(pairs),
              'rejected_documents': {'train': [{'title': r['title'], 'error': r['error']} for r in train_rows if 'error' in r],
                                     'dev': [{'title': r['title'], 'error': r['error']} for r in dev_rows if 'error' in r]},
              'calibration': report_scores(*calibration, artifact['thresholds']),
              'validation': report_scores(*validation, artifact['thresholds'])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output, compress=3)
    report['artifact_sha256'] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.training.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('models/robust_relations.joblib'))
    main(parser.parse_args())
