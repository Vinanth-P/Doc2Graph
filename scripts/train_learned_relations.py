"""Fit relation labels from independent training data; tune on dev only."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from threadpoolctl import threadpool_limits

from scripts.evaluate_external import reconstruct, gold_facts
from src.ingestion.preprocessor import SentenceChunk
from src.nlp.ner import Entity
from src.nlp.learned_relations import FEATURE_VERSION, RELATION_TYPES, build_candidates, LearnedRelationExtractor
from src.nlp.predicate_normalizer import load_relations_config


def labeled_document(record):
    _, sentences, starts, mentions = reconstruct(record)
    chunks = [SentenceChunk(f"gold:{i}", "gold", i, text, starts[i], starts[i] + len(text))
              for i, text in enumerate(sentences)]
    entities, id_map = [], {}
    for index, mention in enumerate(mentions):
        if mention["type"] is None:
            continue
        cid = f"gold:{mention['sent_id']}"
        ent = Entity(f"mention:{index}", mention["text"], mention["text"], mention["type"],
                     mention["local_start"], mention["local_end"], "gold", cid, chunks[mention["sent_id"]].text)
        entities.append(ent)
        id_map[ent.entity_id] = str(mention["cluster"])
    return chunks, entities, id_map


def examples(records, allowed_pairs, training):
    rng = random.Random(20261110)
    candidates, labels, doc_ids = [], [], []
    total_gold, covered_gold = Counter(), Counter()
    for doc_index, record in enumerate(records):
        chunks, entities, ids = labeled_document(record)
        pairs = build_candidates(chunks, entities, ids, allowed_pairs)
        gold = gold_facts(record)
        total_gold.update(f[1] for f in gold)
        positives, negatives = [], []
        for pair in pairs:
            target = [int((pair.source_id, relation, pair.target_id) in gold) for relation in RELATION_TYPES]
            covered_gold.update(relation for relation, value in zip(RELATION_TYPES, target) if value)
            (positives if any(target) else negatives).append((pair, target))
        if training:
            # All positives; deterministic negatives from every document.
            negatives = rng.sample(negatives, min(len(negatives), max(20, len(positives) * 8)))
        for pair, target in positives + negatives:
            candidates.append(pair)
            labels.append(target)
            doc_ids.append(doc_index)
    return candidates, np.asarray(labels, dtype=np.int8), doc_ids, dict(total_gold), dict(covered_gold)


def tune_threshold(labels, probabilities, total_gold):
    # The gold denominator includes relations outside candidate/type coverage.
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    tp = recall[:-1] * labels.sum()
    full_recall = tp / max(total_gold, 1)
    p = precision[:-1]
    f1 = np.divide(2 * p * full_recall, p + full_recall, out=np.zeros_like(p), where=p + full_recall > 0)
    valid = (thresholds >= 0.05) & (tp >= 3)
    if not valid.any():
        return 1.01
    # Stable tie-break: highest threshold among equally scoring candidates.
    best = np.flatnonzero(valid & (f1 == f1[valid].max()))[-1]
    return float(thresholds[best])


def exclude_overlap(training, development):
    """Prefer honest holdout separation over trusting upstream split names."""
    def text_key(record):
        return " ".join(reconstruct(record)[0].casefold().split())
    blocked_titles = {r["title"].strip().casefold() for r in development}
    blocked_text = {text_key(r) for r in development}
    retained, excluded = [], []
    for record in training:
        title, text = record["title"].strip().casefold(), text_key(record)
        if title in blocked_titles or text in blocked_text:
            excluded.append(record["title"])
            continue
        retained.append(record)
        blocked_titles.add(title)
        blocked_text.add(text)
    return retained, excluded


def train(args):
    start = time.monotonic()
    training = json.loads(args.train.read_text(encoding="utf-8"))
    development = json.loads(args.dev.read_text(encoding="utf-8"))
    training, excluded = exclude_overlap(training, development)
    print(f"Excluded {len(excluded)} train records overlapping dev or earlier train records", flush=True)
    config = load_relations_config()
    type_pairs = {r: config[r]["entity_type_pairs"] for r in RELATION_TYPES}
    allowed_pairs = {tuple(p) for pairs in type_pairs.values() for p in pairs}
    candidates, y, _, train_gold, train_covered = examples(training, allowed_pairs, True)
    dev_candidates, dev_y, dev_ids, dev_gold, dev_covered = examples(development, allowed_pairs, False)
    print(json.dumps({"training_documents": len(training), "training_pairs": len(candidates),
                      "development_pairs": len(dev_candidates), "positives": y.sum(axis=0).tolist()}), flush=True)
    # All feature vocabulary/IDF and classifier weights are fitted on train only.
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=100000, min_df=2,
                                 sublinear_tf=True, strip_accents="unicode", dtype=np.float64)
    x = vectorizer.fit_transform(c.text for c in candidates)
    classifiers = []
    with threadpool_limits(limits=2):
        for index, relation in enumerate(RELATION_TYPES):
            model = LogisticRegression(C=4.0, solver="liblinear", class_weight="balanced", max_iter=300,
                                       random_state=20261110)
            model.fit(x, y[:, index])
            classifiers.append(model)
            print(f"Trained {relation}: {int(y[:, index].sum())} positive pairs", flush=True)
    artifact = {"feature_version": FEATURE_VERSION, "relations": RELATION_TYPES, "type_pairs": type_pairs,
                "vectorizer": vectorizer, "classifiers": classifiers, "thresholds": {},
                "metadata": {"train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
                             "dev_sha256": hashlib.sha256(args.dev.read_bytes()).hexdigest(),
                             "train_documents": len(training), "dev_documents": len(development),
                             "excluded_training_titles": excluded,
                             "seed": 20261110, "negative_sampling": "max(20, 8 * positive_pairs) per document",
                             "feature_description": "identity-masked entity-pair context word TF-IDF 1-3 grams",
                             "threshold_selection": "per-label dev F1, at least 3 TP, threshold >= .05"}}
    extractor = LearnedRelationExtractor(artifact)
    scores = extractor.probabilities(dev_candidates)
    totals = Counter()
    metrics = {}
    for index, relation in enumerate(RELATION_TYPES):
        threshold = tune_threshold(dev_y[:, index], scores[:, index], dev_gold[relation])
        artifact["thresholds"][relation] = threshold
        predictions = scores[:, index] >= threshold
        tp = int((predictions & dev_y[:, index].astype(bool)).sum())
        fp = int(predictions.sum()) - tp
        fn = dev_gold[relation] - tp
        metrics[relation] = {"tp": tp, "fp": fp, "fn": fn, "threshold": threshold}
        totals.update(tp=tp, fp=fp, fn=fn)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output, compress=3)
    report = {"metadata": artifact["metadata"], "training_pairs": len(candidates), "dev_pairs": len(dev_candidates),
              "train_gold": train_gold, "train_candidate_coverage": train_covered, "dev_gold": dev_gold,
              "dev_candidate_coverage": dev_covered, "dev_gold_entity_metrics": metrics, "totals": dict(totals),
              "elapsed_seconds": round(time.monotonic() - start, 1),
              "artifact_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix(".training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/learned_relations.joblib"))
    train(parser.parse_args())
