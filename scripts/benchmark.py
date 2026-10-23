#!/usr/bin/env python
"""Benchmark NLP quality against gold sets and optionally train scorers.

Usage:
    python scripts/benchmark.py --eval data/eval/
    python scripts/benchmark.py --eval data/eval/ --train
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def evaluate_ner(gold_path: Path) -> dict:
    """Compute span-level F1 for NER on ner_gold.jsonl."""
    from src.nlp.ner import NERRunner
    from src.ingestion.preprocessor import get_nlp

    records = [json.loads(ln) for ln in gold_path.read_text().splitlines() if ln.strip()]
    if not records:
        print("  [NER] Gold set empty — skipping")
        return {}

    nlp = get_nlp()
    ner = NERRunner()
    tp = fp = fn = 0

    for rec in records:
        text = rec["text"]
        gold_spans = {(s["start"], s["end"], s["type"]) for s in rec["entities"]}
        doc = nlp(text)
        pred_spans = {
            (e.start_char, e.end_char, e.type)
            for e in ner.run(doc, "eval", text)
        }
        tp += len(gold_spans & pred_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def train_relation_scorer(gold_path: Path) -> dict:
    from src.nlp.relation_scorer import RelationScorer
    print("  Training relation scorer ...")
    return RelationScorer.train(str(gold_path))


def train_entity_resolver(gold_path: Path) -> dict:
    from src.graph.entity_resolver import EntityResolver
    print("  Training entity resolver ...")
    return EntityResolver.train(str(gold_path))


def main():
    parser = argparse.ArgumentParser(description="Doc2Graph benchmark & training")
    parser.add_argument("--eval", required=True, help="Path to data/eval/ directory")
    parser.add_argument("--train", action="store_true", help="Train scorers on gold sets")
    args = parser.parse_args()

    eval_dir = Path(args.eval)
    print(f"\n📊 Doc2Graph Benchmark  ({eval_dir})\n" + "─" * 40)

    # NER
    ner_gold = eval_dir / "ner_gold.jsonl"
    if ner_gold.exists():
        result = evaluate_ner(ner_gold)
        print(f"  NER    P={result['precision']}  R={result['recall']}  F1={result['f1']}")
    else:
        print("  [NER] ner_gold.jsonl not found — skipping")

    # Train scorers
    if args.train:
        print()
        rel_gold = eval_dir / "relations_gold.jsonl"
        if rel_gold.exists():
            r = train_relation_scorer(rel_gold)
            print(f"  Relation scorer CV F1: {r['cv_f1_mean']:.3f} ± {r['cv_f1_std']:.3f}")
        else:
            print("  [Relation scorer] relations_gold.jsonl not found — skipping")

        res_gold = eval_dir / "resolution_gold.jsonl"
        if res_gold.exists():
            r = train_entity_resolver(res_gold)
            print(f"  Entity resolver  CV F1: {r['cv_f1_mean']:.3f} ± {r['cv_f1_std']:.3f}")
        else:
            print("  [Entity resolver] resolution_gold.jsonl not found — skipping")

    print()


if __name__ == "__main__":
    main()
