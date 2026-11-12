"""Summarize paired frozen evaluation reports with document-bootstrap intervals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .evaluate_external import SHARED
else:
    from evaluate_external import SHARED


def f1(counts):
    tp, fp, fn = np.asarray(counts).T
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)


def document_counts(report, mode, shared=False):
    rows = []
    for doc in report["documents_detail"]:
        gold = {tuple(f) for f in doc["gold"]}
        predicted = {tuple(f) for f in doc[mode]["predicted"]}
        if shared:
            gold = {f for f in gold if f[1] in SHARED}
            predicted = {f for f in predicted if f[1] in SHARED}
        rows.append([len(predicted & gold), len(predicted - gold), len(gold - predicted)])
    return np.asarray(rows)


def compare(before, after):
    assert before["dataset_sha256"] == after["dataset_sha256"]
    assert [(d["index"], d["title"], d["gold"]) for d in before["documents_detail"]] == [
        (d["index"], d["title"], d["gold"]) for d in after["documents_detail"]]
    result = {"documents": before["documents"], "mapped_gold_facts": before["mapped_gold_facts"],
              "dataset_sha256": before["dataset_sha256"], "comparisons": {}}
    rng = np.random.default_rng(20261110)
    samples = rng.integers(0, before["documents"], size=(2000, before["documents"]))
    for mode in ("end_to_end", "gold_entities"):
        result["comparisons"][mode] = {}
        for scope in ("shared_three", "all_five"):
            a = document_counts(before, mode, scope == "shared_three")
            b = document_counts(after, mode, scope == "shared_three")
            deltas = f1(b[samples].sum(axis=1)) - f1(a[samples].sum(axis=1))
            result["comparisons"][mode][scope] = {
                "baseline": before["metrics"][mode][scope], "revised": after["metrics"][mode][scope],
                "delta_f1": float(f1(b.sum(axis=0)) - f1(a.sum(axis=0))),
                "delta_f1_95pct_paired_bootstrap_interval": np.quantile(deltas, [0.025, 0.975]).tolist(),
            }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--revised", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = json.loads(args.baseline.read_text(encoding="utf-8"))
    after = json.loads(args.revised.read_text(encoding="utf-8"))
    result = compare(before, after)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
