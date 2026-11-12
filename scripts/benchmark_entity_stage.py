"""Benchmark entity recognition and resolution independently for Milestone D.

Evaluates NER and intra-doc resolution on development data using pairwise merge
and B-cubed cluster metrics without collapsing candidates or generating self-links.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.preprocessor import get_nlp
from src.nlp.ner import NERRunner
from src.graph.entity_resolver import EntityResolver
from src.nlp.contracts_v2 import MentionRecord
from src.nlp.contextual_ner import (
    NonDestructiveResolver,
    pairwise_merge_metrics,
    bcubed_metrics,
)


def evaluate_ner_on_document(doc_record: dict, ner_runner: NERRunner, nlp) -> dict:
    """Evaluate NER against gold mentions for a single document."""
    text = doc_record["text"]
    gold_mentions = doc_record["mentions"]
    gold_spans = {(m["start"], m["end"], m["type"]): m["id"] for m in gold_mentions}

    spacy_doc = nlp(text)
    pred_entities = ner_runner.run(spacy_doc, "chunk_0", text)
    pred_spans = {(e.start_char, e.end_char, e.type) for e in pred_entities}

    # Strict exact match
    strict_tp = len(set(gold_spans.keys()) & pred_spans)
    strict_fp = len(pred_spans - set(gold_spans.keys()))
    strict_fn = len(set(gold_spans.keys()) - pred_spans)

    # Relaxed overlap match (same type, overlapping span)
    relaxed_tp = 0
    matched_gold = set()
    for ps, pe, pt in pred_spans:
        for (gs, ge, gt), gid in gold_spans.items():
            if pt == gt and not (pe <= gs or ps >= ge):
                if gid not in matched_gold:
                    relaxed_tp += 1
                    matched_gold.add(gid)
                    break
    relaxed_fp = len(pred_spans) - relaxed_tp
    relaxed_fn = len(gold_spans) - relaxed_tp

    return {
        "strict": {"tp": strict_tp, "fp": strict_fp, "fn": strict_fn},
        "relaxed": {"tp": relaxed_tp, "fp": relaxed_fp, "fn": relaxed_fn},
    }


def evaluate_resolution_on_document(
    doc_record: dict,
    resolver: NonDestructiveResolver,
) -> tuple[dict, dict, list]:
    """Run non-destructive resolver on gold mentions and evaluate against gold clusters."""
    mentions = [
        MentionRecord(
            id=m["id"],
            document_id=doc_record["document_id"],
            text=m["text"],
            type=m["type"],
            start=m["start"],
            end=m["end"],
        )
        for m in doc_record["mentions"]
    ]
    gold_clusters = doc_record["annotated_groups"]

    pred_clusters, links = resolver.cluster_mentions(mentions)
    pw = pairwise_merge_metrics(gold_clusters, pred_clusters)
    bc = bcubed_metrics(gold_clusters, pred_clusters)

    # Identify false merges
    # Map mention_id to gold cluster
    m_to_gold = {}
    for c_id, m_ids in gold_clusters.items():
        for mid in m_ids:
            m_to_gold[mid] = c_id

    false_merges = []
    mentions_by_id = {m.id: m for m in mentions}
    for link in links:
        if link.decision == "linked":
            ga = m_to_gold.get(link.source_mention_id)
            gb = m_to_gold.get(link.target_mention_id)
            if ga is not None and gb is not None and ga != gb:
                false_merges.append({
                    "doc_id": doc_record["document_id"],
                    "mention_a": mentions_by_id[link.source_mention_id].text,
                    "mention_b": mentions_by_id[link.target_mention_id].text,
                    "type": mentions_by_id[link.source_mention_id].type,
                    "score": link.score,
                    "method": link.method,
                })

    return pw, bc, false_merges


def run_benchmark(input_jsonl: Path, limit: int = 0) -> dict:
    nlp = get_nlp()
    ner_runner = NERRunner()
    nd_resolver = NonDestructiveResolver(threshold=0.85)

    records = []
    for line in input_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if limit > 0:
        records = records[:limit]

    ner_counts = {"strict": Counter(), "relaxed": Counter()}
    pw_counts = Counter()
    bc_precisions = []
    bc_recalls = []
    all_false_merges = []

    for doc in records:
        ner_res = evaluate_ner_on_document(doc, ner_runner, nlp)
        ner_counts["strict"].update(ner_res["strict"])
        ner_counts["relaxed"].update(ner_res["relaxed"])

        pw, bc, fms = evaluate_resolution_on_document(doc, nd_resolver)
        pw_counts.update({"tp": pw["tp"], "fp": pw["fp"], "fn": pw["fn"]})
        bc_precisions.append(bc["precision"])
        bc_recalls.append(bc["recall"])
        all_false_merges.extend(fms)

    def summarize_c(c):
        p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": p, "recall": r, "f1": f1, **c}

    avg_bcp = sum(bc_precisions) / len(bc_precisions) if bc_precisions else 1.0
    avg_bcr = sum(bc_recalls) / len(bc_recalls) if bc_recalls else 1.0
    bc_f1 = 2 * avg_bcp * avg_bcr / (avg_bcp + avg_bcr) if (avg_bcp + avg_bcr) > 0 else 0.0

    report = {
        "dataset": input_jsonl.name,
        "documents": len(records),
        "ner": {
            "strict_typed": summarize_c(ner_counts["strict"]),
            "relaxed_boundary_typed": summarize_c(ner_counts["relaxed"]),
        },
        "resolution": {
            "pairwise_merge": summarize_c(pw_counts),
            "bcubed_cluster": {"precision": avg_bcp, "recall": avg_bcr, "f1": bc_f1},
            "total_false_merges": len(all_false_merges),
            "false_merges_sample": all_false_merges[:15],
        },
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/prepared/v2_redocred/regression_dev.jsonl"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("output/evaluation/entity_stage_benchmark.json"))
    args = parser.parse_args()

    report = run_benchmark(args.input, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2))
