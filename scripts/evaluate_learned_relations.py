"""Compare opt-in learned backend with rules; no fitting or threshold changes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_external import reconstruct, gold_facts, graph_predictions, metrics, summarize, oracle_predictions, SHARED
from scripts.train_learned_relations import labeled_document
from src.config import get_settings
from src.pipeline import run_pipeline_nlp
from src.ingestion.preprocessor import get_nlp
from src.nlp.learned_relations import load_learned_extractor, RELATION_TYPES
from src.nlp.predicate_normalizer import PredicateNormalizer


def evaluate(args):
    records = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.limit:
        records = records[:args.limit]
    settings = get_settings()
    settings.relation_backend = args.backend
    settings.learned_relation_model_path = str(args.model.resolve())
    extractor = load_learned_extractor(args.model) if args.backend == "learned" else None
    normalizer = PredicateNormalizer(include_extended_relations=True)
    nlp = get_nlp()
    counts = {mode: defaultdict(Counter) for mode in ("end_to_end", "gold_entities")}
    ner_counts, edge_types, rows, errors = Counter(), Counter(), [], []
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        text, sentences, starts, mentions = reconstruct(record)
        gold = gold_facts(record)
        predicted, oracle, evidence = set(), set(), {}
        try:
            result = run_pipeline_nlp(text.encode(), f"external-{index}.txt", "txt")
            predicted, ner, types, evidence = graph_predictions(result, mentions)
            ner_counts.update(ner)
            edge_types.update(types)
        except Exception as exc:
            errors.append({"index": index, "mode": "end_to_end", "error": str(exc)})
        try:
            if extractor:
                chunks, entities, ids = labeled_document(record)
                triples = extractor.extract(chunks, entities, ids)
                oracle = {(ids[t.subject.entity_id], t.relation_type, ids[t.object.entity_id]) for t in triples}
            else:
                oracle = oracle_predictions(record, sentences, starts, mentions, nlp, normalizer, None)
        except Exception as exc:
            errors.append({"index": index, "mode": "gold_entities", "error": str(exc)})
        row = {"index": index, "title": record["title"], "gold": sorted(gold)}
        for mode, predictions in (("end_to_end", predicted), ("gold_entities", oracle)):
            for scope in ("all_five", "shared_three", *RELATION_TYPES):
                included = set(RELATION_TYPES) if scope == "all_five" else SHARED if scope == "shared_three" else {scope}
                counts[mode][scope].update(metrics({f for f in predictions if f[1] in included},
                                                  {f for f in gold if f[1] in included}))
            row[mode] = {"predicted": sorted(predictions), "false_positives": sorted(predictions - gold),
                         "false_negatives": sorted(gold - predictions), **metrics(predictions, gold)}
        row["false_positive_evidence"] = [{"fact": list(f), "text": evidence.get(f, "")}
                                           for f in sorted(predicted - gold)]
        rows.append(row)
        if (index + 1) % 25 == 0 or index == len(records) - 1:
            print(json.dumps({"backend": args.backend, "completed": index + 1, "total": len(records),
                              "errors": len(errors), "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    report = {"dataset": args.dataset.name, "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
              "backend": args.backend, "documents": len(records), "mapped_gold_facts": sum(len(r["gold"]) for r in rows),
              "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest() if extractor else None,
              "metrics": {mode: {scope: summarize(values) for scope, values in groups.items()}
                          for mode, groups in counts.items()},
              "entity_mentions_exact_typed": summarize(ner_counts), "all_predicted_relation_counts": dict(edge_types),
              "errors": errors, "elapsed_seconds": round(time.monotonic() - started, 1), "documents_detail": rows}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"backend": args.backend, "end_to_end": report["metrics"]["end_to_end"]["all_five"],
                      "gold_entities": report["metrics"]["gold_entities"]["all_five"], "errors": len(errors)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--backend", choices=["rules", "learned"], required=True)
    parser.add_argument("--model", type=Path, default=Path("models/learned_relations.joblib"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    evaluate(parser.parse_args())
