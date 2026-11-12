"""Frozen-code evaluation on Re-DocRED; no training or database writes.

Run each snapshot in its own process. Dataset offsets are reconstructed from
space-joined tokens, and predictions are linked to gold entities by exact typed
mention spans only. Gold labels are never passed to production inference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import re
import unicodedata

# Dataset head -> tail directions, frozen before running the benchmark.
RELATIONS = {
    "P112": ("FOUNDED", True),       # organization -> founder
    "P108": ("EMPLOYED_BY", False), # person -> employer
    "P463": ("MEMBER_OF", False),   # member -> organization
    "P127": ("OWNS", True),         # owned entity -> owner
    "P749": ("SUBSIDIARY_OF", False),
}
SHARED = {"FOUNDED", "EMPLOYED_BY", "MEMBER_OF"}
TYPES = {"PER": "PERSON", "ORG": "ORG", "LOC": "GPE"}


def reconstruct(record):
    # Match production NFKC/whitespace normalization BEFORE deriving offsets.
    # Never use old character offsets after Unicode normalization changes length.
    tokenized = [[re.sub(r"\s+", " ", unicodedata.normalize("NFKC", token)).strip()
                  for token in tokens] for tokens in record["sents"]]
    sentences, token_spans = [], []
    for tokens in tokenized:
        sentence, spans = "", []
        for token in tokens:
            if sentence and token:
                sentence += " "
            start = len(sentence)
            sentence += token
            spans.append((start, len(sentence)))
        sentences.append(sentence)
        token_spans.append(spans)
    starts, offset = [], 0
    for sentence in sentences:
        starts.append(offset)
        offset += len(sentence) + 1
    mentions = []
    for cluster, group in enumerate(record["vertexSet"]):
        for mention in group:
            sid = mention["sent_id"]
            a, b = mention["pos"]
            local_start = token_spans[sid][a][0]
            local_end = token_spans[sid][b - 1][1]
            mentions.append({"cluster": cluster, "sent_id": sid,
                             "start": starts[sid] + local_start, "end": starts[sid] + local_end,
                             "local_start": local_start, "local_end": local_end,
                             "type": TYPES.get(mention["type"]), "text": sentences[sid][local_start:local_end]})
    return " ".join(sentences), sentences, starts, mentions


def gold_facts(record):
    facts = set()
    for label in record["labels"]:
        if label["r"] in RELATIONS:
            relation, reverse = RELATIONS[label["r"]]
            h, t = label["h"], label["t"]
            facts.add((str(t if reverse else h), relation, str(h if reverse else t)))
    return facts


def metrics(predicted, gold):
    tp, fp, fn = len(predicted & gold), len(predicted - gold), len(gold - predicted)
    return {"tp": tp, "fp": fp, "fn": fn}


def summarize(counts):
    tp, fp, fn = (counts.get(k, 0) for k in ("tp", "fp", "fn"))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


def graph_predictions(result, mentions):
    from src.graph.graph_builder import build_resolved_edges
    gold_spans = defaultdict(set)
    for m in mentions:
        if m["type"]:
            gold_spans[(m["start"], m["end"], m["type"])].add(str(m["cluster"]))
    chunks = {c.chunk_id: c for c in result["chunks"]}
    alignment = defaultdict(set)
    predicted_spans = set()
    for ent in result["all_entities"]:
        if ent.chunk_id not in chunks:
            continue
        chunk = chunks[ent.chunk_id]
        span = (chunk.start_char + ent.start_char, chunk.start_char + ent.end_char, ent.type)
        if ent.source != "noun_phrase" and ent.type in TYPES.values():
            predicted_spans.add(span)
        alignment[result["id_map"].get(ent.entity_id, ent.entity_id)].update(gold_spans.get(span, set()))
    def link(identifier):
        targets = alignment.get(identifier, set())
        return next(iter(targets)) if len(targets) == 1 else "unmatched:" + identifier
    predictions = set()
    all_types = Counter()
    evidence = {}
    for edge in build_resolved_edges(result["normalized"], result["id_map"]):
        all_types[edge.relation_type] += 1
        if edge.relation_type in {v[0] for v in RELATIONS.values()}:
            key = (link(edge.source_id), edge.relation_type, link(edge.target_id))
            predictions.add(key)
            evidence[key] = edge.evidence
    return predictions, metrics(predicted_spans, set(gold_spans)), all_types, evidence


def oracle_predictions(record, sentences, starts, mentions, nlp, normalizer, scorer):
    from src.ingestion.preprocessor import SentenceChunk, build_windows
    from src.nlp.ner import Entity
    from src.nlp.coref_resolver import RuleBasedCorefResolver
    from src.nlp.relation_extractor import DepPatternExtractor
    chunks = [SentenceChunk(f"oracle:{i}", "oracle", i, text, starts[i], starts[i] + len(text))
              for i, text in enumerate(sentences)]
    by_chunk = {c.chunk_id: [] for c in chunks}
    cluster_by_id, representatives = {}, {}
    for index, m in enumerate(mentions):
        # MISC has no reliable internal type; keep unsupported mentions out.
        if m["type"] is None:
            continue
        c = chunks[m["sent_id"]]
        ent = Entity(f"oracle-mention:{index}", m["text"], f"gold-cluster:{m['cluster']}",
                     m["type"], m["local_start"], m["local_end"], "gold", c.chunk_id, c.text)
        by_chunk[c.chunk_id].append(ent)
        cluster_by_id[ent.entity_id] = str(m["cluster"])
        representatives.setdefault(m["cluster"], ent.entity_id)
    entities = {e.entity_id: e for group in by_chunk.values() for e in group}
    resolver = RuleBasedCorefResolver()
    # Gold cluster IDs are identifiers, not similar-looking person names.
    # Disable fuzzy alias discovery only in this explicitly oracle diagnostic.
    resolver._is_alias = lambda a, b: a == b
    coref = resolver.resolve(chunks, by_chunk)
    for eid, cluster in cluster_by_id.items():
        coref[eid] = representatives[int(cluster)]
    # Canonicalize pronoun targets to the supplied clusters, not fuzzy aliases.
    for key, target in list(coref.items()):
        if target in cluster_by_id:
            coref[key] = representatives[int(cluster_by_id[target])]
    predicted = set()
    extractor = DepPatternExtractor()
    windows = build_windows(chunks)
    for (window, center), doc in zip(windows, nlp.pipe(" ".join(c.text for c in w) for w, _ in windows)):
        raw = extractor.extract(doc, [e for c in window for e in by_chunk[c.chunk_id]],
                                center.chunk_id, center.sent_idx, coref, entities, window)
        for triple in raw:
            normalized = normalizer.normalize(triple)
            if normalized is None or normalized.relation_type not in {v[0] for v in RELATIONS.values()}:
                continue
            # Production retains all canonical relations irrespective of score.
            h = cluster_by_id.get(normalized.subject.entity_id)
            t = cluster_by_id.get(normalized.object.entity_id)
            if h is not None and t is not None and h != t:
                predicted.add((h, normalized.relation_type, t))
    return predicted


def evaluate(args):
    snapshot, dataset, output = (p.resolve() for p in (args.snapshot, args.dataset, args.output))
    sys.path.insert(0, str(snapshot))
    os.chdir(snapshot)
    # Keep benchmark settings reproducible and independent of the user's .env.
    from src.config import Settings
    import src.config as config
    config._settings = Settings(_env_file=None, window_size=3, window_stride=2,
                                resolution_threshold=0.85, min_related_confidence=0.60,
                                relations_config_path=str(snapshot / "config/relations.yaml"))
    from src.pipeline import run_pipeline_nlp
    from src.ingestion.preprocessor import get_nlp
    from src.nlp.predicate_normalizer import PredicateNormalizer
    from src.nlp.relation_scorer import RelationScorer
    nlp = get_nlp()
    normalizer = PredicateNormalizer(str(snapshot / "config/relations.yaml"), include_extended_relations=True)
    scorer = RelationScorer(str(snapshot / "config/relations.yaml"))
    records = json.loads(dataset.read_text(encoding="utf-8"))
    if args.limit:
        records = records[:args.limit]
    counts = {mode: defaultdict(Counter) for mode in ("end_to_end", "gold_entities")}
    mention_counts, prediction_types, rows, errors = Counter(), Counter(), [], []
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    progress = output.with_suffix(".progress.json")
    for index, record in enumerate(records):
        text, sentences, starts, mentions = reconstruct(record)
        gold = gold_facts(record)
        try:
            result = run_pipeline_nlp(text.encode(), f"external-{index}.txt", "txt")
            predicted, ner_counts, types, evidence = graph_predictions(result, mentions)
            mention_counts.update(ner_counts)
            prediction_types.update(types)
            oracle = oracle_predictions(record, sentences, starts, mentions, nlp, normalizer, scorer)
        except Exception as exc:
            # A failed document must not silently disappear from the denominator.
            errors.append({"index": index, "title": record["title"], "error": str(exc)})
            predicted, oracle, evidence = set(), set(), {}
        row = {"index": index, "title": record["title"], "gold": sorted(gold)}
        for mode, predictions in (("end_to_end", predicted), ("gold_entities", oracle)):
            counts[mode]["all_five"].update(metrics(predictions, gold))
            counts[mode]["shared_three"].update(metrics(
                {f for f in predictions if f[1] in SHARED}, {f for f in gold if f[1] in SHARED}))
            for relation in sorted({v[0] for v in RELATIONS.values()}):
                counts[mode][relation].update(metrics({f for f in predictions if f[1] == relation},
                                                     {f for f in gold if f[1] == relation}))
            row[mode] = {"predicted": sorted(predictions), "false_positives": sorted(predictions - gold),
                         "false_negatives": sorted(gold - predictions), **metrics(predictions, gold)}
        row["false_positive_evidence"] = [{"fact": list(f), "text": evidence.get(f, "")}
                                           for f in sorted(predicted - gold)]
        rows.append(row)
        if (index + 1) % 25 == 0 or index == len(records) - 1:
            status = {"completed": index + 1, "total": len(records), "errors": len(errors),
                      "elapsed_seconds": round(time.monotonic() - started, 1)}
            progress.write_text(json.dumps(status), encoding="utf-8")
            print(json.dumps(status), flush=True)
    report = {
        "dataset": "Re-DocRED dev_revised", "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "snapshot": str(snapshot), "documents": len(records),
        "all_dataset_labels": sum(len(r["labels"]) for r in records),
        "mapped_gold_facts": sum(len(r["gold"]) for r in rows),
        "mapping": RELATIONS, "shared_types": sorted(SHARED),
        "settings": {"window_size": 3, "window_stride": 2, "resolution_threshold": 0.85},
        "spacy_model": nlp.meta.get("name"), "spacy_model_version": nlp.meta.get("version"),
        "metrics": {mode: {key: summarize(value) for key, value in groups.items()} for mode, groups in counts.items()},
        "entity_mentions_exact_typed": summarize(mention_counts),
        "all_predicted_relation_counts": dict(prediction_types), "errors": errors,
        "elapsed_seconds": round(time.monotonic() - started, 1), "documents_detail": rows,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"end_to_end": report["metrics"]["end_to_end"]["all_five"],
                      "gold_entities": report["metrics"]["gold_entities"]["all_five"], "errors": len(errors)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Smoke tests only; final runs use every document")
    evaluate(parser.parse_args())
