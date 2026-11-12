"""Import and validate reviewed finance annotations for Milestone C.

Performs strict schema validation, offset checks, ontology verification, exact/near
deduplication, grouped splitting, loss-mask dataset generation, and manifest generation.
Does not replace unknown labels with supervised zeros.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.nlp.contracts_v2 import ReviewedDocumentPayload
from src.nlp.deduplication import analyze_duplicates_and_groups, compute_content_hash
from src.nlp.dataset_v2 import finance_split


def load_input_documents(input_path: Path) -> List[Dict[str, Any]]:
    """Load JSON, JSONL, or directory of annotation files."""
    records = []
    if input_path.is_file():
        text = input_path.read_text(encoding="utf-8")
        if input_path.suffix.lower() == ".jsonl":
            for line in text.splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        else:
            data = json.loads(text)
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                records.append(data)
    elif input_path.is_dir():
        for file in sorted(input_path.glob("*.json*")):
            records.extend(load_input_documents(file))
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")
    return records


def validate_ontology_constraints(
    doc: ReviewedDocumentPayload,
    ontology: Dict[str, Any],
) -> None:
    """Validate labels and ordered types; same-type semantic direction needs review."""
    ontology_relations = ontology.get("relations", {})
    mentions_by_id = {m.id: m for m in doc.mentions}

    for ann in doc.relation_annotations:
        rel = ann.relation
        if rel not in ontology_relations:
            raise ValueError(f"Unknown relation '{rel}' in document {doc.document_id}")

        rel_cfg = ontology_relations[rel]
        sub = mentions_by_id[ann.subject_mention_id]
        obj = mentions_by_id[ann.object_mention_id]

        # Check entity type pairs for positive decisions
        if ann.decision == "positive":
            allowed_pairs = rel_cfg.get("entity_type_pairs", [])
            if [sub.type, obj.type] not in allowed_pairs and (sub.type, obj.type) not in allowed_pairs:
                raise ValueError(
                    f"Invalid entity types for relation {rel}: [{sub.type}, {obj.type}] not in {allowed_pairs}"
                )


def convert_reviewed_document(
    doc_dict: Dict[str, Any],
    group_key: str,
    ontology: Dict[str, Any],
    allow_fixtures: bool = False,
) -> Dict[str, Any]:
    """Convert to 2.dataset.2 with relation-scoped status and supervision metadata."""
    doc = ReviewedDocumentPayload.model_validate(doc_dict)
    validate_ontology_constraints(doc, ontology)
    if doc.data_origin == 'test_fixture' and not allow_fixtures:
        raise ValueError('Test fixtures require --allow-fixtures and are not human-reviewed gold')
    eligible = doc.review_status in {'reviewed', 'adjudicated'} and (
        doc.data_origin == 'human_reviewed_source' or (allow_fixtures and doc.data_origin == 'test_fixture'))

    relations = list(ontology["relations"])
    content_hash = compute_content_hash(doc.text)
    mentions_by_id = {m.id: m for m in doc.mentions}

    # Group mentions by their ID (each mention has an individual ID, or group if mapped)
    annotated_groups = {m.id: [m.id] for m in doc.mentions}

    # Map reviewed positive and negative decisions
    positives: Set[Tuple[str, str, str]] = set()
    reviewed_negatives: Set[Tuple[str, str, str]] = set()
    annotations_by_pair: Dict[Tuple[str, str, str], Any] = {}

    for ann in doc.relation_annotations:
        key = (ann.subject_mention_id, ann.relation, ann.object_mention_id)
        if not eligible or ann.review_status not in {'reviewed', 'adjudicated'}:
            continue
        annotations_by_pair[key] = ann
        if ann.decision == "positive":
            positives.add(key)
        elif ann.decision == "explicit_negative":
            reviewed_negatives.add(key)

    candidates = []
    mention_ids = [m.id for m in doc.mentions]

    for sub_id in mention_ids:
        sub_type = mentions_by_id[sub_id].type
        for obj_id in mention_ids:
            if sub_id == obj_id:
                continue
            obj_type = mentions_by_id[obj_id].type

            # Check if any ontology relation supports this type pair or if explicitly annotated
            supported = any(
                [sub_type, obj_type] in cfg.get("entity_type_pairs", [])
                for cfg in ontology["relations"].values()
            )
            has_annotation = any(
                (sub_id, r, obj_id) in annotations_by_pair for r in relations
            )

            if not supported and not has_annotation:
                continue

            targets = [None] * len(relations)
            assertion_by_relation = {}
            supervision_by_relation = {}
            for i, rel in enumerate(relations):
                key = (sub_id, rel, obj_id)
                ann = annotations_by_pair.get(key)
                if key in positives:
                    targets[i] = 1
                    attributes = {name: getattr(ann, name) if getattr(ann, name) != 'unknown' else None
                                  for name in ('polarity', 'modality', 'lifecycle')}
                    assertion_by_relation[rel] = {
                        'targets': attributes,
                        'loss_mask': {name: value is not None for name, value in attributes.items()},
                        'time_qualifiers': ann.time_qualifiers,
                    }
                elif key in reviewed_negatives:
                    targets[i] = 0
                if targets[i] is not None:
                    supervision_by_relation[rel] = {
                        'annotation_id': ann.annotation_id, 'reviewer': ann.reviewer,
                        'review_status': ann.review_status, 'evidence_spans': ann.evidence_spans,
                        'evidence_text': ann.evidence_text, 'notes': ann.notes,
                        'guidelines_version': doc.review_guidelines_version,
                        'review_timestamp': doc.review_timestamp,
                    }
            candidates.append({
                'subject_group': sub_id, 'object_group': obj_id, 'type_supported': supported,
                'targets': targets, 'loss_mask': [t is not None for t in targets],
                'assertions_by_relation': assertion_by_relation,
                'supervision_by_relation': supervision_by_relation,
            })

    output_mentions = [
        {
            "id": m.id,
            "document_id": doc.document_id,
            "text": m.text,
            "start": m.start,
            "end": m.end,
            "type": m.type,
            "sentence_idx": m.sentence_idx,
            "detector": m.detector,
        }
        for m in doc.mentions
    ]

    return {
        "schema_version": "2.dataset.2",
        "document_id": doc.document_id,
        "source": doc.license_or_rights,
        "title": doc.title,
        "doc_group": group_key,
        "content_hash": content_hash,
        "text": doc.text,
        "mentions": output_mentions,
        "annotated_groups": annotated_groups,
        "gold_relations": sorted(list(positives)),
        "reviewed_negatives": sorted(list(reviewed_negatives)),
        "candidates": candidates,
        "reviewer_provenance": {
            "reviewer": doc.reviewer,
            "guidelines_version": doc.review_guidelines_version,
            "timestamp": doc.review_timestamp,
            "scope": doc.annotation_scope,
            "license": doc.license_or_rights,
        },
        "supervision_scope": "mention_pair_explicit_review",
        "data_origin": doc.data_origin,
        "document_review_status": doc.review_status,
        "original_annotations": [a.model_dump() for a in doc.relation_annotations],
    }


def import_finance_data(
    input_path: Path,
    ontology_path: Path,
    output_dir: Path,
    near_dedup_threshold: float = 0.85,
    seed: str = "doc2graph-v2-finance-20261111",
    allow_fixtures: bool = False,
) -> Dict[str, Any]:
    """Execute the full import, validation, deduplication, and export pipeline."""
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    ontology = yaml.safe_load(ontology_path.read_text(encoding="utf-8"))
    raw_records = load_input_documents(input_path)
    if not raw_records:
        raise ValueError(f"No documents found at input path: {input_path}")

    # Validate all raw documents before processing
    validated_raw = []
    for r in raw_records:
        doc = ReviewedDocumentPayload.model_validate(r)
        validate_ontology_constraints(doc, ontology)
        if doc.data_origin == 'test_fixture' and not allow_fixtures:
            raise ValueError('Test fixtures require --allow-fixtures and are not human-reviewed gold')
        validated_raw.append(doc.model_dump())
    ids = [r['document_id'] for r in validated_raw]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate document IDs are not allowed')

    # Run deduplication and group clustering
    kept_records, exclusions, group_map = analyze_duplicates_and_groups(
        validated_raw, near_threshold=near_dedup_threshold
    )

    output_dir.mkdir(parents=True)
    splits = ("train", "validation", "calibration", "locked_test")
    handles = {name: (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") for name in splits}

    counts = Counter()
    assignments = []
    reviewer_counts = Counter()

    try:
        for r in kept_records:
            doc_id = r["document_id"]
            group_key = group_map.get(doc_id, r.get("doc_group", doc_id))
            split = finance_split(group_key, seed=seed)

            converted = convert_reviewed_document(r, group_key, ontology, allow_fixtures=allow_fixtures)
            handles[split].write(json.dumps(converted, ensure_ascii=False) + "\n")

            counts[split] += 1
            reviewer_counts[r.get("reviewer", "unassigned")] += 1

            for candidate in converted["candidates"]:
                counts["candidates"] += 1
                counts["positive_labels"] += candidate["targets"].count(1)
                counts["explicit_negative_labels"] += candidate["targets"].count(0)
                counts["unknown_labels"] += candidate["targets"].count(None)

            assignments.append({
                "document_id": doc_id,
                "split": split,
                "group_key": group_key,
                "content_hash": converted["content_hash"],
            })
    finally:
        for h in handles.values():
            h.close()

    source_hashes = {}
    if input_path.is_file():
        source_hashes[input_path.name] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    else:
        for f in sorted(input_path.glob("*.json*")):
            source_hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()

    manifest = {
        "schema_version": "2.dataset.2",
        "dataset_name": "reviewed_finance_v2",
        "seed": seed,
        "near_dedup_threshold": near_dedup_threshold,
        "label_order": list(ontology["relations"]),
        "counts": dict(counts),
        "source_hashes": source_hashes,
        "ontology_sha256": hashlib.sha256(ontology_path.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output_hashes": {
            name: hashlib.sha256((output_dir / f"{name}.jsonl").read_bytes()).hexdigest()
            for name in splits
        },
        "exclusions": exclusions,
        "group_membership_including_exclusions": group_map,
        "assignments": assignments,
        "reviewer_summary": dict(reviewer_counts),
        "split_proportions": {"train": .60, "validation": .15, "calibration": .15, "locked_test": .10},
        "locked_test_sealed": False,
        "release_holdout_available": False,
        "contains_test_fixtures": any(r['data_origin'] == 'test_fixture' for r in validated_raw),
        "review_provenance_verified": False,
        "dependency_hashes": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                              for name in ('src/nlp/contracts_v2.py', 'src/nlp/dataset_v2.py', 'src/nlp/deduplication.py')},
        "limitations": [
            "Any labels with review_status 'pending_human_review' are proposed drafts, not independent test truth.",
            "All unreviewed pairs are explicitly masked (target=None, loss_mask=False).",
            "Near-duplicate documents share the same group_key and partition.",
            "The locked_test partition is only an assignment: plaintext files are not sealed or access-controlled.",
            "Reviewer fields are recorded attestations, not independently verified human-review provenance.",
            "Test fixtures are for tooling validation only, never training or independent evaluation truth.",
            "Same-type argument direction and financial semantics still require human review.",
        ],
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to input JSON/JSONL file or directory")
    parser.add_argument("--ontology", type=Path, default=Path("config/ontology_v2.yaml"), help="Ontology YAML")
    parser.add_argument("--output", type=Path, required=True, help="Output directory (must not exist)")
    parser.add_argument("--near-dedup-threshold", type=float, default=0.85, help="Near-dedup Jaccard threshold")
    parser.add_argument("--seed", type=str, default="doc2graph-v2-finance-20261111", help="Grouping seed")
    parser.add_argument('--allow-fixtures', action='store_true', help='Tooling tests only; output is not gold data')
    args = parser.parse_args()

    manifest = import_finance_data(
        input_path=args.input,
        ontology_path=args.ontology,
        output_dir=args.output,
        near_dedup_threshold=args.near_dedup_threshold,
        seed=args.seed,
        allow_fixtures=args.allow_fixtures,
    )
    print(json.dumps({"status": "imported", "counts": manifest["counts"], "exclusions": len(manifest["exclusions"])}))
