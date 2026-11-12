"""Fit per-relation calibration thresholds and evaluate reliability for Milestone F.

Fits thresholds exclusively on the calibration split without touching locked test.
Preserves supervision masks: never turns unknown pairs into negative zeros.
Disables unsupported relations and relations failing target precision.
Records dataset sha256, model sha256, and supervision provenance.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nlp.contracts_v2 import MentionRecord
from src.nlp.contextual_relations import (
    RELATION_TYPES,
    RELATION_TO_IDX,
    SimpleVocabulary,
    ContextualRelationModel,
    generate_directional_candidates,
    load_ontology_type_pairs,
)
from src.nlp.split_custody import verify_split_custody


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_calibration_records(calibration_path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(calibration_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_calibration(
    model_dir: Path,
    calibration_file: Path,
    output_path: Optional[Path] = None,
    target_precision: float = 0.70,
    min_support: int = 5,
    min_negative_support: int = 5,
    allow_fixtures: bool = False,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fit probability thresholds per relation on the calibration split."""
    cal_name = calibration_file.name.lower()
    if "test" in cal_name and "calibration" not in cal_name:
        raise ValueError(f"CRITICAL: Calibration script must never access test split: {calibration_file}")

    if not calibration_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")

    if not (model_dir / "model.pt").exists():
        raise FileNotFoundError(f"Model file not found: {model_dir / 'model.pt'}")

    model_sha256 = file_sha256(model_dir / "model.pt")
    cal_file_sha256 = file_sha256(calibration_file)

    print(f"Loading model and vocab from {model_dir}...")
    vocab_data = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    vocab = SimpleVocabulary.from_dict(vocab_data)

    manifest = json.loads((model_dir / "training_manifest.json").read_text(encoding="utf-8"))
    hidden_dim = manifest.get("hidden_dim", 256)
    embed_dim = manifest.get("embed_dim", 128)
    num_layers = manifest.get("num_layers", 2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ContextualRelationModel(
        vocab_size=len(vocab.token_to_idx),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.0,
    )
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
    model.to(device).eval()

    ontology_type_pairs = load_ontology_type_pairs()

    print(f"Loading calibration split from {calibration_file}...")
    records = load_calibration_records(calibration_file)
    print(f"Loaded {len(records)} calibration documents.")

    # Task 5: Enforce authoritative split custody
    verify_split_custody(
        records=records,
        expected_split="calibration",
        manifest_path=manifest_path,
        allow_fixtures=allow_fixtures,
    )

    # Provenance counts: count explicit positives and explicit negatives
    explicit_negs_count = 0
    pos_count_total = 0
    for doc in records:
        for cand_rec in doc.get("candidates", []):
            for t, m in zip(cand_rec.get("targets", []), cand_rec.get("loss_mask", [])):
                if m:
                    if t == 1:
                        pos_count_total += 1
                    elif t == 0:
                        explicit_negs_count += 1

    # Fact-level pooling: (doc_id, s_grp, rel_name, o_grp) -> max prob, label
    fact_probs: Dict[Tuple[str, str, str, str], float] = {}
    fact_labels: Dict[Tuple[str, str, str, str], int] = {}

    total_candidates = 0

    with torch.no_grad():
        for doc in records:
            doc_id = doc["document_id"]
            text = doc["text"]
            mentions = [MentionRecord.model_validate(m) for m in doc["mentions"]]
            mentions_by_id = {m.id: m for m in mentions}

            mid_to_group: Dict[str, str] = {}
            for gid, mids in doc.get("annotated_groups", {}).items():
                for mid in mids:
                    mid_to_group[mid] = str(gid)

            cand_supervision: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for cand_rec in doc.get("candidates", []):
                s_g = str(cand_rec.get("subject_group"))
                o_g = str(cand_rec.get("object_group"))
                cand_supervision[(s_g, o_g)] = {
                    "targets": cand_rec.get("targets", [None] * len(RELATION_TYPES)),
                    "loss_mask": cand_rec.get("loss_mask", [False] * len(RELATION_TYPES)),
                }

            cands = generate_directional_candidates(
                document_id=doc_id,
                text=text,
                mentions=mentions,
                ontology_type_pairs=ontology_type_pairs,
            )
            if not cands:
                continue

            valid_cands = []
            batch_ids = []
            batch_markers = []
            for cand in cands:
                toks = vocab.tokenize(cand.context_text)
                ids, s_s, s_e, o_s, o_e, is_truncated = vocab.encode(toks)
                # Blocker 4: Abstain if argument markers are truncated or missing
                if s_s == -1 or s_e == -1 or o_s == -1 or o_e == -1:
                    continue
                valid_cands.append(cand)
                batch_ids.append(ids)
                batch_markers.append([s_s, s_e, o_s, o_e])

            if not valid_cands:
                continue

            total_candidates += len(valid_cands)
            inp = torch.tensor(batch_ids, dtype=torch.long, device=device)
            pos = torch.tensor(batch_markers, dtype=torch.long, device=device)

            out = model(inp, pos)
            probs = torch.sigmoid(out["relation_logits"]).cpu().numpy()

            for idx, cand in enumerate(valid_cands):
                s_mid = cand.subject_mention_id
                o_mid = cand.object_mention_id
                s_grp = mid_to_group.get(s_mid, s_mid)
                o_grp = mid_to_group.get(o_mid, o_mid)
                pair_key = (s_grp, o_grp)

                cand_info = cand_supervision.get(pair_key)

                for r_idx, rel_name in enumerate(RELATION_TYPES):
                    p = float(probs[idx, r_idx])

                    # Blocker 2: Only evaluate reviewed/supervised outcomes!
                    # If cand_info is present, check loss_mask for this relation
                    if cand_info is not None:
                        is_supervised = cand_info["loss_mask"][r_idx]
                        if not is_supervised:
                            # Unknown label! Must NOT be labeled 0 as negative!
                            continue
                        target_val = cand_info["targets"][r_idx]
                        if target_val is None:
                            continue
                        label = int(target_val)
                    else:
                        gold_facts = {(str(f[0]), str(f[1]), str(f[2])) for f in doc.get("gold_relations", [])}
                        if (s_grp, rel_name, o_grp) in gold_facts:
                            label = 1
                        else:
                            # Unknown without reviewed negative supervision: do NOT label as 0!
                            continue

                    fact_key = (doc_id, s_grp, rel_name, o_grp)
                    if fact_key not in fact_probs or p > fact_probs[fact_key]:
                        fact_probs[fact_key] = p
                        fact_labels[fact_key] = label

    scores_by_rel: Dict[str, List[float]] = defaultdict(list)
    labels_by_rel: Dict[str, List[int]] = defaultdict(list)
    for (d_id, s_g, rel_name, o_g), p in fact_probs.items():
        label = fact_labels[(d_id, s_g, rel_name, o_g)]
        scores_by_rel[rel_name].append(p)
        labels_by_rel[rel_name].append(label)

    print(f"Total calibration candidates evaluated: {total_candidates}")
    print(f"Total supervised facts evaluated: {len(fact_probs)}")

    thresholds_manifest: Dict[str, Any] = {
        "calibration_split": str(calibration_file),
        "calibration_file_sha256": cal_file_sha256,
        "model_sha256": model_sha256,
        "total_documents": len(records),
        "total_candidates": total_candidates,
        "total_supervised_facts": len(fact_probs),
        "explicit_negatives_count": explicit_negs_count,
        "explicit_negatives_available": (explicit_negs_count > 0),
        "target_precision": target_precision,
        "probability_calibration": "uncalibrated_raw_sigmoid",
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "relations": {},
    }

    grid = [round(x, 2) for x in np.linspace(0.1, 0.95, 18)]

    for rel_name in RELATION_TYPES:
        y_true = np.array(labels_by_rel[rel_name])
        y_prob = np.array(scores_by_rel[rel_name])

        pos_count = int(np.sum(y_true == 1)) if len(y_true) > 0 else 0
        neg_count = int(np.sum(y_true == 0)) if len(y_true) > 0 else 0
        total_count = len(y_true)

        brier = float(np.mean((y_prob - y_true) ** 2)) if total_count > 0 else 0.0

        if pos_count < min_support:
            # Low or zero support: enforce abstention, disable relation explicitly
            thresholds_manifest["relations"][rel_name] = {
                "threshold": 1.01,
                "disabled": True,
                "support": pos_count,
                "negative_support": neg_count,
                "total_instances": total_count,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "brier_score": brier,
                "status": "unsupported_disabled",
                "abstention_policy": "enforced_abstention",
            }
            continue

        # Task 3: Require reviewed negative support for precision qualification
        if neg_count < min_negative_support:
            thresholds_manifest["relations"][rel_name] = {
                "threshold": 1.01,
                "disabled": True,
                "support": pos_count,
                "negative_support": neg_count,
                "total_instances": total_count,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "brier_score": brier,
                "status": "uncalibrated_insufficient_negatives",
                "abstention_policy": "abstain_insufficient_negatives",
                "note": (
                    f"Relation has {pos_count} positive facts but only {neg_count} reviewed negatives "
                    f"(< {min_negative_support}). Cannot certify precision without reviewed negative supervision."
                ),
            }
            continue

        best_thresh = 0.50
        best_f1 = -1.0
        best_p = 0.0
        best_r = 0.0

        target_met_thresh = None
        target_met_p = 0.0
        target_met_r = 0.0
        target_met_f1 = 0.0

        for thresh in grid:
            preds = (y_prob >= thresh).astype(int)
            tp = int(np.sum((preds == 1) & (y_true == 1)))
            fp = int(np.sum((preds == 1) & (y_true == 0)))
            fn = int(np.sum((preds == 0) & (y_true == 1)))

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                best_p = p
                best_r = r

            if p >= target_precision and r > 0.0:
                if target_met_thresh is None or f1 > target_met_f1:
                    target_met_thresh = thresh
                    target_met_p = p
                    target_met_r = r
                    target_met_f1 = f1

        if target_met_thresh is not None:
            thresholds_manifest["relations"][rel_name] = {
                "threshold": float(target_met_thresh),
                "disabled": False,
                "support": pos_count,
                "total_instances": total_count,
                "precision": float(target_met_p),
                "recall": float(target_met_r),
                "f1": float(target_met_f1),
                "brier_score": brier,
                "status": "calibrated",
                "abstention_policy": "predict_above_threshold",
            }
        else:
            # Target precision not met: DO NOT claim calibrated or fall back to max F1!
            thresholds_manifest["relations"][rel_name] = {
                "threshold": 1.01,
                "disabled": True,
                "support": pos_count,
                "total_instances": total_count,
                "precision": float(best_p),
                "recall": float(best_r),
                "f1": float(best_f1),
                "brier_score": brier,
                "status": "uncalibrated_failed_target",
                "abstention_policy": "abstain_precision_not_met",
            }

    if output_path is None:
        output_path = model_dir / "thresholds.json"

    output_path.write_text(json.dumps(thresholds_manifest, indent=2), encoding="utf-8")
    print(f"Calibration thresholds successfully written to {output_path}")

    # Print summary table
    print("\nCalibration Results Summary:")
    print(f"{'Relation':<18} {'Thresh':<8} {'Support':<8} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Status'}")
    print("-" * 75)
    for r, info in thresholds_manifest["relations"].items():
        print(
            f"{r:<18} {info['threshold']:<8.2f} {info['support']:<8} "
            f"{info['precision']:<10.3f} {info['recall']:<8.3f} {info['f1']:<8.3f} {info['status']}"
        )

    return thresholds_manifest


def main():
    parser = argparse.ArgumentParser(description="Calibrate relation probability thresholds.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/contextual_v2"), help="Path to model directory.")
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=Path("data/prepared/v2_redocred/calibration.jsonl"),
        help="Path to calibration split JSONL.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path.")
    parser.add_argument("--target-precision", type=float, default=0.70, help="Target minimum precision.")
    parser.add_argument("--min-support", type=int, default=5, help="Minimum support for calibrated status.")
    parser.add_argument("--allow-fixtures", action="store_true", help="Allow test fixtures in input.")

    args = parser.parse_args()
    run_calibration(
        model_dir=args.model_dir,
        calibration_file=args.calibration_file,
        output_path=args.output,
        target_precision=args.target_precision,
        min_support=args.min_support,
        allow_fixtures=args.allow_fixtures,
    )


if __name__ == "__main__":
    main()
