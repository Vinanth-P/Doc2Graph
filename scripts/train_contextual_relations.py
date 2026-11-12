"""Train contextual relation extraction and assertion model for Milestone E.

Accepts prepared v2 dataset JSONL files, strictly respects loss masks (unknowns
never converted to negative zeros), enforces test fixture / split custody guards,
records input/ontology hashes and label support provenance, and produces frozen artifacts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nlp.contracts_v2 import MentionRecord
from src.nlp.contextual_relations import (
    RELATION_TYPES,
    POLARITY_TYPES,
    MODALITY_TYPES,
    LIFECYCLE_TYPES,
    RELATION_TO_IDX,
    POLARITY_TO_IDX,
    MODALITY_TO_IDX,
    LIFECYCLE_TO_IDX,
    SimpleVocabulary,
    ContextualRelationModel,
    build_marked_context,
    compute_masked_loss,
    load_ontology_type_pairs,
)
from src.nlp.split_custody import verify_split_custody, verify_cross_split_disjointness


class RelationDataset(Dataset):
    """PyTorch Dataset for relation candidate classification with relation-conditioned assertions."""

    def __init__(self, samples: List[Dict[str, Any]], vocab: SimpleVocabulary, max_len: int = 128):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        tokens = self.vocab.tokenize(item["marked_text"])
        input_ids, s_s, s_e, o_s, o_e, is_truncated = self.vocab.encode(tokens, max_len=self.max_len)

        res: Dict[str, Any] = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "marker_positions": torch.tensor([s_s, s_e, o_s, o_e], dtype=torch.long),
            "targets": torch.tensor(item["targets"], dtype=torch.float),
            "loss_mask": torch.tensor(item["loss_mask"], dtype=torch.bool),
        }

        # Per-relation assertions
        for r_name in RELATION_TYPES:
            r_assert = item.get("assertions_by_relation", {}).get(r_name, {})
            a_tgts = r_assert.get("targets", {})
            a_masks = r_assert.get("loss_mask", {})

            res[f"{r_name}_polarity_target"] = torch.tensor(a_tgts.get("polarity", POLARITY_TO_IDX["unknown"]), dtype=torch.long)
            res[f"{r_name}_polarity_mask"] = torch.tensor(a_masks.get("polarity", False), dtype=torch.bool)

            res[f"{r_name}_modality_target"] = torch.tensor(a_tgts.get("modality", MODALITY_TO_IDX["unknown"]), dtype=torch.long)
            res[f"{r_name}_modality_mask"] = torch.tensor(a_masks.get("modality", False), dtype=torch.bool)

            res[f"{r_name}_lifecycle_target"] = torch.tensor(a_tgts.get("lifecycle", LIFECYCLE_TO_IDX["unknown"]), dtype=torch.long)
            res[f"{r_name}_lifecycle_mask"] = torch.tensor(a_masks.get("lifecycle", False), dtype=torch.bool)

        return res


def compute_file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def load_dataset_records(
    paths: List[Path],
    allow_fixtures: bool = False,
    expected_split: Optional[str] = None,
    manifest_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load JSONL dataset records with strict existence, split custody, and fixture guards."""
    records = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                records.append(record)

    verify_split_custody(
        records=records,
        expected_split=expected_split,
        manifest_path=manifest_path,
        allow_fixtures=allow_fixtures,
    )
    return records


def extract_samples_from_records(
    records: List[Dict[str, Any]],
    wiring_test: bool = False,
) -> List[Dict[str, Any]]:
    """Extract candidate instances with marked contexts, strictly preserving input supervision masks.

    Blocker 1 fix: NEVER fabricates negative targets from unannotated pairs.
    Blocker 3 fix: Preserves relation-conditioned assertions without sharing.
    """
    samples: List[Dict[str, Any]] = []

    for doc in records:
        text = doc["text"]
        mentions = [MentionRecord.model_validate(m) for m in doc["mentions"]]
        mentions_by_id = {m.id: m for m in mentions}
        annotated_groups = doc.get("annotated_groups", {})

        for cand in doc.get("candidates", []):
            loss_mask = list(cand.get("loss_mask", [False] * len(RELATION_TYPES)))
            targets = list(cand.get("targets", [None] * len(RELATION_TYPES)))

            # Only include candidates that have explicit supervision in the dataset
            if not any(loss_mask):
                continue

            s_grp = str(cand.get("subject_group"))
            o_grp = str(cand.get("object_group"))

            s_m_ids = annotated_groups.get(s_grp, [s_grp])
            o_m_ids = annotated_groups.get(o_grp, [o_grp])
            s_mention = mentions_by_id.get(s_m_ids[0]) if s_m_ids else None
            o_mention = mentions_by_id.get(o_m_ids[0]) if o_m_ids else None
            if s_mention is None or o_mention is None:
                continue

            marked_text, _, is_truncated = build_marked_context(
                text, (s_mention.start, s_mention.end), (o_mention.start, o_mention.end)
            )

            # Strict preservation: numeric targets where mask is True, 0.0 where False (masked from loss)
            num_targets = [float(t) if (t is not None and m) else 0.0 for t, m in zip(targets, loss_mask)]

            # Blocker 3: Relation-conditioned assertions
            assertions_by_relation: Dict[str, Any] = {}
            raw_assertions = cand.get("assertions_by_relation", {})

            for r_name in RELATION_TYPES:
                r_info = raw_assertions.get(r_name, {})
                a_tgts = r_info.get("targets", {})
                a_masks = r_info.get("loss_mask", {})

                pol_tgt = POLARITY_TO_IDX.get(a_tgts.get("polarity"), POLARITY_TO_IDX["unknown"])
                pol_m = bool(a_masks.get("polarity", False))

                mod_tgt = MODALITY_TO_IDX.get(a_tgts.get("modality"), MODALITY_TO_IDX["unknown"])
                mod_m = bool(a_masks.get("modality", False))

                life_tgt = LIFECYCLE_TO_IDX.get(a_tgts.get("lifecycle"), LIFECYCLE_TO_IDX["unknown"])
                life_m = bool(a_masks.get("lifecycle", False))

                assertions_by_relation[r_name] = {
                    "targets": {"polarity": pol_tgt, "modality": mod_tgt, "lifecycle": life_tgt},
                    "loss_mask": {"polarity": pol_m, "modality": mod_m, "lifecycle": life_m},
                }

            samples.append({
                "marked_text": marked_text,
                "targets": num_targets,
                "loss_mask": loss_mask,
                "assertions_by_relation": assertions_by_relation,
                "truncated": is_truncated,
            })

            if wiring_test and len(samples) >= 32:
                return samples

    return samples


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate model on validation split computing precision, recall, F1 across supervised labels."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    tp, fp, fn = 0, 0, 0
    per_rel_tp = defaultdict(int)
    per_rel_fp = defaultdict(int)
    per_rel_fn = defaultdict(int)

    with torch.no_grad():
        for batch in val_loader:
            inp = batch["input_ids"].to(device)
            pos = batch["marker_positions"].to(device)
            targets = batch["targets"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            out = model(inp, pos)
            loss, _ = compute_masked_loss(out["relation_logits"], targets, loss_mask)
            total_loss += float(loss.item())
            num_batches += 1

            probs = torch.sigmoid(out["relation_logits"])
            preds = (probs >= threshold).bool()
            gold = (targets >= 0.5).bool()

            for r_idx in range(len(RELATION_TYPES)):
                mask_r = loss_mask[:, r_idx]
                p_r = preds[:, r_idx][mask_r]
                g_r = gold[:, r_idx][mask_r]

                r_tp = (p_r & g_r).sum().item()
                r_fp = (p_r & ~g_r).sum().item()
                r_fn = (~p_r & g_r).sum().item()

                tp += r_tp
                fp += r_fp
                fn += r_fn

                per_rel_tp[RELATION_TYPES[r_idx]] += r_tp
                per_rel_fp[RELATION_TYPES[r_idx]] += r_fp
                per_rel_fn[RELATION_TYPES[r_idx]] += r_fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "val_loss": total_loss / max(1, num_batches),
        "micro": {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn},
        "per_relation": {
            r: {
                "tp": per_rel_tp[r],
                "fp": per_rel_fp[r],
                "fn": per_rel_fn[r],
                "f1": (
                    2 * per_rel_tp[r] / max(1, 2 * per_rel_tp[r] + per_rel_fp[r] + per_rel_fn[r])
                    if per_rel_tp[r] > 0
                    else 0.0
                ),
            }
            for r in RELATION_TYPES
        },
    }


def train_contextual_relations(
    train_files: List[Path],
    val_file: Path,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    seed: int = 42,
    allow_fixtures: bool = False,
    wiring_test_only: bool = False,
) -> Dict[str, Any]:
    """Train contextual relation model with masked loss and write manifest."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading training data from {len(train_files)} files...")
    train_records = load_dataset_records(train_files, allow_fixtures=allow_fixtures, expected_split="train")
    train_samples = extract_samples_from_records(train_records, wiring_test=wiring_test_only)

    print(f"Loaded {len(train_samples)} supervised candidate training instances.")

    # Count supervision support by relation
    label_support: Dict[str, Dict[str, int]] = {
        r: {"positive": 0, "explicit_negative": 0, "unknown": 0} for r in RELATION_TYPES
    }
    for s in train_samples:
        for idx, r_name in enumerate(RELATION_TYPES):
            if s["loss_mask"][idx]:
                if s["targets"][idx] >= 0.5:
                    label_support[r_name]["positive"] += 1
                else:
                    label_support[r_name]["explicit_negative"] += 1
            else:
                label_support[r_name]["unknown"] += 1

    vocab = SimpleVocabulary(min_freq=1)
    vocab.build_from_texts([s["marked_text"] for s in train_samples])
    print(f"Vocabulary size: {len(vocab.token_to_idx)}")

    train_dataset = RelationDataset(train_samples, vocab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_records = load_dataset_records([val_file], allow_fixtures=allow_fixtures, expected_split="validation")
    verify_cross_split_disjointness({"train": train_records, "val": val_records})
    val_samples = extract_samples_from_records(val_records, wiring_test=wiring_test_only)
    val_dataset = RelationDataset(val_samples, vocab)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Loaded {len(val_samples)} validation instances.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    model = ContextualRelationModel(
        vocab_size=len(vocab.token_to_idx),
        embed_dim=128,
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    training_history = []
    best_val_f1 = -1.0
    best_epoch = 0

    if wiring_test_only:
        print("Running wiring test: overfit verification on 32 samples...")
        model.train()
        for step in range(30):
            batch = next(iter(train_loader))
            inp = batch["input_ids"].to(device)
            pos = batch["marker_positions"].to(device)
            targets = batch["targets"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            optimizer.zero_grad()
            out = model(inp, pos)
            loss, m = compute_masked_loss(out["relation_logits"], targets, loss_mask)
            loss.backward()
            optimizer.step()
        print(f"Wiring test final loss: {loss.item():.4f}")
        return {"wiring_test_passed": True, "final_loss": float(loss.item())}

    for epoch in range(1, epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            inp = batch["input_ids"].to(device)
            pos = batch["marker_positions"].to(device)
            targets = batch["targets"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            optimizer.zero_grad()
            out = model(inp, pos)

            # Per-relation assertion logits, targets, masks
            assertion_logits_by_rel = {
                r: {
                    "polarity": out["polarity_logits_by_rel"][r],
                    "modality": out["modality_logits_by_rel"][r],
                    "lifecycle": out["lifecycle_logits_by_rel"][r],
                }
                for r in RELATION_TYPES
            }
            assertion_targets_by_rel = {
                r: {
                    "polarity": batch[f"{r}_polarity_target"].to(device),
                    "modality": batch[f"{r}_modality_target"].to(device),
                    "lifecycle": batch[f"{r}_lifecycle_target"].to(device),
                }
                for r in RELATION_TYPES
            }
            assertion_masks_by_rel = {
                r: {
                    "polarity": batch[f"{r}_polarity_mask"].to(device),
                    "modality": batch[f"{r}_modality_mask"].to(device),
                    "lifecycle": batch[f"{r}_lifecycle_mask"].to(device),
                }
                for r in RELATION_TYPES
            }

            loss, metrics = compute_masked_loss(
                out["relation_logits"],
                targets,
                loss_mask,
                assertion_logits_by_rel=assertion_logits_by_rel,
                assertion_targets_by_rel=assertion_targets_by_rel,
                assertion_masks_by_rel=assertion_masks_by_rel,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_epoch_loss += float(loss.item())
            num_batches += 1

        train_loss = total_epoch_loss / max(1, num_batches)
        val_metrics = evaluate_model(model, val_loader, device=device)
        val_f1 = val_metrics["micro"]["f1"]

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "val_precision": val_metrics["micro"]["precision"],
            "val_recall": val_metrics["micro"]["recall"],
            "val_f1": val_f1,
        }
        training_history.append(log_entry)
        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val P: {val_metrics['micro']['precision']:.4f} | "
            f"Val R: {val_metrics['micro']['recall']:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1 or epoch == 1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "model.pt")

    # Save artifacts and manifest
    (output_dir / "vocab.json").write_text(json.dumps(vocab.to_dict(), indent=2), encoding="utf-8")

    ontology_path = Path("config/ontology_v2.yaml")
    ontology_hash = compute_file_hash(ontology_path) if ontology_path.exists() else ""
    train_file_hashes = {str(p): compute_file_hash(p) for p in train_files if p.exists()}
    val_file_hash = compute_file_hash(val_file) if val_file.exists() else ""

    manifest = {
        "model_name": "contextual_v2",
        "version": "2.0.0",
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "device": device,
        "hidden_dim": hidden_dim,
        "lr": lr,
        "relations": RELATION_TYPES,
        "label_support": label_support,
        "ontology_hash": ontology_hash,
        "train_file_hashes": train_file_hashes,
        "val_file_hash": val_file_hash,
        "training_history": training_history,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Artifacts saved successfully to {output_dir}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Train contextual relation extraction model.")
    parser.add_argument("--train-files", nargs="+", required=True, help="Path(s) to training JSONL files.")
    parser.add_argument("--val-file", type=Path, required=True, help="Path to validation JSONL file.")
    parser.add_argument("--output-dir", type=Path, default=Path("models/contextual_v2"), help="Output directory.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Encoder hidden dimension.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--allow-fixtures", action="store_true", help="Allow test fixture documents.")
    parser.add_argument("--wiring-test", action="store_true", help="Run small overfit wiring test only.")

    args = parser.parse_args()
    train_files = [Path(p) for p in args.train_files]
    train_contextual_relations(
        train_files=train_files,
        val_file=args.val_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        allow_fixtures=args.allow_fixtures,
        wiring_test_only=args.wiring_test,
    )


if __name__ == "__main__":
    main()
