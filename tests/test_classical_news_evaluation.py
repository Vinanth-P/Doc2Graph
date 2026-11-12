"""Unit tests for classical news evaluation and dataset split integrity."""
import json
from pathlib import Path
import pytest

from scripts.evaluate_classical_news import compute_prf1, audit_feasibility_gates


def test_compute_prf1_edge_cases():
    # Zero counts
    p, r, f1 = compute_prf1(0, 0, 0)
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0

    # Perfect counts
    p, r, f1 = compute_prf1(10, 0, 0)
    assert p == 100.0
    assert r == 100.0
    assert f1 == 100.0

    # Standard counts
    p, r, f1 = compute_prf1(8, 2, 2)  # TP=8, FP=2 (P=80%), FN=2 (R=80%)
    assert p == 80.0
    assert r == 80.0
    assert f1 == 80.0


def test_audit_feasibility_gates():
    passing_summary = {
        "ner": {"micro": {"f1": 88.5}},
        "relations_end_to_end": {"micro": {"precision": 85.0, "recall": 60.0, "f1": 70.0}},
        "silent_inference_failures": 0,
    }
    audit = audit_feasibility_gates(passing_summary)
    assert audit["all_gates_passed"] is True
    assert audit["release_recommendation"] == "PROMOTE_CLASSICAL_NEWS"

    failing_summary = {
        "ner": {"micro": {"f1": 80.0}},  # Fails < 85%
        "relations_end_to_end": {"micro": {"precision": 75.0, "recall": 45.0, "f1": 55.0}},
        "silent_inference_failures": 1,
    }
    audit_fail = audit_feasibility_gates(failing_summary)
    assert audit_fail["all_gates_passed"] is False
    assert audit_fail["release_recommendation"] == "KEEP_DEFAULT_RULES"


def test_splits_group_isolation_and_disjointness():
    """Verify that document groups and text content are strictly disjoint across splits."""
    splits_dir = Path("data/classical_news/splits")
    manifest_path = splits_dir / "split_manifest.json"
    assert manifest_path.exists(), "split_manifest.json must exist"

    splits_data = {}
    group_sets = {}
    content_hash_sets = {}

    for sname in ["dev", "locked_test", "challenge"]:
        split_file = splits_dir / f"{sname}.jsonl"
        assert split_file.exists(), f"Split file {split_file} missing"

        docs = [json.loads(line) for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        splits_data[sname] = docs
        group_sets[sname] = {d["group_key"] for d in docs}
        content_hash_sets[sname] = {d["content_hash"] for d in docs}

    # Verify dev vs locked_test have NO overlapping group_keys (strict group isolation)
    dev_locked_groups = group_sets["dev"] & group_sets["locked_test"]
    assert len(dev_locked_groups) == 0, f"Cross-split group leakage between dev and locked_test: {dev_locked_groups}"

    # Verify dev vs locked_test have NO overlapping content hashes
    dev_locked_hashes = content_hash_sets["dev"] & content_hash_sets["locked_test"]
    assert len(dev_locked_hashes) == 0, f"Duplicate content found between dev and locked_test: {dev_locked_hashes}"
