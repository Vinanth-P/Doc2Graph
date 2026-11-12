"""Split custody and group disjointness enforcement across dataset preparation, training, and calibration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


PROTECTED_SPLITS = {"test", "regression_test", "locked_test", "holdout"}


def load_manifest_assignments(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load doc_id -> {split, group_key, content_hash} from an authoritative manifest.json."""
    if not manifest_path.exists():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assignments: Dict[str, Dict[str, Any]] = {}
    for item in data.get("assignments", []):
        doc_id = item.get("document_id")
        if doc_id:
            assignments[doc_id] = item
    return assignments


def verify_split_custody(
    records: List[Dict[str, Any]],
    expected_split: Optional[str] = None,
    manifest_path: Optional[Path] = None,
    allow_fixtures: bool = False,
) -> None:
    """Enforce authoritative split custody, rejecting protected test records and split mismatches.

    Raises:
        ValueError if any record belongs to a protected split, fails split assignment, or is an unauthorized fixture.
    """
    assignments: Dict[str, Dict[str, Any]] = {}
    if manifest_path is not None and manifest_path.exists():
        assignments = load_manifest_assignments(manifest_path)
    else:
        # Check default locations
        default_manifest = Path("data/prepared/v2_redocred/manifest.json")
        if default_manifest.exists():
            assignments = load_manifest_assignments(default_manifest)

    for doc in records:
        doc_id = doc.get("document_id", "")
        data_origin = doc.get("data_origin", "")
        doc_split = doc.get("split")
        doc_source = doc.get("source", "")

        # 1. Guard against test fixtures
        if data_origin == "test_fixture" and not allow_fixtures:
            raise ValueError(
                f"Split custody violation: document '{doc_id}' is marked data_origin='test_fixture' "
                f"and cannot be used without allow_fixtures=True."
            )

        # 2. Check record's own split/source metadata
        if doc_source in PROTECTED_SPLITS or doc_split in PROTECTED_SPLITS:
            actual = doc_split or doc_source
            if expected_split not in PROTECTED_SPLITS:
                raise ValueError(
                    f"Protected split custody violation: document '{doc_id}' is marked '{actual}' "
                    f"and cannot be loaded for '{expected_split}'!"
                )

        if expected_split and doc_split and doc_split != expected_split:
            if not (expected_split == "val" and doc_split == "validation") and not (
                expected_split == "validation" and doc_split == "val"
            ):
                raise ValueError(
                    f"Split custody violation: document '{doc_id}' specifies split '{doc_split}', expected '{expected_split}'."
                )

        # 3. Check authoritative manifest assignments
        if doc_id in assignments:
            assignment = assignments[doc_id]
            assigned_split = assignment.get("split", "")
            if assigned_split in PROTECTED_SPLITS and (not expected_split or expected_split not in PROTECTED_SPLITS):
                raise ValueError(
                    f"Authoritative split violation: document '{doc_id}' is assigned to protected split '{assigned_split}' "
                    f"in the dataset manifest and cannot be used in '{expected_split}'!"
                )
            if expected_split:
                normalized_assigned = "val" if assigned_split == "validation" else assigned_split
                normalized_expected = "val" if expected_split == "validation" else expected_split
                if normalized_assigned != normalized_expected and assigned_split not in PROTECTED_SPLITS:
                    # If document is assigned to train but being loaded in calibration, etc.
                    raise ValueError(
                        f"Authoritative split mismatch: document '{doc_id}' is assigned to '{assigned_split}' "
                        f"in the dataset manifest, cannot be used as '{expected_split}'."
                    )


def verify_cross_split_disjointness(
    splits: Dict[str, List[Dict[str, Any]]],
    manifest_path: Optional[Path] = None,
) -> None:
    """Enforce strict document ID and group_key disjointness across dataset splits."""
    assignments = {}
    if manifest_path is not None and manifest_path.exists():
        assignments = load_manifest_assignments(manifest_path)
    else:
        default_manifest = Path("data/prepared/v2_redocred/manifest.json")
        if default_manifest.exists():
            assignments = load_manifest_assignments(default_manifest)

    seen_docs: Dict[str, str] = {}
    seen_groups: Dict[str, str] = {}

    for split_name, records in splits.items():
        for doc in records:
            doc_id = doc.get("document_id", "")
            if not doc_id:
                continue

            # Check document disjointness
            if doc_id in seen_docs:
                prev_split = seen_docs[doc_id]
                raise ValueError(
                    f"Cross-split document leakage: document '{doc_id}' appears in both '{prev_split}' and '{split_name}'!"
                )
            seen_docs[doc_id] = split_name

            # Check group disjointness (prevent leaking related documents / entities)
            group_key = doc.get("group_key") or assignments.get(doc_id, {}).get("group_key")
            if group_key and group_key != "default":
                if group_key in seen_groups:
                    prev_split = seen_groups[group_key]
                    raise ValueError(
                        f"Cross-split group leakage: group '{group_key}' (doc '{doc_id}') appears in both '{prev_split}' and '{split_name}'!"
                    )
                seen_groups[group_key] = split_name
