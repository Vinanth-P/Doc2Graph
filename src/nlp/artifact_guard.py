"""Shared artifact validator enforcing model rejection and calibration eligibility at runtime."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ArtifactValidationResult:
    is_valid: bool
    is_provisional: bool
    model_sha256: str
    thresholds: Dict[str, float]
    disabled_relations: Set[str]
    artifact_status: str
    rejection_reasons: List[str]
    manifest: Dict[str, Any]
    thresholds_manifest: Dict[str, Any]


def validate_model_artifacts(
    model_dir: Path,
    allow_research_override: bool = False,
    target_precision: Optional[float] = None,
) -> ArtifactValidationResult:
    """Validate model artifacts and thresholds, failing closed on rejected or unverified calibration.

    Args:
        model_dir: Path to directory containing model.pt, vocab.json, training_manifest.json, thresholds.json.
        allow_research_override: If True, allows loading rejected research baseline artifacts for evaluation,
                                 flagging outputs as provisional.
        target_precision: Target precision to enforce. If None, reads from thresholds.json.
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    req_files = ["model.pt", "vocab.json", "training_manifest.json", "thresholds.json"]
    for fname in req_files:
        if not (model_dir / fname).exists():
            raise FileNotFoundError(f"Required artifact {fname} missing from {model_dir}")

    actual_model_hash = file_sha256(model_dir / "model.pt")
    manifest = json.loads((model_dir / "training_manifest.json").read_text(encoding="utf-8"))
    thresholds_data = json.loads((model_dir / "thresholds.json").read_text(encoding="utf-8"))

    rejection_reasons: List[str] = []

    # 1. Check explicit rejection flags
    if manifest.get("artifact_status") == "rejected_research_baseline" or manifest.get("rejected", False):
        rejection_reasons.append("Model is explicitly marked as rejected_research_baseline")

    # 2. Check model hash custody between model.pt and thresholds.json
    recorded_model_hash = thresholds_data.get("model_sha256")
    if recorded_model_hash and recorded_model_hash != actual_model_hash:
        rejection_reasons.append(
            f"Model hash mismatch: thresholds recorded {recorded_model_hash}, actual model.pt is {actual_model_hash}"
        )

    # 3. Check relation calibration eligibility
    thresholds: Dict[str, float] = {}
    disabled_relations: Set[str] = set()

    t_precision = target_precision if target_precision is not None else thresholds_data.get("target_precision", 0.70)
    relations_info = thresholds_data.get("relations", {})

    calibrated_count = 0
    for r_name, r_info in relations_info.items():
        thresh = float(r_info.get("threshold", 0.5))
        thresholds[r_name] = thresh

        is_disabled = r_info.get("disabled", False)
        status = r_info.get("status", "unknown")
        prec = float(r_info.get("precision", 0.0))

        # Check for false claims of calibration:
        # e.g. status claims calibrated but achieved precision is below target!
        is_ineligible = (
            is_disabled
            or status != "calibrated"
            or (status == "calibrated" and prec < t_precision)
        )

        if is_ineligible:
            disabled_relations.add(r_name)
            if status == "calibrated" and prec < t_precision:
                rejection_reasons.append(
                    f"Relation {r_name} claims calibrated status but precision {prec:.2%} is below target {t_precision:.2%}"
                )
        else:
            calibrated_count += 1

    if calibrated_count == 0:
        rejection_reasons.append("All relation heads are disabled or fail calibration precision target")

    is_valid = len(rejection_reasons) == 0

    if not is_valid:
        if not allow_research_override:
            raise ValueError(
                f"Artifact validation failed for {model_dir}: {'; '.join(rejection_reasons)}. "
                "Normal execution rejected. Use allow_research_override=True for research-only evaluation."
            )
        artifact_status = "rejected_research_baseline"
        is_provisional = True
    else:
        artifact_status = "verified_calibrated"
        is_provisional = False

    return ArtifactValidationResult(
        is_valid=is_valid,
        is_provisional=is_provisional,
        model_sha256=actual_model_hash,
        thresholds=thresholds,
        disabled_relations=disabled_relations,
        artifact_status=artifact_status,
        rejection_reasons=rejection_reasons,
        manifest=manifest,
        thresholds_manifest=thresholds_data,
    )
