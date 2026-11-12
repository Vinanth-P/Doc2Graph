"""Relation confidence scorer — Logistic Regression on interpretable features.

Features (all classical, no embeddings):
  1. dep_distance       — shorter = higher confidence
  2. is_svo             — SVO pattern flag
  3. is_appos           — appositive pattern flag
  4. is_prep            — prepositional pattern flag
  5. is_nominal         — nominal/role-noun pattern flag
  6. verb_in_whitelist  — 1 if predicate lemma is in relations.yaml
  7. has_negation       — 1 if negation in governor's subtree
  8. entity_type_pair   — fixed one-hot vocabulary, identical across processes

Training:
  Run scripts/benchmark.py --train to fit the LR on relations_gold.jsonl
  using k-fold CV (k=5). Saves model to models/relation_scorer.joblib.

Inference:
  If model file is absent (e.g., no gold set yet), falls back to a simple
  heuristic score based on pattern type and whitelist membership.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.nlp.relation_extractor import RawTriple
from src.nlp.predicate_normalizer import load_relations_config
from src.type_map import NODE_TYPES

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = _PROJECT_ROOT / "models" / "relation_scorer.joblib"

# Whitelist lemmas built from relations.yaml
def _whitelist_lemmas(config: dict) -> frozenset[str]:
    lemmas = set()
    for conf in config.values():
        for entry in conf.get("lemmas", []):
            lemmas.add(entry["lemma"])
    return frozenset(lemmas)


FEATURE_VERSION = 2
TYPE_PAIRS = tuple((a, b) for a in sorted(NODE_TYPES) for b in sorted(NODE_TYPES))


def _feature_vector(triple: RawTriple, whitelist: frozenset[str]) -> list[float]:
    return [
        1.0 / (1.0 + triple.dep_distance),          # normalized distance
        float(triple.source == "svo"),
        float(triple.source == "appos"),
        float(triple.source == "prep"),
        float(triple.source == "nominal"),
        float(triple.predicate in whitelist),
        float(bool(triple.has_negation)),
        *[float((triple.subject.type, triple.object.type) == pair) for pair in TYPE_PAIRS],
    ]


class RelationScorer:
    """LR confidence scorer with heuristic fallback when no model is trained."""

    def __init__(self, config_path: str = "config/relations.yaml"):
        self._config = load_relations_config(config_path)
        self._whitelist = _whitelist_lemmas(self._config)
        self._model = self._load_model()

    def _load_model(self):
        if MODEL_PATH.exists():
            try:
                import joblib
                artifact = joblib.load(MODEL_PATH)
                # Old artifacts used encounter-order type IDs and cannot safely
                # consume the deterministic feature schema.
                if isinstance(artifact, dict) and artifact.get("feature_version") == FEATURE_VERSION:
                    return artifact["model"]
            except Exception:
                pass
        return None

    def score(self, triple: RawTriple) -> float:
        """Return a confidence in [0, 1]."""
        if triple.has_negation:
            return 0.0
        fv = _feature_vector(triple, self._whitelist)
        if self._model is not None:
            try:
                prob = self._model.predict_proba([fv])[0][1]
                return float(prob)
            except Exception:
                pass
        # Heuristic fallback
        return self._heuristic(triple)

    def _heuristic(self, triple: RawTriple) -> float:
        base = {"svo": 0.70, "appos": 0.65, "prep": 0.55, "nominal": 0.60}
        score = base.get(triple.source, 0.50)
        if triple.predicate in self._whitelist:
            score += 0.15
        if triple.has_negation:
            score -= 0.30
        if triple.dep_distance > 5:
            score -= 0.10
        return max(0.0, min(1.0, score))

    def score_batch(self, triples: list[RawTriple]) -> list[float]:
        return [self.score(t) for t in triples]

    # ── Training (called from scripts/benchmark.py) ───────────────────────────

    @staticmethod
    def train(gold_path: str, config_path: str = "config/relations.yaml") -> dict:
        """Train LR on relations_gold.jsonl with k-fold CV. Saves model."""
        import joblib
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        config = load_relations_config(config_path)
        whitelist = _whitelist_lemmas(config)

        records = [json.loads(ln) for ln in Path(gold_path).read_text().splitlines() if ln.strip()]
        if not records:
            raise ValueError("Gold set is empty.")

        X, y = [], []
        for rec in records:
            # Gold format: {"predicate": "...", "source": "...", "dep_distance": N,
            #               "has_negation": bool, "subject_type": "...", "object_type": "...",
            #               "label": 1 or 0}
            dummy = _DummyTriple(rec)
            X.append(_feature_vector(dummy, whitelist))
            y.append(rec["label"])

        X_arr = np.array(X)
        y_arr = np.array(y)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
        ])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, X_arr, y_arr, cv=skf, scoring="f1")

        # Fit on all data for the final model
        pipe.fit(X_arr, y_arr)
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump({"feature_version": FEATURE_VERSION, "model": pipe}, MODEL_PATH)

        return {
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std":  float(cv_scores.std()),
            "n_samples":  len(y),
        }


class _DummyTriple:
    """Reconstruct a RawTriple-like object from a gold record dict."""
    def __init__(self, rec: dict):
        self.source = rec.get("source", "svo")
        self.dep_distance = rec.get("dep_distance", 1)
        self.has_negation = rec.get("has_negation", False)
        self.predicate = rec.get("predicate", "")

        class _E:
            def __init__(self, t): self.type = t
        self.subject = _E(rec.get("subject_type", "PERSON"))
        self.object  = _E(rec.get("object_type", "ORG"))
