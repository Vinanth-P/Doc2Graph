"""Entity resolver — classical ML: TF-IDF cosine + Jaro-Winkler + initials-key.

Blocking:
  1. Token-overlap (stopword-filtered, corporate suffixes removed, bucket cap 100)
  2. Initials-key  ("IBM" ↔ "International Business Machines")

Scoring (Logistic Regression, trained on resolution_gold.jsonl, k-fold CV):
  - jaro_winkler similarity
  - TF-IDF char n-gram cosine similarity
  - token overlap ratio
  - initials match (binary feature, NOT a separate bonus)
  - abbreviation score (one name is a prefix abbreviation of the other)

Cross-document resolution:
  Before writing a new entity, query Neo4j full-text index + initials_key
  property index for existing candidates and score with the same LR.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional

from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.nlp.ner import Entity

MODEL_PATH = Path("models/entity_resolver.joblib")

# ── Corporate suffix stoplist ─────────────────────────────────────────────────

CORP_STOP = frozenset({
    "inc", "corp", "ltd", "llc", "co", "company", "group",
    "university", "foundation", "institute", "association",
    "organization", "agency", "corporation", "enterprises",
    "holdings", "partners", "solutions", "services", "systems",
    "technologies", "technology", "international", "global",
})


# ── Utility functions ─────────────────────────────────────────────────────────

def _tokenize(name: str) -> list[str]:
    tokens = re.sub(r"[^\w\s]", " ", name.lower()).split()
    return [t for t in tokens if t not in CORP_STOP and len(t) > 1]


def compute_initials_key(name: str) -> str:
    """'International Business Machines' → 'IBM', 'Elon Musk' → 'EM'."""
    tokens = re.sub(r"[^\w\s]", " ", name).split()
    return "".join(t[0].upper() for t in tokens if t[0].isalpha())


def _token_overlap_ratio(a: str, b: str) -> float:
    ta, tb = set(_tokenize(a)), set(_tokenize(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _abbreviation_score(a: str, b: str) -> float:
    """1.0 if one name is a plausible abbreviation of the other (initials match)."""
    ka, kb = compute_initials_key(a), compute_initials_key(b)
    # 'IBM' matches key of 'International Business Machines'
    a_clean = re.sub(r"\W", "", a.upper())
    b_clean = re.sub(r"\W", "", b.upper())
    if ka == b_clean or kb == a_clean:
        return 1.0
    if ka == kb and len(ka) >= 2:
        return 0.8
    return 0.0


def _lucene_escape(text: str) -> str:
    """Escape Lucene special characters for Neo4j full-text queries."""
    specials = r'\+-&&||!(){}[]^"~*?:/'
    return "".join(f"\\{c}" if c in specials else c for c in text)


# ── TF-IDF cosine ─────────────────────────────────────────────────────────────

class _TFIDFCache:
    """Lazy char-ngram TF-IDF similarity (refit per batch)."""

    def fit(self, names: list[str]) -> None:
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self._matrix = self._vec.fit_transform(names)
        self._names = names

    def cosine(self, a: str, b: str) -> float:
        try:
            va = self._vec.transform([a])
            vb = self._vec.transform([b])
            return float(cosine_similarity(va, vb)[0][0])
        except Exception:
            return 0.0


_tfidf = _TFIDFCache()


# ── Feature vector ────────────────────────────────────────────────────────────

def _feature_vector(a_name: str, b_name: str) -> list[float]:
    return [
        JaroWinkler.similarity(a_name, b_name),
        _tfidf.cosine(a_name, b_name),
        _token_overlap_ratio(a_name, b_name),
        float(_abbreviation_score(a_name, b_name) >= 0.8),
        _abbreviation_score(a_name, b_name),
    ]


# ── Blocking ──────────────────────────────────────────────────────────────────

def build_candidate_pairs(
    entities: list[Entity],
    bucket_cap: int = 100,
) -> list[tuple[Entity, Entity]]:
    """Return candidate pairs via token-overlap and initials-key blocking."""
    pairs: set[tuple[str, str]] = set()  # store (id_a, id_b) tuples
    eid_map = {e.entity_id: e for e in entities}

    # 1. Token-overlap blocking
    tok_idx: dict[str, list[str]] = defaultdict(list)
    for e in entities:
        for tok in _tokenize(e.canonical_name):
            tok_idx[tok].append(e.entity_id)

    for bucket in tok_idx.values():
        if len(bucket) > bucket_cap:
            continue
        for a_id, b_id in combinations(bucket, 2):
            key = (min(a_id, b_id), max(a_id, b_id))
            pairs.add(key)

    # 2. Initials-key blocking
    init_idx: dict[str, list[str]] = defaultdict(list)
    for e in entities:
        k = compute_initials_key(e.canonical_name)
        if len(k) >= 1:   # allow single-char keys (e.g. "IBM" → "I.B.M" → "IBM")
            init_idx[k].append(e.entity_id)

    for bucket in init_idx.values():
        if len(bucket) > bucket_cap:
            continue
        for a_id, b_id in combinations(bucket, 2):
            key = (min(a_id, b_id), max(a_id, b_id))
            pairs.add(key)

    # Filter: same entity type, distinct entity IDs
    result = []
    for a_id, b_id in pairs:
        if a_id == b_id:
            continue
        a, b = eid_map.get(a_id), eid_map.get(b_id)
        if a and b and a.type == b.type:
            result.append((a, b))
    return result


# ── Resolver ──────────────────────────────────────────────────────────────────

class EntityResolver:
    """Merge near-duplicate entities within and across documents."""

    def __init__(self, threshold: float = 0.85):
        self._threshold = threshold
        self._lr = self._load_model()

    def _load_model(self):
        if MODEL_PATH.exists():
            try:
                import joblib
                return joblib.load(MODEL_PATH)
            except Exception:
                pass
        return None

    def _score(self, a: Entity, b: Entity) -> float:
        fv = _feature_vector(a.canonical_name, b.canonical_name)
        if self._lr is not None:
            try:
                return float(self._lr.predict_proba([fv])[0][1])
            except Exception:
                pass
        # Heuristic fallback
        jw = JaroWinkler.similarity(a.canonical_name, b.canonical_name)
        abbr = _abbreviation_score(a.canonical_name, b.canonical_name)
        return max(jw * 0.7 + abbr * 0.3, jw if abbr > 0.8 else 0.0)

    def resolve_within_doc(
        self, entities: list[Entity]
    ) -> dict[str, str]:
        """Return {entity_id: canonical_entity_id} mapping (Union-Find)."""
        if not entities:
            return {}

        _tfidf.fit([e.canonical_name for e in entities])
        pairs = build_candidate_pairs(entities)

        # Union-Find
        parent = {e.entity_id: e.entity_id for e in entities}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in pairs:
            if self._score(a, b) >= self._threshold:
                ra, rb = find(a.entity_id), find(b.entity_id)
                if ra != rb:
                    parent[rb] = ra

        return {eid: find(eid) for eid in parent}

    async def resolve_against_graph(
        self, entity: Entity, session
    ) -> Optional[str]:
        """Check Neo4j for an existing entity matching this one.

        Returns the existing entity ID to merge into, or None to create new.
        """
        escaped = _lucene_escape(entity.canonical_name)
        init_key = compute_initials_key(entity.canonical_name)

        # Full-text candidates
        ft_result = await session.run(
            """
            CALL db.index.fulltext.queryNodes('entity_fulltext', $q)
            YIELD node
            WHERE node.type = $type
            RETURN node.id AS id, node.name AS name
            LIMIT 20
            """,
            q=escaped, type=entity.type,
        )
        ft_rows = [r.data() async for r in ft_result]

        # Initials-key candidates (catches abbreviation expansions)
        ik_result = await session.run(
            """
            MATCH (n:Entity {initials_key: $ik, type: $type})
            RETURN n.id AS id, n.name AS name
            """,
            ik=init_key, type=entity.type,
        )
        ik_rows = [r.data() async for r in ik_result]

        # Deduplicate candidates
        seen, candidates = set(), []
        for row in ft_rows + ik_rows:
            if row["id"] not in seen:
                seen.add(row["id"])
                candidates.append(row)

        for cand in candidates:
            cand_name = cand["name"]
            fv = _feature_vector(entity.canonical_name, cand_name)
            if self._lr is not None:
                try:
                    prob = float(self._lr.predict_proba([fv])[0][1])
                    if prob >= self._threshold:
                        return cand["id"]
                    continue
                except Exception:
                    pass
            # Heuristic fallback
            jw = JaroWinkler.similarity(entity.canonical_name, cand_name)
            abbr = _abbreviation_score(entity.canonical_name, cand_name)
            score = max(jw * 0.7 + abbr * 0.3, jw if abbr > 0.8 else 0.0)
            if score >= self._threshold:
                return cand["id"]

        return None

    # ── Training ──────────────────────────────────────────────────────────────

    @staticmethod
    def train(gold_path: str) -> dict:
        """Train LR on resolution_gold.jsonl with k-fold CV."""
        import joblib
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        records = [json.loads(ln) for ln in Path(gold_path).read_text().splitlines() if ln.strip()]
        if not records:
            raise ValueError("Gold set is empty.")

        X, y = [], []
        for rec in records:
            # Format: {"name_a": "...", "name_b": "...", "label": 1 or 0}
            fv = _feature_vector(rec["name_a"], rec["name_b"])
            X.append(fv)
            y.append(rec["label"])

        X_arr, y_arr = np.array(X), np.array(y)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
        ])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, X_arr, y_arr, cv=skf, scoring="f1")

        pipe.fit(X_arr, y_arr)
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(pipe, MODEL_PATH)

        return {
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "n_samples": len(y),
        }
