# Local Model Artifacts & Versioning

Doc2Graph relies on lightweight classical ML models (scikit-learn + rapidfuzz) to run efficiently on CPU without transformer or GPU dependencies.

## Canonical Model Paths

The canonical models loaded by the pipeline are defined in `config/config.yaml` (and overridable via environment variables in `src/config.py`):

1. **Relation Scorer**: `models/relation_scorer.joblib`
   - Fitted scikit-learn pipeline with `feature_version: 2`.
   - Evaluates dependency parse triples and entity type pairs.
   - Retrain via:
     ```bash
     python -c "from src.nlp.relation_scorer import RelationScorer; print(RelationScorer.train('data/eval/relations_gold.jsonl'))"
     ```

2. **Entity Resolver**: `models/entity_resolver.joblib`
   - RapidFuzz string distance, token set, and acronym matching classifier for intra- and cross-document coreference.
   - Retrain via:
     ```bash
     python -c "from src.graph.entity_resolver import EntityResolver; print(EntityResolver.train('data/eval/resolution_gold.jsonl'))"
     ```

## Artifact Versioning & Distribution

- Dead or legacy binaries (such as `.v1.joblib` checkpoints) are omitted from version control and Docker builds.
- When scaling or distributing larger weight sets, model artifacts can be tracked via Git LFS, DVC, or pulled at container startup via an initialization script.
