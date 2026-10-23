# Local model artifacts

`relation_scorer.joblib` contains a dictionary with `feature_version: 2` and
`model`, a fitted scikit-learn pipeline. The feature vocabulary uses a fixed
ordering of entity type pairs and one-hot encoding. Negated relations are
excluded from asserted graph facts regardless of the model score.

The previous encounter-order encoding was inconsistent between training and
prediction. Legacy unversioned artifacts are ignored in favor of the heuristic
fallback until retraining. `relation_scorer.v1.joblib` preserves the previous
local artifact for reference; it is not loaded by the pipeline.

The v2 relation scorer was trained on the existing 20 labeled feature rows in
`data/eval/relations_gold.jsonl`. Five-fold stratified cross-validation produced
mean F1 0.88, standard deviation 0.098. These small-data results do not establish
general-domain accuracy or probability calibration. No spaCy weights were changed.

To retrain only this scorer from the project root:

```powershell
python -c "from src.nlp.relation_scorer import RelationScorer; print(RelationScorer.train('data/eval/relations_gold.jsonl'))"
```

`data/eval/relation_text_gold.jsonl` is a separate, hand-authored regression set
for full-text extraction and is not used to fit the scorer. Evaluate it with
`python scripts/benchmark.py --eval data/eval/`.
