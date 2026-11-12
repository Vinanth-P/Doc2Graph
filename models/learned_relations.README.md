# Experimental learned relation extractor

`learned_relations.joblib` contains a trained TF-IDF vectorizer, five logistic-regression classifiers, development-selected thresholds and training metadata. It predicts relation labels from identity-masked entity-pair contexts; it does not require a dependency-pattern match or a verb whitelist. Inference is local, with no LLM/API call and no additional runtime dependencies beyond the existing scikit-learn stack.

This is a classical supervised baseline, not a transformer. It supports only `FOUNDED`, `EMPLOYED_BY`, `MEMBER_OF`, `OWNS` and `SUBSIDIARY_OF`. Acquisition, investment, reporting lines, advisory roles, and other rule-backend categories are not learned by this artifact.

## Training

- Source: [Re-DocRED](https://github.com/tonytan48/Re-DocRED), authors' training and development splits; license in `data/external/redocred/LICENSE`.
- 3,052 retained training documents; one source training document (`Lark Force`) was excluded because it also appears in development.
- 32,204 entity-pair training examples, with sampled negative pairs; all development/test candidates are evaluated without negative sampling.
- 100,000 learned TF-IDF features and 500,000 learned classifier coefficients.
- Vocabulary, IDF and classifier weights are fitted exclusively on training records. Per-label thresholds use the 500 development documents. No test data enters training or threshold selection.
- `learned_relations.training.json` records split hashes, candidate coverage, sampling, thresholds, counts and the artifact hash.

```powershell
python scripts/train_learned_relations.py --train data/external/redocred/train_revised.json --dev data/external/redocred/dev_revised.json --output models/learned_relations.joblib
```

Training that command replaces the specified artifact. Use a different output filename for further experiments, and retain the frozen artifact for the published test comparison. Only load trusted local joblib artifacts.

## Try without changing the database

```powershell
python scripts/inspect_document.py Doc2Graph_Corporate_Finance_MA_Dummy.pdf --relation-backend learned --output output/finance_graph_learned.json
```

This explicitly selects the learned backend for that invocation and writes a graph with evidence. It does not alter the running server or Neo4j. The learned backend replaces rule-based relationship extraction for that run; it does not silently mix rule predictions into its results. NER and entity resolution remain the existing pipeline components.

For an experimental server configuration, set `RELATION_BACKEND=learned` and optionally `LEARNED_RELATION_MODEL_PATH` in the environment or `.env`, then restart the backend. The default remains `rules`. Selecting `learned` with missing/incompatible weights fails explicitly rather than silently falling back to rules.

## Limitations

Entity recognition and resolution still limit end-to-end performance. Scores are classifier outputs tuned after negative sampling, not calibrated probabilities of truth. The training ontology does not distinguish proposed, negated, conditional or historical financial events sufficiently for reliable transaction-state claims. Cross-sentence context is limited to two closest mention-pair contexts, not full document reasoning.

Do not infer reliability from node connectivity. Consult the frozen held-out comparison in `output/evaluation/LEARNED_REPORT.md` before considering use beyond experimentation. Additional finance-domain labels and polarity/modality validation are required before default promotion.
