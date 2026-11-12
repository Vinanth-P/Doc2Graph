# SemEval feature-based SVM: implementation handoff

## Agreed scope

Build a local, CPU-capable, non-deep-learning relation classifier for Doc2Graph using the full SemEval-2010 Task 8 label inventory. Use all 8,000 official training examples for the final refit after development choices are frozen; keep the 2,717 official test examples out of fitting and tuning. No finance/news filtering. “Whole dataset” means full category coverage and all official training data, not mixing evaluation labels into training.

The research target is UTD's feature-based SVM approach. Distinguish three artifacts: a simple baseline, an explicitly UTD-inspired implementation, and a faithful reproduction only if every required feature/resource and classifier choice is reproduced. Do not describe a plain TF-IDF classifier as the original UTD system.

Primary references:

- [Rink and Harabagiu: UTD system paper](https://aclanthology.org/S10-1057.pdf).
- [SemEval task definition, data and scoring](https://aclanthology.org/S10-1006.pdf).

The UTD paper reports 82.19% F1 and 77.92% accuracy. Its architecture predicts the relation family first, then uses a separate direction classifier for each family. It uses 45 binary feature types drawn from lexical context, WordNet, dependency analysis, semantic roles, nominalizations and external lexical/relational resources. That result is a reference, not a promised outcome for our implementation. Read the complete paper and make a feature-by-feature implementation ledger before calling anything a reproduction.

## Product boundary

Phase 1 accepts a sentence with two marked noun phrases and predicts one directed relationship or Other. This matches the benchmark task. Phase 2 may detect candidate noun phrases automatically and produce a graph from documents, after Phase 1 has been evaluated. NER for people/organizations/locations is insufficient for ordinary nouns such as wheel, vehicle, water and bottle.

Do not compare the new classification score directly with the old finance or news extraction scores. The tasks, supplied inputs, labels and datasets differ. Report pair classification and automatic document extraction separately.

Preserve existing backends, artifacts, reports and unrelated edits. Existing rules stay the default during the experiment. Add an isolated `semeval_svm` model/API; no redesign of the UI or neural retraining is required for Phase 1.

## 1. Inventory and experiment declaration

Inspect applicable repository instructions, dependencies, working tree, prior review documents and model contracts. Useful existing components include source maps, model manifests, Stanford classical bridge and graph storage. Reuse only after testing the relevant boundary. The existing bridge's blocking reads and prior dataset-provenance issues must not be inherited silently.

Create `docs/UTD_SVM_EXPERIMENT.md` before training. Record data sources, hashes, label order, split seed, feature variants, solver choices, search budget, metrics and acceptance rules. All thresholds below are project proposals, not published guarantees. Pin Python/library and preprocessing versions actually tested on this machine.

## 2. Dataset acquisition and parsing

Retrieve the official release through the task's sources where available; if a mirror is necessary, preserve original IDs, files, provenance and verification evidence. Obtain the official scorer and its usage instructions. Record archive/file hashes, URLs, retrieval date and dataset terms. Do not fabricate sentences or gold annotations to fill missing files.

Write `scripts/prepare_semeval2010.py` and `src/nlp/semeval_dataset.py`:

- Parse the original ID, quoted tagged sentence, relation line and comment format correctly. Exclude comments and label lines from model features.
- Remove only entity marker tags while calculating their exact offsets in resulting text. Do not use text.find to rediscover repeated strings. Retain original tagged text and a conversion map.
- Validate exactly one e1 and e2, nonempty spans, legal nonoverlapping boundaries, unique IDs and a recognized directed label. Reject malformed records explicitly.
- Keep supplied phrase boundaries; do not substitute entity heads for benchmark spans. Store heads separately if features need them.
- Preserve original text without destructive whitespace or Unicode transformations. Any normalized preprocessing needs an explicit source map.
- Verify expected official split counts. Inventory all label supports. Audit duplicate/near-duplicate text and cross-split overlap; disclose official overlap without silently rewriting the benchmark.

Suggested record: example_id, source_split, text, e1_start/end/text, e2_start/end/text, relation_family, direction, original_label, content_hash. Store comments separately as audit-only metadata.

Model labels are the nine named families with both directions plus Other: 19 output labels. Other has no direction. Centralize round-trip conversion between native labels, family/direction and ordered graph arguments. Graph serialization should initially retain native family names and explicit source/target roles to avoid misleading inverse verb names.

## 3. Split discipline

Within official training data, create a deterministic development split, initially about 80/20. Stratify by directed label where feasible and keep duplicates together. Publish the actual membership, class counts and seed. If grouping prevents exact proportions, preserve isolation and report the difference.

During development, fit dictionaries, vectorizers, feature selection, scalers and classifiers only on the training portion. Fixed pretrained linguistic resources are permitted with provenance. Cache deterministic linguistic annotations by text and resource hash; never use test-wide statistics to choose features.

Choose model/features on development only. Freeze the recipe, then refit all trainable transformations and classifiers on all 8,000 official training examples. Evaluate that final artifact once on the official test split. If test results inform another change, disclose test reuse; it no longer provides fresh model-selection evidence. Never overwrite historical runs to hide that history.

If probability calibration or abstention is later required, use held-out training data or properly nested out-of-fold predictions. Refitting must not silently invalidate thresholds learned for another score distribution.

## 4. Implement the classifier in stages

### A. Reproducible lexical baseline

Implement binary sparse categorical lexical features and a linear SVM. Include argument surfaces, surrounding context and between-argument tokens with argument-specific prefixes. Record training speed, vocabulary size, development metrics and support. A single 19-class classifier is a useful comparison baseline, clearly named as such.

### B. UTD-style classification structure

Implement a ten-family SVM (nine families plus Other) and nine direction SVMs. Train each direction model only on training examples whose GOLD family is that relation. At inference, select the direction model using the PREDICTED family. Do not feed gold test families into the real inference path. Report oracle-family direction accuracy separately as a diagnostic if useful.

Use a fixed feature contract shared across training and inference. Handle missing training classes explicitly; never silently substitute random direction. Save family/direction class orders and solver parameters. Sparse LinearSVC is a pragmatic initial implementation choice; check the paper's actual software/kernel/settings and record departures rather than claim equivalence.

### C. Linguistic and knowledge features

Create a ledger with paper feature description, source section, implementation, dependency version, availability and deviation. Implement lexical/POS features, argument heads, ordered dependency-path features and WordNet-derived features first, following the paper where practical. Use classical POS/parsing components only; do not silently import neural spaCy, Stanza, transformer or neural semantic-role models.

Inventory the original semantic-role and external-resource requirements. For unavailable or impractical legacy resources, record the omission and resulting system name `UTD-inspired SVM`. Do not fill gaps using an LLM, invented facts or PDF-specific patterns. An optional replacement resource requires an explicit ablation and must remain non-neural under this project's constraint.

Use boolean categorical vectors for the reproduction-oriented feature track. Any TF-IDF, character n-gram or alternative feature extension is a separately named experiment. Classical features that generalize across the dataset are permitted; document-specific rules that dictate outputs are not.

Avoid unbounded searches. Initial development grid: C in {0.01, 0.1, 1, 10}, class_weight in {None, balanced}, fixed seed, recorded convergence settings. Reuse cached features. Expand only when development evidence motivates it. Report convergence warnings and measured runtime; do not assume a GPU is needed.

## 5. Scoring and decision

The primary metric is the official task's direction-aware macro F1 over nine relation families, excluding Other as a scored family. Use the official scorer, not a naive macro average over 19 labels. Wrong direction must not earn correct-relation credit. Compare any Python implementation with the official scorer on adversarial fixtures, including direction reversals, Other errors and absent labels.

Also report 19-label accuracy, per-family precision/recall/F1, support, confusion matrix, direction errors, paired bootstrap intervals and saved example-level predictions. Explicitly distinguish metric units and percentages. Keep original test IDs when writing scorer input.

Required development comparison: lexical baseline; two-stage lexical model; added POS/dependency features; added semantic features when available. Use identical splits and scoring. Report gains or regressions, including feature/resource cost. Do not claim that more features necessarily improve accuracy.

Proposed gates: data/label/scorer correctness must pass before interpreting scores. Choose the best reproducible development configuration. A final official-test macro F1 >=75% is a useful Phase 1 target, while 82.19% remains the published reference. Neither figure is a guarantee. Below target, deliver the trained research artifact, evidence and failure analysis; do not expand into broad automatic PDF extraction to disguise weak classification.

## 6. Artifact and inference contract

Suggested files:

| Path | Purpose |
|---|---|
| `src/nlp/semeval_dataset.py` | Strict parser, labels and offsets |
| `src/nlp/utd_features.py` | Versioned sparse features and resource adapter |
| `src/nlp/utd_svm.py` | Family/direction classifiers and artifact loader |
| `scripts/prepare_semeval2010.py` | Data manifest and development membership |
| `scripts/train_utd_svm.py` | Training, development search, final refit |
| `scripts/evaluate_utd_svm.py` | Official scoring and report generation |
| `scripts/predict_utd_svm.py` | Sentence + explicit span prediction CLI |
| `models/utd_svm/` | Versioned model, manifest and model card |
| `tests/test_semeval_dataset.py` | Parser and label invariants |
| `tests/test_utd_svm.py` | Training/inference and scoring boundaries |

Predict accepts sentence text and e1/e2 offsets, returning native directed label, canonical argument roles, raw decision scores, model version and evidence. SVM margins are not probabilities: do not put them into confidence fields restricted to [0,1], or apply an arbitrary sigmoid and call it calibrated. Introduce a typed raw-margin field or return nullable confidence with documented semantics. Other produces no semantic edge.

Manifest: artifact schema, code revision/hash, dependency versions, resource hashes, dataset hashes and counts, feature ledger, split IDs, seed, solver settings, label order, final-refit flag, development metrics, test evaluation history and model hashes. Loader rejects incompatible or incomplete artifacts. Restrict pickle/joblib loading to trusted locally produced artifacts.

## 7. Tests that demonstrate the behavior

Test repeated noun phrases, Unicode, quoted text, malformed/nested markers, multiword spans, reverse direction, Other, missing classes and unknown feature values. Verify all nine families and both directions survive parsing/serialization.

Use synthetic fixtures only for code tests, visibly marked fixtures. Test that development/test-only tokens cannot enter a fitted training vocabulary and that official test IDs never reach training. Test official scorer parity, including wrong direction. Check prediction equality before/after saving the model and CPU inference with the pinned linguistic pipeline. Test bridge timeouts if Java is reused.

Run meaningful focused checks, then the existing repository suite. Label environmental lock failures separately; do not remove old backend tests. Exact score assertions should use fixed small fixtures only, not require an invented benchmark score.

## 8. Phase 2: automatic document extraction, separately gated

Only after classification evaluation, prototype classical noun-phrase detection and sentence-local candidate selection. Preserve source offsets and use a fixed documented candidate policy. Report candidate recall, runtime and extraction precision/recall independently of marked-pair classification.

SemEval labels one supplied pair per example; unmarked pairs in its sentences are NOT reviewed negatives. Do not generate all noun pairs and label the rest Other. Other is a task-specific residual class, not a guarantee that two arbitrary phrases have no relationship. Automatic extraction needs independently reviewed candidate-level data and its own rejection/abstention evaluation.

Do not assume all detected pairs are meaningful or force edges for graph connectivity. No cross-document merging solely by name. Keep assertion status unknown unless a separately evaluated component supplies it. Verify text/PDF mapping and live graph/API persistence before claiming a complete document-to-graph backend. Report unsupported cases and dropped candidates.

## Completion package for Gemini

Deliver executable setup/train/predict/evaluate commands tested locally; pinned source/resource manifests; actual trained model; feature ledger; official development/test scoring outputs; per-label errors and ablations; tests; measured memory/runtime; and an honest limitations/model card. If a resource cannot be obtained, finish the available baseline and state exactly which UTD features are missing. Never label constructed annotations human-reviewed or report the paper's score as our result.

The first implementation milestone is the dataset parser, scorer parity and lexical SVM baseline. Continue through available non-neural features, model selection and final full-training refit. Do not start a new domain-restricted subset or neural backend.
