# Learned relation extractor: held-out evaluation

## Decision

A real supervised relation model has now been trained and integrated as an optional backend. It learns relationship labels from text; inference makes no LLM/API calls. It improves recall and F1 on external test documents, but precision remains unacceptable. **Do not promote this candidate to the default.** The default remains `rules`, which also performs poorly on this benchmark.

This is a TF-IDF plus logistic-regression baseline, not a transformer or a complete solution for arbitrary PDFs. It supports only FOUNDED, EMPLOYED_BY, MEMBER_OF, OWNS and SUBSIDIARY_OF. It does not learn acquisition, advisory or investment relations.

## Held-out results

Both backends processed all 500 Re-DocRED test documents with zero exceptions. There are 581 mapped gold relationships. Metrics use exact typed mention alignment and entity-cluster relationships; unmatched predictions count as false positives. Other relationship labels are outside this evaluation.

| Input / backend | Correct | False positives | Missed | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Production entities / rules | 4 | 39 | 577 | 9.30% | 0.69% | 1.28% |
| Production entities / learned | 62 | 846 | 519 | 6.83% | 10.67% | 8.33% |
| Gold entities / rules | 20 | 27 | 561 | 42.55% | 3.44% | 6.37% |
| Gold entities / learned | 194 | 501 | 387 | 27.91% | 33.39% | 30.41% |

The end-to-end F1 gain is 7.05 percentage points; the paired document-bootstrap 95% interval is [4.22, 10.01] points (2,000 resamples). This supports a narrow benchmark improvement, not production reliability. Precision decreased, and the candidate fails both preregistered minimum promotion criteria: 50% precision and 20% recall. Finance-specific validation would still be required if those criteria passed.

The gold-entity diagnostic shows that entity recognition/resolution is a substantial bottleneck. Even with supplied gold entities, the learned extractor makes 501 false-positive predictions, so fixing NER alone will not solve the problem. Production exact typed entity-mention F1 is 53.21%.

## What was learned and how leakage was controlled

- Source: the authors' [Re-DocRED dataset](https://github.com/tonytan48/Re-DocRED), with license and split provenance under `data/external/redocred`.
- Training used 3,052 documents and 32,204 sampled entity-pair examples. One source training document, `Lark Force`, was excluded because it also occurs in development.
- Vocabulary, IDF and 500,000 classifier coefficients were fitted on training data only. Entity names are masked; candidates do not require a hand-written verb match. Existing NER, entity resolution and schema type constraints remain.
- The 500 development documents selected per-label thresholds. Development results are not held-out performance claims.
- Weights, thresholds and inference code were frozen before test download. The test title/text audit found no train/development overlaps. No model tuning followed inspection of test results.
- Model SHA-256: `76ab8b2dc4888ca49e91f0c55f46a60c79c0a2fcb0417658e3c94cd12e169bf5`.
- Test SHA-256: `ea673b7ef91e16510d7eb88863d3903e47318dc5653ef361266f27ae3b84723c`.

The test is now observed and must not be treated as a fresh holdout for future tuning. These Wikipedia-derived text results do not establish generalization to unseen financial PDFs, layout extraction, negation, proposed transactions or conditional events. Classifier scores are not calibrated truth probabilities.

## Implementation and validation

`src/nlp/learned_relations.py` implements local inference. `scripts/train_learned_relations.py` creates the actual weight artifact. The `learned` pipeline branch bypasses rule/dependency relationship extraction; it does not silently blend in rule edges. Missing or incompatible weights fail explicitly. Default behavior remains unchanged.

The full test suite passed: **184 passed, 1 skipped** (optional Neo4j integration). Tests cover identity masking, cross-sentence provenance, candidate generation without verb rules, split-overlap exclusion, artifact validation and bypass of rule extraction. Frozen source/artifact hashes were rechecked after evaluation with no mismatches.

An initial attempt ran both evaluations concurrently, conflicting with the pipeline's single-process file lock. Those invalid end-to-end runs are preserved as `rules_test.lock_conflict.json` and `learned_test.lock_conflict.json`; they are excluded from the table. Both backends were then rerun sequentially with identical frozen weights and thresholds. The corrected reports each contain 500 documents and no errors.

## Reproduce and inspect

Run these evaluation commands sequentially:

```powershell
python scripts/evaluate_learned_relations.py --dataset data/external/redocred/test_revised.json --backend rules --output output/evaluation/rules_test.json
python scripts/evaluate_learned_relations.py --dataset data/external/redocred/test_revised.json --backend learned --output output/evaluation/learned_test.json
python scripts/compare_external_evaluation.py --baseline output/evaluation/rules_test.json --revised output/evaluation/learned_test.json --output output/evaluation/learned_test_comparison.json
```

For an offline experimental graph from the original PDF:

```powershell
python scripts/inspect_document.py Doc2Graph_Corporate_Finance_MA_Dummy.pdf --relation-backend learned --output output/finance_graph_learned.json
```

This writes evidence-bearing predictions without changing Neo4j. More edges are not proof of correctness.

See [model documentation](../../models/learned_relations.README.md), [protocol](learned_protocol.md), [freeze manifest](learned_freeze_manifest.json), [overlap audit](test_overlap_audit.json), [comparison](learned_test_comparison.json), and the per-label metrics and document evidence in [learned test results](learned_test.json).
