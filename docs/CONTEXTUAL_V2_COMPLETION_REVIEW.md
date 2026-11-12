# Review of the claimed A–H completion

## Verdict

**Do not accept the claim that all milestones are implemented and verified.** Several components exist and the current suite passes (248 passed, 1 skipped), but critical supervision, calibration and persistence defects remain. The research gate fails. Final release validation has not occurred. Keeping `rules` as default is the correct non-promotion decision, not evidence that the redesign succeeded.

This review does not modify training code, model weights, thresholds or historical reports. Focused reproduction output is saved in `output/evaluation/v2_completion_review_checks.json`.

## Blocking findings

### 1. Unknown labels become supervised negatives again

`scripts/train_contextual_relations.py:188` samples from `doc_unannotated`, writes nine zero targets and sets all nine loss-mask entries to true. Default `negative_ratio=2.0` enables this path for training and validation. A correct masked-loss function cannot undo incorrect labels created upstream.

Reproduction using the first eligible prepared training document (`redocred:train:0:442295ca21cc`): input explicit negative entries = **0**; emitted all-zero, fully supervised samples = **2**. This directly contradicts the claimed no-unknown-to-zero guarantee.

**Required:** preserve supplied targets/masks. Unknown pairs can be retained for unsupervised objectives or annotation selection, but not relabeled as negatives. Fail clearly if the corpus lacks the reviewed negatives required for the intended discriminative training. Add an input-to-dataset regression test, not only a zero-gradient test of the loss. Review and retrain the resulting artifact; a code patch does not repair existing weights.

### 2. Calibration bypasses unknown masks and the precision gate

`scripts/calibrate_v2.py:143` labels every non-gold candidate as zero instead of consulting reviewed supervision masks. This includes finance labels that the corpus does not annotate. It counts mention-pair instances, not unique document facts, so repeated mentions can inflate support. It also evaluates candidate/label combinations without applying the final per-relation type policy.

At line 220, failing the requested precision target falls back to maximum F1. At line 234 the output is still called `calibrated`. The saved FOUNDED entry claims calibrated status with **3.75% precision against a 70% target**. Finance labels with zero positives get threshold 0.80, which permits predictions; it is not enforced abstention. Inference ignores calibration status and uses only thresholds.

**Required:** calibrate only reviewed outcomes with matching prediction units, relation constraints and deployment prevalence. Unknowns stay masked. Disable unsupported labels explicitly. Distinguish threshold selection from probability calibration; raw sigmoid values must not be stored as `calibrated_confidence`. Require dataset/model hashes and split membership, not a filename substring, to check calibration isolation. Do not generate replacement thresholds from a corpus with no reviewed negatives and claim validation.

### 3. Status supervision is not relation-specific end to end

The importer correctly writes `assertions_by_relation`, but `extract_samples_from_records` takes the first relation's attributes and breaks (`scripts/train_contextual_relations.py:159`). The model has one polarity/modality/lifecycle output per pair. `ContextualRelationExtractor` assigns those same attributes to every emitted relation.

For example, an existing ownership relationship and a proposed acquisition between the same companies cannot receive different statuses through this architecture. Heads without supervised labels still emit argmax categories; no support gate forces unknown status.

**Required:** relation-conditioned assertion outputs/losses, or relation-specific samples with a consistent inference contract. Preserve per-relation evidence and reviewer provenance. Enforce unknown status for unsupported heads. Test distinct simultaneous relation statuses and partially masked assertion labels through loader, loss and inference.

### 4. Truncation silently removes arguments

`SimpleVocabulary.encode` truncates to 128 tokens. Candidate generation allows up to 800 characters between mentions and always records `truncated=False`. Missing marker positions become -1; `ContextualRelationModel.forward` substitutes the CLS vector when an argument marker is absent.

Reproduction with `Alice`, 150 intervening `word` tokens, then `Acme`: encoded marker positions are **[1, 3, -1, -1]**. The object is absent, but classification is still possible. This invalidates a blanket no-lost-provenance claim.

**Required:** an argument-preserving context builder, explicit truncation/coverage diagnostics, and rejection or abstention when complete argument spans cannot fit. Test long, reversed, overlapping and Unicode-normalized mentions. Mask padding appropriately in the sequence encoder; currently padded positions receive positional embeddings and enter the bidirectional GRU.

### 5. Claim status is lost at the persistence boundary

`src/pipeline.py:243` converts every v2 claim into `NormalizedTriple`, including negated/planned/unknown claims, without carrying polarity, modality or lifecycle. The result dictionary additionally contains `v2_claims` and `v2_canonical_claims`, but the graph/API persistence path does not consume those fields. Tests that assert the dictionary keys exist do not establish status-preserving graph writes.

There is also an offset-domain mismatch to address: mentions use normalized preprocessor/chunk offsets, while this branch passes the earlier `text` variable into extraction. NFKC/whitespace changes can make those offsets point to different source text. The source-map implementation is not integrated into this path.

**Required:** a versioned claim/event persistence adapter, evidence/source-map fidelity, and storage round-trip tests proving a proposed/negated transaction is not exported as an unconditional edge. Keep uncertain claims inspectable without asserting completion. Do not drop the additional v2 fields at serialization or offline graph inspection.

### 6. Aggregation still collapses transactions and changes confidence without validation

`src/nlp/claim_aggregation.py:34` groups only by canonical subject, relation and object. Different events/times are therefore merged. `time_quals.update` overwrites conflicting qualifiers; evidence span keys lack document identity. The aggregator adds an arbitrary confidence bonus for multiple spans. `calibrated_confidence or raw_score` also treats a legitimate calibrated value of zero as missing.

**Required:** document/event/time-aware claim identity, explicit contradiction handling, correct optional-value handling and no unvalidated confidence bonus. Preserve distinct transactions and their source claims. Test two deals between the same companies and confidence exactly zero.

### 7. “Comprehensive final evaluation” is only a partial oracle diagnostic

Although its docstring lists three modes, `scripts/evaluate_v2.py:113` calls only `evaluate_document_oracle`. It does not measure the full production pipeline, predicted-cluster impact, per-label metrics, assertion confusion or evidence quality. The default limit is 100 documents. The engineering PASS and `236+` tests are hard-coded strings, not measured gate checks. `representative_finance_support=5` is hard-coded despite fixture status, and no fixture evaluation is actually invoked by the displayed main path.

The report compares a 100-document run to hard-coded 500-document baseline totals. Recomputing the earlier oracle baseline on the same first 100 source documents gives:

| Same first 100 development documents, gold entities | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Earlier learned model | 31 | 139 | 94 | 18.24% | 24.80% | 21.02% |
| Reported contextual v2 | 31 | 591 | 94 | 4.98% | 24.80% | 8.30% |

The earlier number is recomputed from the saved fact sets; v2 values are from its supplied report, not a fresh rerun here. This is not an improvement. The v2 evaluator also permits all nine predictions against gold data mapped to five relations; a corrected benchmark must explicitly declare matched label scope before comparing.

**Required:** implement the advertised modes or mark them unimplemented. Use identical document IDs, label scope and scoring for comparisons. Retain per-document predictions, errors, hashes and evidence. Count verified human finance support from provenance, never fixtures or a constant. Do not mark H complete merely because gate-checking code executed.

### 8. Entity-stage numbers do not show a solved entity detector

`output/evaluation/entity_stage_benchmark.json` covers **50** documents. Its strict typed NER F1 is **38.89%**. Pairwise resolution F1 80.15% and B-cubed F1 94.01% are separate clustering metrics, evaluated with supplied gold mentions. They are not NER scores or proof of end-to-end improvement; the report includes false merges.

**Required:** a matched old/new resolver comparison, explicit mention-input mode, candidate/coverage analysis and real NER improvement where indicated. Correct metric behavior for missing predicted mentions and empty prediction sets. Keep D marked partial until its stated quality exit condition is demonstrated.

### 9. Training provenance and fixture exclusion are not enforced

The training manifest records hyperparameters/history but not input file hashes, split membership, ontology hash, label-support counts, fixture exclusions or artifact hashes. `load_dataset_records` silently ignores missing files and accepts records without checking `data_origin=test_fixture`. A report saying “reviewed finance annotations” is not substantiated by this manifest.

**Required:** fail on missing inputs, require compatible dataset manifests, exclude test fixtures and protected partitions, verify group/hash isolation, and record actual supervision counts by relation/status. A filename test alone is not split isolation. Record the model as a from-scratch embedding + bidirectional GRU baseline; contextual processing is real, but it is not a pretrained language encoder.

## Revised milestone status

| Milestone | Defensible status |
|---|---|
| A | Descriptive audit exists; causal claims need traced examples/ablations |
| B | Contracts and covering source map implemented; source-map integration not complete |
| C | Tooling exists; real reviewed finance data and protected holdout remain absent |
| D | Resolver prototype benchmarked; entity-quality exit condition not demonstrated |
| E | Neural prototype trained; upstream masking and relation/status architecture defects block acceptance |
| F | Threshold-fitting prototype exists; calibration/abstention guarantees fail |
| G | Opt-in execution path exists; graph persistence loses assertion semantics |
| H | Partial development oracle diagnostic only; research gate failed and final validation not performed |

## Next corrective task for Gemini

Do not train again or declare completion yet. First fix and test, in this order:

1. End-to-end supervision-mask preservation, fixture/split guards and artifact provenance.
2. Complete argument retention, normalized/source offsets and explicit coverage failures.
3. Relation-conditioned assertion supervision and support-aware unknown/abstention behavior.
4. Claim/event-aware serialization, persistence and aggregation.
5. Honest evaluation modes, matched baselines and data-driven gate evidence.
6. Calibration on genuinely reviewed outcomes; disable unsupported labels instead of falling back to a failed target.

Add regression tests reproducing the defects above before implementing each fix. Preserve existing artifacts as rejected experiments. Then obtain sufficient reviewed positive/negative/status supervision, retrain and reassess on the correct splits. Passing unit tests or a tiny-subset overfit test is not the research milestone's quality exit condition.
