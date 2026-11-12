# External generalization evaluation

**Contextual v2 completion review:** The claimed A–H completion is not supported. The trainer promotes unknowns to negatives, calibration can fail its target while reporting success, and graph persistence drops claim status. The reported v2 score is a partial oracle diagnostic, not final end-to-end validation. See [the verified findings and corrective task list](../../docs/CONTEXTUAL_V2_COMPLETION_REVIEW.md).

**Finance importer review:** Pending machine labels could leak into supervision, multi-relation statuses were overwritten, and a plain-text test partition was incorrectly marked sealed. These paths have been corrected; the pilot is explicitly test-fixture material. See [the workflow review](../../docs/FINANCE_WORKFLOW_REVIEW.md). No model-quality improvement or genuine locked holdout is established.

**V2 implementation review:** The received milestone report had incorrect causal claims and source-map/status-default bugs. See [dataset preparation and review status](../../docs/V2_DATA_PREPARATION.md) and the [corrected computed audit](v2_error_audit.md). A/B are not fully validated, and C now has a conservative masked-label dataset artifact; finance annotations and a fresh holdout remain missing.

**Latest — 11 November 2026:** Production-aware retraining and a pretrained relation encoder both failed validation. The extraction-quality problem remains unresolved; neither candidate replaced the default. See [the follow-up report](FOLLOWUP_REPORT.md) for matched comparisons and limitations.

**Follow-up:** A supervised relation classifier has now been trained and evaluated on the separate 500-document test split. See [the learned-model report](LEARNED_REPORT.md). It improves recall but has excessive false positives and remains opt-in. The report below records the earlier rules-only experiment; statements about no replacement model being trained apply to that earlier stage.

## Decision

The finance-PDF result did **not** establish a reliable general-purpose model improvement. Both frozen versions perform poorly on this external text benchmark. The revised rules add some coverage, but there is no clear paired statistical evidence of improvement on the shared relation types.

Keep the PDF parsing, provenance and graph-integrity fixes as useful engineering changes. Treat the added relation rules as provisional, not as a validated general extractor. The next model-development step should replace or augment dependency-pattern candidate generation with a learned relation extractor. Retraining the existing confidence scorer alone cannot recover relationships that were never proposed.

No production extraction rules, weights or stored database graphs were changed during this evaluation. No replacement model has been trained or validated yet.

## Data and controls

- All **500** independently annotated documents from the authors' Re-DocRED development split; no cherry-picking by successful extraction or document topic. Both versions completed all documents with **zero pipeline exceptions**.
- The dataset contains 17,284 relation annotations. Only **613 facts (3.55%)** map to the five evaluated project relations: FOUNDED, EMPLOYED_BY, MEMBER_OF, OWNS and SUBSIDIARY_OF. This is a deliberately narrow schema-overlap evaluation, not a score over all graph relationships.
- Original source: repository HEAD `1f8f7a762d2a15360f28d3772597a163d5da41bd`. Revised source: working files frozen before running this benchmark. The two learned model artifacts have identical hashes.
- Dataset, source and evaluator hashes are preserved in `freeze_manifest.json`; the evaluation rules were recorded in `protocol.md` before the runs. The only evaluator adjustments before the full run corrected Unicode offsets and ensured supplied gold cluster IDs could not be fuzzy-merged.
- These documents were not used to develop the project changes. Whether spaCy's original pretraining/training data overlapped with the public corpus is unknown.
- This development split is now inspected development evidence. The separate test split was neither downloaded nor inspected and must remain untouched during future training and tuning.

## Results

Strict end-to-end scoring requires matching relation direction and entity identity. Entity identities are aligned through exact typed mention spans. Unmatched or ambiguously merged endpoints are scored as wrong, without using relation labels to choose a favorable match.

| Evaluation | Baseline precision | Revised precision | Baseline recall | Revised recall | Baseline F1 | Revised F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| End-to-end, five mapped types | 1.92% | 6.90% | 0.16% | 0.65% | 0.30% | 1.19% |
| End-to-end, three shared types | 1.92% | 3.92% | 0.26% | 0.52% | 0.46% | 0.92% |
| Gold entities/clusters supplied, five types | 22.64% | 28.30% | 1.96% | 2.45% | 3.60% | 4.50% |
| Gold entities/clusters supplied, three shared types | 22.64% | 23.40% | 3.13% | 2.87% | 5.50% | 5.12% |

The revised end-to-end run recovers **4 of 613** mapped gold facts, with **54 scored false positives**. The baseline recovers **1 of 613**, with 51 scored false positives. On the three types both versions already support, the revised run recovers 2 of 383 compared with 1 of 383.

Supplying gold entity mentions, types and mention clusters improves revised recovery to **15 of 613**. This is a diagnostic of the extraction/normalization stages, not a deployable model or a formal upper bound. It retains heuristic handling of unannotated pronouns and the project's existing entity-type restrictions.

Exact typed entity mention F1 increases from **50.63% to 52.39%**. Entity handling improves modestly, but this does not solve relation extraction.

The paired document-bootstrap 95% interval for the shared-type end-to-end F1 change is **-0.008 to +1.512 percentage points**, including zero. The gold-entity shared-type F1 change is negative, with an interval also spanning zero. The small number of true positives and wide intervals do not support a robust generalization claim.

## Interpretation and limits

1. **Sparse candidate generation is a major bottleneck.** Even with labeled entities and their identities supplied, most mapped facts never become correct relations. The current learned scorer evaluates triples proposed by rules; it does not generate missing entity pairs or infer their relation from document context.
2. **Entity alignment contributes substantially to the strict end-to-end penalty.** 38 of the revised run's 54 scored false positives have at least one unmatched endpoint. These include span/type differences and bad entity merges; they must not all be described as fabricated semantic claims. The gold-entity diagnostic helps separate this issue from relation extraction.
3. **Schema restrictions also reduce recall.** A post-run coverage audit found that 415 of the 613 mapped annotations fit the revised configuration's entity-type pairs. The remaining 198 are outside that structural capability. Even generously using only 415 as the denominator would cap gold-entity recovery at 15/415, or 3.61%. The original registered denominator was not changed to improve the reported score.
4. **Some broad rules produce wrong meanings.** For example, a sentence saying someone “started his career” can become FOUNDED because the normalizer maps `start` to founding and the parser finds a nearby organization. This is evidence that adding verbs to a whitelist is insufficient.
5. **The benchmark is not finance-PDF accuracy.** It contains Wikipedia-derived prose, excludes PDF layout/OCR, and lacks many finance-specific labels. Re-DocRED's “parent organization” is mapped to the project's SUBSIDIARY_OF category; exact corporate/legal semantics can differ. Dataset labeling and strict boundary conventions also affect scores. These are custom projected metrics, not official leaderboard scores.
6. **The local training sets are tiny and do not train relation generation.** The checked-in scorer and resolver gold files each contain 21 rows; the 15-document local relation regression set is not an independent external test. These counts do not establish the historical provenance of the serialized weights, but show the current reproducible training/evaluation resources are limited.

## Next model-development step

Use a separate training corpus to learn relationships from entity-pair and document context, including negative pairs and cross-sentence cases. Keep a stable relation ontology, distinguish planned/negated facts, and select thresholds on development data only. Evaluate the frozen candidate on the untouched test split and on independently labeled finance documents with real PDF layout variation before replacing the production extractor. Measure correct triples and evidence, not connectivity alone.

A learned extractor still needs evaluation; this report does not promise that choosing a transformer or using a larger model will fix the problem automatically. This turn implements the external evaluation and establishes the bottlenecks, not a trained replacement.

## Reproduce and inspect

```powershell
python scripts/evaluate_external.py --snapshot output/evaluation/snapshots/baseline --dataset data/external/redocred/dev_revised.json --output output/evaluation/baseline.json
python scripts/evaluate_external.py --snapshot output/evaluation/snapshots/revised --dataset data/external/redocred/dev_revised.json --output output/evaluation/revised.json
python scripts/compare_external_evaluation.py --baseline output/evaluation/baseline.json --revised output/evaluation/revised.json --output output/evaluation/comparison.json
python -m pytest -q tests/test_external_evaluation.py
```

`baseline.json` and `revised.json` contain per-document predictions, gold facts, false positives/negatives, false-positive evidence, per-relation counts and execution failures. `comparison.json` contains the paired bootstrap results (2,000 resamples; fixed seed). The evaluator's **8 tests pass**, covering direction, deduplication, offsets, missing predictions, strict endpoint alignment, ambiguous merges, oracle identity isolation and identical-run comparisons.

Sources: [Re-DocRED authors and dataset](https://github.com/tonytan48/Re-DocRED), [paper](https://arxiv.org/abs/2205.12696), [original DocRED format](https://github.com/thunlp/DocRED/blob/master/data/README.md), [original relation-name metadata](https://huggingface.co/datasets/thunlp/docred/tree/main/data). The downloaded license and dataset provenance are in `data/external/redocred/`.
