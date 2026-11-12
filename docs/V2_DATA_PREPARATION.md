# V2 dataset preparation and review status

## What is implemented

`scripts/prepare_v2_dataset.py` converts existing Re-DocRED annotations into a separate, mention-preserving dataset. It does not call NER, merge predicted mentions, train a model, or change the production graph. The converter keeps annotated group membership separately from mention IDs; document-level facts are not broadcast as ground truth for every local mention pair.

The label order comes from `config/ontology_v2.yaml`. Every candidate has parallel `targets` and `loss_mask` arrays:

- Explicit positive: `1`, mask `true`.
- Explicit reviewed negative: `0`, mask `true` (supported by the supervision utility, not supplied by this corpus conversion).
- Unannotated/unknown: `null`, mask `false`.
- Unmatched or ambiguous predicted endpoints: all labels masked by `pair_supervision`.
- Assertion attributes absent from source annotations: `null`, mask `false`.

Never replace unknown labels with supervised zeros. A future trainer must replace nulls with numerical placeholders only after preserving the mask, compute loss only over supervised entries, and skip zero-supervision batches without dividing by zero. Do not infer no-relation supervision from `targets` containing no positive values. The current conversion has **zero reviewed negative labels**, so it cannot by itself support trustworthy discriminative training.

`align_mentions` performs exact typed alignment as a read-only diagnostic. It returns sets of gold mention IDs and preserves unmatched/ambiguous predictions. It is not an identity resolver and does not claim to fix NER. Boundary-overlap matching may be introduced as a separately reviewed training diagnostic; do not silently relax the primary evaluator.

## Prepared artifact

Location: `data/prepared/v2_redocred/`.

| Partition | Documents | Purpose |
|---|---:|---|
| train | 2,431 | Development training pool |
| validation | 306 | Model selection diagnostics |
| calibration | 315 | Threshold/calibration diagnostics |
| regression_dev | 500 | Previously inspected original development data |
| regression_test | 500 | Previously inspected original test data |

One training record was excluded for title/text overlap. The artifact contains 76,496 candidates, 3,737 positive label entries, zero explicit negatives and 684,727 unknown entries **across all five partitions**, not training alone.

`manifest.json` records source/code/ontology/output hashes, label order, deterministic assignments, counts, exclusions and limitations. All original training records have been used in earlier experiments: the new validation/calibration partitioning is for future controlled development, not a claim of untouched data. No release holdout was created. Only exact title/normalized-text duplication is checked; semantic company/deal/template grouping and approximate duplication remain outstanding.

Reproduce into a new directory:

```powershell
python scripts/prepare_v2_dataset.py --output data/prepared/v2_redocred_run2
```

The script refuses to overwrite an existing directory. Outputs use `schema_version=2.dataset.1`, a dataset interchange format distinct from `DocumentAnnotationPayload` for individual mention-level claims. `gold_relations` retain all mapped facts, including type-unsupported positives; `type_supported` makes this explicit. Do not drop unsupported positives from evaluation denominators. Evidence annotations have not been converted into evidence-selection targets.

## Reviewed finance data still required

Build a separate annotation pilot; do not relabel this artifact as finance gold. For each document retain source rights/provenance, document/company/deal/template group keys, reviewer and guideline version, exact mention spans, identity decisions, directed relations, minimal evidence and explicit polarity/modality/lifecycle annotations.

For a candidate pair, record each reviewed label as positive, negative or unknown. A reviewer saying “not reviewed” is different from “reviewed and absent.” Negated acquisitions are claims with negative polarity, not automatically no-relation examples. Signed agreements are not automatically completed acquisitions. Keep multiple statuses/events and disagreements separate.

Required annotation acceptance:

1. Exact spans round-trip to the recorded document text; IDs and references validate.
2. All listed positive/negative decisions have reviewer provenance and defined annotation scope.
3. Ambiguous/missing evidence remains unknown; adjudicate disagreements on a shared subset.
4. Split related companies/deals/versions/templates before assigning train/validation/calibration/locked test.
5. Audit exact and approximate duplication; protect final-test labels before training.
6. Include reviewed no-relation examples and planned, cancelled, conditional and negated events.

A reviewed-finance importer and heuristic near-duplicate grouping have since been implemented and corrected; see [the finance workflow review](FINANCE_WORKFLOW_REVIEW.md). Finance exports use `2.dataset.2` with relation-scoped status and provenance, whereas the Re-DocRED converter described here still uses `2.dataset.1`. Learned identity resolution and a masked-loss trainer are **not yet implemented**. Actual finance review and holdout custody remain outstanding.

## Corrections to the received milestone report

- Milestone A contains useful diagnostics but is not a completed causal audit. 731/880 false positives have unmatched endpoints; this does not prove those relationships are semantically correct or that boundaries alone caused them.
- Oracle-recovered end-to-end misses are 148, not 144; oracle still misses 418 of the end-to-end misses. Four end-to-end successes disappear under oracle inputs. These are set-intersection diagnostics, not NER-only versus model-only causal labels.
- The previous audit incorrectly asserted a four-sentence candidate cutoff. The code caps a distance feature at four, not candidate distance.
- Premature merges and missing-label negatives are credible failure mechanisms, but their contribution requires traced examples/controlled ablations. A dot-product/sigmoid scorer is not inherently incapable of discriminating negatives.
- Milestone B needed repairs: missing status must default to unknown, document-level references/spans must validate, and Unicode normalization needs covering-span provenance. These repairs now have regression tests. Pydantic `frozen=True` prevents field reassignment but does not recursively freeze nested containers.
- Source normalization is lossy. The map now covers contributing source characters, including combining sequences and partial ligatures; it is not an invertible one-character-to-one-character mapping.

The original audit is preserved in `output/evaluation/v2_error_audit.initial.md`; the corrected audit and computed counts are in `v2_error_audit.md` and `.json`. No old model weights or benchmark metrics were rewritten.

## Current milestone assessment

A: descriptive audit corrected; manual causal adjudication and merge/preprocessing ablations remain.

B: core contracts/source mapping repaired and tested; further schema hardening and source-layout integration remain.

C: public-corpus conversion, loss-mask representation and reproducible development manifests implemented. Reviewed negatives, finance annotations, semantic grouping and a fresh locked test remain blockers to a quality claim.

D–H: not completed by this work. The production backend remains unchanged. No extraction-accuracy improvement is claimed from dataset plumbing or passing tests.
