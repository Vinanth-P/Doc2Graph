# Finance workflow fixtures — not human-reviewed gold

`pilot_reviewed_annotations.json` is a six-document tooling demonstration. The filename and illustrative `expert_reviewer_*`, timestamp, `human_gold`, and review-status fields do **not** establish real human review or verified source provenance. No independent review evidence was provided with this artifact.

Each document is now explicitly tagged `data_origin: test_fixture`. The importer rejects it by default. To exercise validation, masking and export in a test-only directory:

```powershell
python scripts/import_reviewed_finance.py --input data/finance_pilot/pilot_reviewed_annotations.json --output output/finance_fixture_check --allow-fixtures
```

The outputs are fixtures too. Do not use them as training data, calibrated validation truth or an untouched test set. Statements about real companies in these examples have not been independently verified and are not factual reference material. Semantic annotation decisions, including no-relation versus negated-claim labels, require real adjudication before reuse beyond tooling tests.

The `proposals/` directory contains unreviewed machine proposals. A pending review status must always result in masked labels.
