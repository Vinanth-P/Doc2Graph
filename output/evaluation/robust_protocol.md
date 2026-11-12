# Production-aware training protocol

This follow-up addresses the gold-entity versus production-entity mismatch and excessive false positives. It is not a fresh generalization study: development and test results from the earlier model have already been observed.

1. Retain the previous weights and all old reports. Write a separate `models/robust_relations.joblib` artifact. Keep inference features, entity extraction and relation schema unchanged to isolate training changes.
2. Exclude training documents overlapping development by title or normalized text. Generate training candidates through the existing production pipeline and align them with exact typed gold mentions. Unmatched and ambiguous predicted entities supply negative examples. Retain all production and gold-entity candidate negatives; remove balanced class weighting. Fit vocabulary/IDF and logistic-regression coefficients only on training documents.
3. Use even-index development documents for threshold selection: maximize true positives subject to at least 50% precision and five unique correct facts per label. Disable a label if no threshold qualifies. Duplicate predictions aligned to the same gold fact cannot inflate support. Keep unsupported facts in the recall denominator.
4. Use odd-index development documents for a separate validation diagnostic. These documents were observed in earlier work, so this is not an untouched holdout. No threshold selection from this validation result.
5. Record language-gate rejections explicitly. Preserve their gold denominator in diagnostics. Do not suppress unexpected pipeline failures.
6. Freeze the new artifact before running the old test split as a regression check. Do not label that check a new holdout. Report precision, recall and F1, including regressions and disabled labels.
7. Do not automatically promote an artifact solely for fewer false positives or a better F1. Require the earlier 50% precision / 20% recall gate and independent finance/polarity validation before changing the default.

No new relation word lists, PDF-specific names, or LLM calls are introduced. This is an incremental supervised-training correction; it does not expand the five-label ontology.
