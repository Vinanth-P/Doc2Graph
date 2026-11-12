# Frozen external evaluation protocol

Registered before running either version on the downloaded documents.

- Dataset: all 500 documents in the authors' Re-DocRED `dev_revised.json`. No sampling by topic, successful extraction, or relation presence. Dataset SHA-256 and all snapshot file hashes are in `freeze_manifest.json`.
- Baseline: repository HEAD `1f8f7a762d2a15360f28d3772597a163d5da41bd`, before the finance fixes. Revised: working source frozen at the start of this evaluation. Neither snapshot receives edits after seeing results. Model weights are unchanged.
- Input: space-joined dataset sentence tokens. End-to-end runs use the production text pipeline with its own sentence splitting, NER, coreference, scoring and entity resolution. No gold labels or entities enter that run.
- Scope: P112 founder -> FOUNDED (reversed), P108 employer -> EMPLOYED_BY, P463 member of -> MEMBER_OF, P127 owned by -> OWNS (reversed), P749 parent organization -> SUBSIDIARY_OF. All labeled instances count even when the pipeline does not support their entity types. Other relations, including generic RELATED_TO and geographic relationships, are excluded from precision/recall rather than guessed into a label.
- Report both all five mapped types and the three already supported in the baseline: FOUNDED, EMPLOYED_BY, MEMBER_OF. The all-five comparison includes added ontology coverage, not only improved extraction.
- Metrics: micro precision, recall, F1 and per-relation TP/FP/FN. Strict typed entity mention P/R/F1 for PER->PERSON, ORG->ORG and LOC->GPE. Report ignored relation counts and label coverage.
- Relation endpoints are linked to annotated clusters using exact typed character spans of their mentions. A predicted merged node matching multiple gold clusters is unmatched; gold labels are never used to choose its target. Unmatched endpoints remain false positives. Deduplicate graph facts per document.
- Separate diagnostic: gold mention spans, entity types and mention clusters supplied to the same dependency extractor. Unsupported MISC/NUM/TIME entities are omitted; their mapped gold relations remain in the denominator. Unannotated pronouns remain heuristic. This is not an achievable production score or an official Re-DocRED leaderboard score.
- Pipeline exceptions remain logged and count as missed gold relations, rather than dropping failed documents.
- No scorer fitting, new rules, threshold tuning, or model selection against these results. Once inspected, this dev set becomes development evidence; future trained/replacement models require a separate untouched test split.
- Decision: retain useful plumbing/cleanup work, but do not declare broad generalization merely because scores improve. Low recall even with gold entities indicates candidate extraction is a bottleneck that retraining the current binary scorer cannot fix. Do not auto-deploy a replacement or rewrite stored graphs based on this limited benchmark.

This evaluates prose in a different domain mix, not PDF layout, all corporate-finance relations, OCR, or complete real-world factual correctness. Pretrained spaCy corpus overlap is unknown; the claim is only that these project changes were not developed using these documents.

Sources: https://github.com/tonytan48/Re-DocRED and https://github.com/thunlp/DocRED.
