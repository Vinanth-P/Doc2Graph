# Review of classical news completion claims

## Verdict

The backend is a useful working prototype, but the reported 93.09% NER F1 and 42.58% relation F1 are diagnostic scores against constructed, automatically labeled examples. They are not independently reviewed news-domain accuracy. Keep `rules` as default. Do not tune the model around these scores or present the constructed locked_test split as a validated independent holdout.

This review does not alter model artifacts, source code or historical benchmark reports. Direct dataset inspection results are saved in `output/evaluation/classical_news_review_checks.json`.

## 1. Annotation provenance is incorrectly asserted by code

`scripts/prepare_news_review.py`, `build_raw_documents`, constructs documents from literal examples in Python. The automatic annotation path uses KNOWN_PERSONS/KNOWN_ORGS/KNOWN_LOCS lists and text patterns such as `works for`, `based in` and `located in`. It writes reviewer IDs `curated_pilot` / `annotator_verified_pilot` and `review_status=human_verified_gold` itself. Creating review packets does not demonstrate completed human review.

The current locked_test file contains 30 constructed documents and 114 positive labels, all carrying the automatically assigned curated_pilot provenance. This is not evidence of independent adjudication, real source ingestion or complete gold labels. Mark these records as fixtures/machine drafts and pending review. Keep their scores explicitly diagnostic. Do not merely change reviewer strings to make them eligible.

Required: import real permitted source documents with provenance, build annotation packets, accept actual human review separately, and make evaluation eligibility depend on that review. Obtain a fresh group-disjoint evaluation set after development is frozen. Existing inspected examples are development/fixtures, regardless of their filenames.

## 2. Gold spans and relation coverage are defective

`make_mention` uses full_text.find(text), always returning the first occurrence. Inspection found duplicate typed mention spans in 14 of the 30 development documents. Different mentions of repeated names can therefore collapse onto the same location. This distorts both exact-span NER and mention-level relation scoring.

The locked test has ZERO positive LIVES_IN labels despite explicit residence sentences. For example, news:test:001 contains `Dimon lives in New York.` but its mention list omits Dimon; the automatic name list prevents the relation from being labeled. Correct model predictions on such missing annotations can be counted as false positives.

Required: explicit occurrence-specific offsets, complete sentence-level annotation, reference and span validation, reviewed handling of short-name mentions, and regression tests for repeated names. Keep unknown/unreviewed outcomes separate from negatives. The evaluator currently treats every predicted tuple absent from the positive set as false; that requires a completely reviewed scope, which these generated labels do not provide. Do not infer model syntactic limitations causally from these flawed labels alone.

## 3. The lock fix does not implement timeouts

The client now uses RLock, which addresses the described nested-acquisition deadlock. However, startup checks elapsed time around a blocking stdout.readline(); a silent live child can block that read indefinitely. annotate also uses blocking readline, and request_timeout is not enforced. A stalled bridge can therefore still hang processing while holding the shared client lock.

Required: deadline-aware reader/queue or equivalent subprocess protocol, timeout-triggered cleanup and restart, response request-ID validation, and tests using a fake child that stays alive without emitting a line. Test startup silence, request silence and delayed/stale replies without requiring Java. Do not claim the hang class eliminated from one successful runtime test.

## 4. Provenance and graph integration need boundary checks

The extractor tests for source_map.get_source_span, but SourceMap exposes normalized_to_source/source_to_normalized instead. The supplied mapping is not used through that branch. Specify the raw/normalized offset contract and test nontrivial transformations rather than relying on identical-text smoke cases.

`entity_to_chunks` is populated after iterating a group's mentions and records only the last mention's sentence-derived chunk. Confirm that these IDs refer to actual persisted chunks; verify all mention-to-chunk links through a database read. Entity IDs use truncated ASCII slugs, which can collide for distinct names within a document; use collision-resistant identities and test punctuation, non-ASCII names and long shared prefixes.

The displayed integration suite contains live extractor and text-pipeline checks, a split-file check and saved-report assertions. It does not execute a live Neo4j write/read or PDF extraction round trip. Name those checks accurately and add the missing boundary tests before claiming end-to-end storage/evidence verification. Silent rejected candidates and dropped links need counters; counting caught extraction exceptions does not establish zero lost evidence.

## 5. Test verification and report reproduction

Review run: `pytest -q --basetemp=output/pytest_classical_review` produced 273 passed, 1 skipped and 1 failure. The failure was a shared pipeline-lock contention in tests/test_classical_news_contracts.py::test_pipeline_nlp_classical_news, not demonstrated model or scoring failure. Pytest cache permissions also generated environmental warnings. This is a different current test count from the submitted 271-passed summary.

The evaluator defaults to output/classical_news_evaluation.json and a model_backend/results schema, while the report-presence test reads output/classical_news_evaluation_report.json and backend/feasibility_gates fields. Provide one reproducible report-generation command and versioned schema that binds saved tables to the evaluator, inputs, native predictions and model hashes. Checking a saved release-decision string does not reproduce a benchmark.

The isolated retry of the pipeline-lock failure passed: `pytest -q tests/test_classical_news_contracts.py::test_pipeline_nlp_classical_news --basetemp=output/pytest_classical_review_retry` (1 passed, 9.14 seconds). This supports transient lock contention; it is not a fresh all-green full-suite run.

No full model benchmark was rerun in this review because the label provenance and coverage defects already prevent interpreting it as independent accuracy. No new quality estimate is claimed.

## Next task for the implementer

1. Correct provenance descriptions and mark constructed data as fixtures/pending review; preserve prior reports as superseded diagnostics.
2. Repair span annotation and review eligibility; add repeated-name and incomplete-label scoring regressions.
3. Implement bounded subprocess reads and request-ID checks with deterministic fake-process regressions.
4. Repair and verify offset, entity identity and persisted chunk/evidence boundaries.
5. Collect genuinely reviewed source material, freeze a fresh evaluation split, and reproduce text/PDF metrics with a single report generator.

Do not retrain, promote or select a narrow high-performing relation based on these fixture scores. The classical model remains a plausible candidate, but its real target-domain accuracy is still unestablished.
