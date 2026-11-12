# Corrective implementation follow-up

## Second review: implementation improved, closure still blocked

Verified after the latest corrective changes: `pytest -q --basetemp=output/pytest_followup_review_2` reports **260 passed, 1 skipped** in 12.07 seconds, with an environmental pytest-cache permission warning. The earlier sections below describe the previous revision; this section records the current findings.

Confirmed improvements: Cypher now includes assertion properties and claim/deal merge keys; positive-only calibration is disabled by a negative-support gate; unsupported assertions default to unknown; raw sigmoid scores no longer populate calibrated_confidence. Existing rejected thresholds are rejected by the new guard in normal execution. These are real improvements, but the following boundary cases remain.

Direct reproduction results: `output/evaluation/v2_followup_guard_checks.json`. Probe artifacts are isolated under `output/evaluation/followup_guard_probe`; they contain placeholder bytes, not a trained checkpoint. No historical models were changed or retrained.

### A. Artifact validation still fails open for incomplete metadata

`src/nlp/artifact_guard.py` only checks model hash equality when `model_sha256` is present. A manifest with no hash and only one relation entry claiming precision 0.9 is accepted as `verified_calibrated`. Relations omitted from the threshold manifest are not added to disabled_relations, so inference uses its default 0.5 threshold for those labels. The reproduction accepted missing hash custody and left omitted ACQUIRED enabled.

Required: require hash custody and compatible metadata; initialize all supported relation labels as disabled and enable only individually verified entries. Validate finite thresholds/precision and recognized labels. Add missing-hash and partial-manifest regressions through the actual pipeline loader.

### B. Split custody is permissive and same-split grouping is incorrectly rejected

`verify_split_custody` accepts an unregistered document without split metadata even when its explicitly supplied manifest path does not exist; it silently falls back to the default manifest. Unknown IDs are not rejected. No content-hash check binds a record to its registered content.

`verify_cross_split_disjointness` raises when two different documents share a group within the SAME split. Reproduction: group probe-group appears in both 'train' and 'train'. Grouping related documents together is the intended behavior, not leakage. Also, calibration calls only verify_split_custody; it does not independently check its groups against the model's actual training/validation membership.

Required: require the specified authoritative manifest, registered document membership and matching content hashes. Permit repeated groups within one split; reject group reuse across different splits. Bind calibration membership to the training artifact's recorded document/group membership. Add missing-manifest, unknown-ID, changed-content, same-split-group and cross-split calibration-group tests.

### C. Contradiction handling is order-dependent; persistence identity differs by boundary

Reproduced in build_resolved_edges for one deal:

- affirmed, negated, affirmed -> affirmed
- affirmed, affirmed, negated -> unknown

The implementation cannot distinguish missing knowledge from a known conflict. Its Cypher CASE logic has the same reset behavior: a subsequent affirmed observation replaces conflict-derived unknown. Preserve the set of observed assertions or explicit conflict state so permutation and retry do not erase contradictions.

In-memory aggregation keys on deal identity, whereas Cypher keys on deal identity AND claim_id. Two claim IDs for one deal collapse to one edge in a single batch (retaining the first ID), but separate database writes can create two edges. Choose one consistent identity contract across batching, aggregation and database writes, while preserving source claim IDs. Test input order, duplicate retries and batch partitioning.

The new persistence regression uses AsyncMock and inspects query text/parameters; it does not execute a live Neo4j write/read round trip. No live database round trip was performed in this review either. Describe that verification accurately and add integration coverage before claiming storage behavior is verified.

### Next handoff

Resolve A-C with boundary-level regression tests before retraining. Keep rules as default and preserve historical experiments. The new suite count is verified; completion of every corrective guarantee and model-quality improvement are not.

## Verdict

The fixes are substantial, but the claim that all corrective tasks are resolved is not supported. Keep `rules` as default. Do not retrain or promote yet.

Verified locally: `pytest -q --basetemp=output/pytest_completion_review` reports **255 passed, 1 skipped** (10.93 seconds). A first run encountered temporary-directory permission errors; the workspace-local rerun passed. A pytest cache permission warning remains environmental. Passing tests do not establish model quality or complete persistence coverage.

This review changes no model, thresholds, or implementation. Existing artifacts remain historical experiments.

## Remaining corrective tasks for Gemini

### 1. Complete the actual database persistence path

`src/graph/graph_builder.py`, `_upsert_edges`, still uses `MERGE (a)-[r:RELATION]->(b)` without claim/event identity. Its SET clauses do not write polarity, modality, lifecycle, time qualifiers, or claim ID. Adding those fields to Python dataclasses and API schemas does not persist them. Distinct transactions still collapse at the database boundary.

The in-memory edge aggregation also keys only on deal_id or year (not date), and retains the first claim's status when subsequent claims merge. Implement explicit claim/event identity, contradiction handling and serialization suitable for Neo4j properties. Add a database write/read round-trip regression for two transactions between the same entities and negated/planned claims. Include the API read path.

### 2. Enforce artifact rejection and calibration eligibility at runtime

`src/pipeline.py` loads the preserved thresholds and disables only two named statuses. It does not validate model hashes, enforce a rejected-artifact state, or honor the explicit `disabled` boolean independently. Missing thresholds fall back to 0.5 through the extractor.

The existing `models/contextual_v2/thresholds.json` still marks FOUNDED calibrated at precision 0.03754 against target 0.70. Preserving historical files is correct; allowing the normal opt-in path to treat them as eligible is not. Add a shared artifact validator that fails closed on absent, incompatible, rejected or unverified calibration. If historical experiments must execute, require an explicit research-only override and label their outputs accordingly. Test the real loader, not just a directly constructed extractor.

### 3. Require reviewed negative support for precision qualification

`scripts/calibrate_v2.py` records `explicit_negatives_available`, but does not use it to gate fitting. Once positive support reaches `min_support`, a positive-only relation can qualify with apparent precision 1.0: there are no observed negatives to produce false positives. Require appropriate per-relation reviewed negative support and representative evaluation coverage before enabling precision-qualified thresholds; otherwise abstain and report insufficient supervision. Add a positive-only calibration regression.

### 4. Make assertion outputs support-aware and confidence honest

`ContextualRelationExtractor` still assigns argmax polarity/modality/lifecycle for every accepted relation without checking assertion supervision support. The legacy checkpoint adapter copies shared assertion weights into every relation head; compatibility does not establish relation-specific training. Unsupported assertion outputs must remain `unknown`, with support metadata bound to the checkpoint.

The extractor also sets `calibrated_confidence=prob` for a raw sigmoid probability, even though the new threshold manifest explicitly records `uncalibrated_raw_sigmoid`. Preserve raw_score and leave calibrated_confidence unset unless an actual probability calibration transform has been fitted and validated. Add inference-level tests for both guarantees.

### 5. Enforce split custody rather than only recording hashes

The trainer's loader checks `source` only for two protected strings; it does not enforce the expected train/validation membership or compare cross-split document/group overlaps. Calibration still uses a filename heuristic for test exclusion and records hashes without checking membership against an authoritative split manifest. Hash recording is useful provenance, not proof of isolation.

Require compatible preparation manifests, explicit split membership and document/group disjointness across training, validation and calibration. Reject protected examples even if their file is renamed. Add regressions for the same document/group across splits and protected records placed in a file named calibration.jsonl.

## Acceptance of the next correction

Reproduce each remaining failure, repair its full caller-to-storage or loader-to-inference path, and add regression tests at those boundaries. Preserve historical artifacts. Report which checks were actually run and distinguish mock query tests from a live database round trip. Only after these engineering corrections should reviewed finance supervision, retraining and independent quality evaluation proceed. No new model-quality result is established by this review.
