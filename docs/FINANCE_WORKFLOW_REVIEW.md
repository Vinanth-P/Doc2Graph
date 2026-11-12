# Finance workflow implementation review

The supplied completion report overstated the importer guarantees. The initial five tests did not cover pending-review supervision, competing relation statuses or duplicate group bridges. The following defects were corrected without retraining a model or changing the production backend.

| Finding | Correction |
|---|---|
| A pending machine proposal with `decision=positive` received target 1; pending negatives received 0 | Document and annotation review gates now control supervision. Missing/unverified provenance and pending annotations stay masked. |
| Multiple relations on one pair overwrote shared polarity/modality/lifecycle and reviewer fields | Finance export version `2.dataset.2` stores assertion targets, evidence and provenance per relation. |
| Repeated same-relation annotations silently used the last status | Duplicate relation-pair annotations require adjudication; no silent last-write behavior. Multi-event support remains future work. |
| Reviewer fields and timestamps were optional even for reviewed documents | Reviewed document metadata now requires a named reviewer, guidelines version and timezone-aware ISO timestamp. This validates structure, not human authenticity. |
| Exact duplicate exclusion happened before group union, potentially dropping a bridge between groups | Group all documents before exclusion. Namespace document/group keys and reject duplicate document IDs. |
| Duplicate text with different labels could be discarded | Retain differently annotated copies together; exclude only identical content and annotation provenance. |
| Split code was 60/15/10/15, report claimed 60/15/15/10 | Corrected bucket boundaries and record proportions in manifest. Small-corpus counts remain variable. |
| Plaintext test output was always labeled sealed | Manifest records `locked_test_sealed=false`, `release_holdout_available=false`; actual custody/access control is not implemented. |
| Pilot claimed reviewed gold using illustrative expert identities | Pilot is explicitly `test_fixture`, rejected by default. `--allow-fixtures` is test-only and is recorded in manifest. |
| Annotation guide conflated explicit denial with no relation | Denial is a positive claim with negative polarity; no-relation negatives require separate scoped review. |

The converter retains original annotations even when their supervision is masked. Unknown sentence indices stay unknown instead of becoming sentence zero. The importer validates all documents and ontology constraints before creating output directories. Relevant dependency hashes and group membership, including excluded records, are included in the manifest.

Validation after corrections: **233 tests passed, one optional integration test skipped**. The fixture-only CLI export also completed in `output/finance_workflow_review_check/`; it records six fixture documents, `contains_test_fixtures=true`, `review_provenance_verified=false`, and `locked_test_sealed=false`. None of that output is independent gold or release-test evidence. `git diff --check` found no whitespace errors.

## What is not complete

- No genuine reviewed finance corpus or verified reviewer/source provenance has been established.
- A hash split is not a sealed or untouched release test. No claim of independent model quality follows from it.
- A type-pair check cannot validate semantic direction when both arguments are organizations.
- Shingle similarity cannot guarantee discovery of all related deals, templates or paraphrases.
- Repeated events with different statuses need an event-scoped annotation/export schema, not a single relation-pair label.
- The proposal tool uses heuristic suggestions and currently reads text, not PDF binaries. Those suggestions are not learned extraction improvements.
- No training or accuracy claim is justified by this tooling review. The default remains `rules`.

Next work is real annotation/adjudication and a documented holdout-custody process. Build the contextual trainer only against the versioned masked-label contract, with fixtures excluded and reviewed evidence available.
