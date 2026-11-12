# Doc2Graph: pretrained classical ML news-domain build guide

## Objective and decision

Build an English news and biography document-to-graph application using existing pretrained, non-neural statistical models. Adapt the product ontology to the models rather than retraining a finance extractor. No generative LLM, transformer, neural NER, neural parser or neural coreference in this backend's inference path.

The first deliverable is a reproducible model compatibility and quality benchmark, not a replacement production pipeline. Stanford CRF NER plus the legacy supervised Stanford relation extractor is a candidate, not a guaranteed high-accuracy solution. Stop at a failed feasibility gate instead of building a polished interface over poor extraction.

Preserve existing finance code, models, reports and tests. Add an isolated `classical_news` domain/backend. Leave `rules` as the default until promotion criteria are met. The earlier finance corrective review remains relevant to reused storage/identity components; do not finish unrelated neural training work as a prerequisite for this experiment.

## Product scope

- English, machine-readable PDFs and plain text containing news reports, short biographies and organization descriptions.
- Explicit sentence-local mentions of people, organizations and geographical locations.
- Four supported relation families; verify native label spelling, argument order and type restrictions against the downloaded model and its training schema before freezing the adapter.

| Native model label | Proposed graph label | Proposed direction |
|---|---|---|
| Work_For | WORKS_FOR | PERSON -> ORGANIZATION |
| Live_In | LIVES_IN | PERSON -> LOCATION |
| OrgBased_In | BASED_IN | ORGANIZATION -> LOCATION |
| Located_In | LOCATED_IN | LOCATION -> LOCATION |
| None / native no-relation label | No edge | Not a semantic graph relation |

Treat this mapping as a design proposal until verified against native examples. In particular, do not guess location direction from mention order. Do not remap WORKS_FOR to ownership, membership or founding. Do not claim detection of acquisitions, investments, family relationships, causality, or cross-sentence relations.

Initially omit pronoun resolution, global entity linking and cross-document merging. Keep document-scoped entity identities and explicit mention records. Equal names alone do not prove that two people are the same. Unknown temporal/assertion status stays unknown; the model does not establish current employment or residence merely by extracting a relationship from an old report.

The UI should say “extracted relationship” and show the source sentence and document. It must not present a prediction as a verified, current real-world fact. Negated, hypothetical and historical examples must be represented in evaluation even if the model cannot handle them well.

## Verified external starting points

Checked 2026-11-12:

1. [Stanford CRF NER](https://stanfordnlp.github.io/CoreNLP/tools_crf_ner.html): pretrained English models, including PERSON/ORGANIZATION/LOCATION. Use the feature-based CRF, not a neural model with a CRF output layer.
2. [Stanford supervised relation extractor](https://nlp.stanford.edu/software/relationExtractor.html): documents a pretrained classifier for Live_In, Located_In, OrgBased_In, Work_For and None, exposed through the `relation` annotator. This is legacy software; verify that the chosen distribution actually contains compatible code and model resources. Do not substitute `openie` or `kbp` and call it the same model.
3. [CoreNLP distribution](https://stanfordnlp.github.io/CoreNLP/): the current page lists 4.6.0 and Java 11+. Pin a tested release and matching model jars. CoreNLP documents GPL licensing and a commercial licensing option; include the exact downloaded code/model license notices in the dependency inventory.
4. [Classical constituency parser](https://stanfordnlp.github.io/CoreNLP/parse.html): `parse` supports the English PCFG model. Explicitly select and record the model; do not enable neural `depparse` or neural coreference as a hidden dependency.

These pages document capabilities, not performance on our PDFs. Do not copy published NER scores into our benchmark or assume an old serialized classifier works unchanged in the newest distribution.

## Milestone 0: inventory and freeze the experiment

Read applicable repository instructions and inspect the working tree before editing. Relevant existing files include `src/pipeline.py`, `src/config.py`, `src/ingestion/source_map.py`, `src/nlp/contracts_v2.py`, `src/graph/graph_builder.py`, `src/search/query_engine.py`, and `src/api/schemas.py`.

Create `docs/CLASSICAL_NEWS_EXPERIMENT.md` before examining evaluation results. Record scope, mapping, splits, thresholds, scoring, gates and exclusions. Record known legacy issues separately. Do not compare relation F1 on news to finance/Re-DocRED F1 as a matched improvement.

Deliverable: inventory and proposed dependency/component manifest. No model retraining.

## Milestone 1: prove the pretrained stack works

1. Inspect Java availability and machine memory. Download only official artifacts into an isolated ignored directory, such as `.runtime/stanford_classical/`. Record version, original URL, SHA256, model resource paths and licenses. Keep install and launch instructions repeatable on Windows.
2. Inspect the chosen release source/resources to establish the relation model class, expected parser features, native labels, native entity mentions and argument ordering. Check all enabled annotators for neural dependencies. Feature-based CRF, maximum-entropy POS, PCFG parsing and a classical relation classifier are acceptable; standard tokenization and deterministic preprocessing are acceptable too.
3. Run the model before connecting it to the graph. Proposed annotator sequence is `tokenize,ssplit,pos,lemma,ner,parse,relation`; verify exact prerequisites and settings for the pinned release. Explicitly pin NER and PCFG resources. Do not rely on default full pipelines.
4. Save native outputs for at least 20 varied smoke sentences covering supported relations, no relation, reversed mentions, several entities, negation, historical statements and Unicode. These are wiring fixtures, not gold quality results. Do not hardcode their outputs as extraction rules.
5. Confirm both entity and relation annotations are actually emitted. A successful Java exit or valid JSON is insufficient. If the standard JSON serializer omits legacy `RelationMentionsAnnotation`, implement a small Java adapter reading the annotation objects directly. Do not parse a human-readable toString representation.
6. Export native relation label, ordered argument spans, argument mention type, sentence span and available score semantics. If only predicted labels are exposed, leave confidence null. If distributions are available, record them as raw scores until calibration is justified.

Suggested bridge: a persistent local Java process receiving JSONL requests and returning JSONL responses. Put diagnostics on stderr. Include request IDs, UTF-8 I/O, startup health check, bounded input sizes, timeout/error responses and clean shutdown. Python should launch with an argument array, not shell-built commands. A local HTTP bridge is also acceptable if bound to loopback and it preserves all annotations.

Gate: deterministic model loading, actual native relation output, auditable non-DL components and correct resource provenance. If the legacy extractor is unavailable or incompatible, report the exact failure. Do not silently switch to a neural backend. Finish the CRF NER benchmark independently if feasible; present alternatives before changing the relation approach.

## Milestone 2: build the benchmark before full integration

Use two separate evaluations:

### A. Native-task compatibility evaluation

Locate the relation model's original dataset/schema through its official documentation and inspect its training properties. Use only lawfully available data and record provenance and split overlap. Dataset naming alone does not establish that a pretrained model has not seen it. Public benchmark results are supplementary if checkpoint training membership cannot be verified.

### B. Independent target-domain evaluation

Start with 30 development documents and 30 separately held evaluation documents, approximately 5-15 sentences each, from permitted news/biography sources. This is a feasibility pilot, not enough by itself for a broad production claim. Seek at least 30 positive examples per relation across the evaluation collection; if the natural collection lacks support, gather more documents rather than fabricate labels. Report natural-document metrics separately from any deliberately balanced challenge set.

Group documents by source story/event/person or organization profile and near-duplicate content before splitting. Keep related rewrites in one split. Preserve URLs, retrieval dates, document/text hashes, document IDs, group IDs, split assignment and annotation version. Training provenance of the pretrained model may be incomplete; disclose that limitation rather than claiming proven absence of contamination.

Human reviewers mark exact entity spans/types and all in-scope directed relations within the evaluation scope. Record explicit negative candidate pairs where the annotation is complete. Unknown or unreviewed pairs are not negatives. Keep an adjudication log for ambiguity. Machine-generated annotations may assist review but never count as independently reviewed gold without that review. If reviewers are unavailable, build the annotation tooling and stop short of claiming validated quality.

Add a challenge set with no-relation co-occurrences, reversed relation direction, several organizations in one sentence, acronyms, historical employment, negation, quotations and repeated names. Do not restrict evaluation to model successes. Report skipped, errored and unsupported sentences and their effect on coverage.

Freeze evaluation labels and split manifests before tuning. Do not examine held-out errors during tuning; once inspected, that split becomes development data for subsequent changes.

### Required metrics

- NER: exact character span AND entity type micro precision/recall/F1; per-type support and F1. Token accuracy is not a substitute.
- Relation oracle mode, only if the native API truly allows supplied gold mentions: exact directed relation F1 with gold mentions. Otherwise report this mode unavailable; do not label predicted-NER results oracle.
- End-to-end text mode: ordered subject span/type, relation label and object span/type must match. Unmatched endpoints count as false positives; missed gold relations count as false negatives.
- Report entity-linked fact scoring separately from mention-level scoring if implemented. Explicitly define deduplication units; never mix mention counts and fact counts.
- PDF mode: compare extraction from a machine-readable PDF with the source text for the same documents, using the source map. Report text extraction and span-alignment failures separately.
- Per-relation P/R/F1, support, macro F1, micro F1, TP/FP/FN, document bootstrap confidence intervals, error rate, coverage, latency and peak memory.
- Persist per-document predictions and gold, evidence text, errors, code/version hashes and scoring configuration. No fuzzy name matching that silently forgives incorrect endpoints.

Evaluate the current rules backend only where an explicit, semantically equivalent label mapping exists. Do not map unrelated labels just to manufacture a before/after comparison.

## Milestone 3: feasibility decision

The following are proposed project targets, not expected pretrained scores or universal standards. Freeze them before results; report deviations explicitly.

| Measure | Initial feasibility target |
|---|---:|
| Exact entity micro F1 | >= 85% |
| End-to-end relation precision | >= 80% |
| End-to-end relation recall | >= 50% |
| End-to-end relation micro F1 | >= 60% |
| Supported-label and argument direction mapping | Fully tested |
| Silent inference failures or lost evidence | 0 |

Show per-label results so one easy label cannot hide failures elsewhere. A label with insufficient support remains experimental. Explicitly assess negation/historical errors before describing outputs as facts.

If the model fails, first inspect token/span conversion, argument order, model resource compatibility and annotation definitions. Fix actual adapter defects and rerun development only. Do not force edges, add example-specific rules or repeatedly tune against the held-out split. If the correctly wired model still fails, document that result and stop full integration. A future classical retraining proposal (CRF or feature-based relation classifier) is a separate decision; do not quietly start training.

## Milestone 4: implement the isolated backend

Suggested files (names are new proposals, not existing working commands):

| File | Responsibility |
|---|---|
| `config/ontology_classical_news.yaml` | Versioned types, directed labels and mapping |
| `src/nlp/stanford_classical_client.py` | Java bridge lifecycle and typed responses |
| `src/nlp/classical_news.py` | Native annotation adapter; no invented predictions |
| `scripts/probe_stanford_classical.py` | Runtime/model capability audit |
| `scripts/evaluate_classical_news.py` | Frozen split scoring and artifacts |
| `scripts/prepare_news_review.py` | Annotation packets and manifest validation |
| `java/stanford-classical-bridge/` | Pinned Java adapter build, if necessary |
| `tests/test_classical_news_contracts.py` | Offsets, mapping, errors and identity |
| `tests/test_classical_news_integration.py` | Actual model and database integration |

Wire a new explicit backend into Settings, CLI, API and pipeline; keep its ontology separate from finance. Ensure the graph writer's allowlist includes the exact new labels. Use request-specific settings rather than mutating globals. Return a clear unavailable-backend error when Java/models are missing; no silent fallback.

Reuse ingestion and source maps, but test their boundaries. Java character offsets use UTF-16 code units; Python indexes Unicode code points. Convert explicitly, including surrogate pairs/emoji. Preserve raw extracted text, normalized model input and offset maps. Assert entity text equals its mapped source slice. Keep page/chunk provenance where available. Do not locate repeated names using text.find.

Preserve the native model's mention semantics. Head spans and full entity spans are different; document conversion and evaluate it. Record any native argument that cannot be aligned instead of silently dropping it. Include sentence-level evidence so surrounding qualifiers remain visible.

Keep raw model scores distinct from calibrated probabilities. An acceptance threshold may be fitted on a separate reviewed development/calibration split if scores exist; unsupported labels abstain. Bind model, parser, NER, bridge, vocabulary/resources, ontology and threshold hashes together. Missing entries must fail closed rather than inherit permissive defaults.

## Milestone 5: graph and API semantics

Store document-scoped mentions and evidence-backed claims. Use stable IDs derived from document identity, source spans, relation and extractor version, with an explicit event identity only when supported. Retain all source claim IDs when displaying aggregates.

Do not merge different people globally by name or different facts solely by subject/relation/object. Use one identity policy across in-memory aggregation, batch writes and database MERGE. Reordering inputs, splitting batches or retrying must yield equivalent stored claims. Preserve sets of observed assertions or explicit conflict state; do not let a later observation erase a known contradiction.

Unknown polarity/modality/time must remain unknown, unless a separately evaluated component establishes them. The new relation model does not provide a license to default every claim to affirmed/completed/current. Distinguish model relationship hypotheses from verified assertion status in API and UI.

Store complex properties in a supported representation (for example JSON strings decoded and validated on read). Test write -> database read -> API serialization -> display, including evidence and statuses. Mock query tests are useful but are not live database round trips. Run live tests only against an isolated test database and preserve the existing user graph.

## Milestone 6: validation and promotion

Required regression cases:

1. All four native labels, no-relation output, direction and type mapping.
2. UTF-16/Python offsets, Unicode normalization, repeated names, paragraphs and PDF boundaries.
3. Missing resources, incompatible jars, malformed bridge response, process failure and timeout.
4. No neural annotator activated and no fallback to the legacy neural pipeline.
5. Unsupported confidence/status remains unknown; absent thresholds do not enable labels.
6. Missing/changed split manifests, duplicate source stories across splits; same-group documents permitted within one split.
7. Claim identity invariant under order, batching and retries; contradiction state preserved.
8. Actual native-model smoke test and live isolated database/API round trip, separately marked from unit tests.

Run appropriate tests plus the full existing suite. Fix new failures without removing finance tests merely because the new backend has a narrower ontology. Environment failures must be reported separately from model or code failures.

Promotion requires the frozen quality targets, enough per-label reviewed support, inspectable confidence intervals, passing integration tests and a fresh representative evaluation after any tuning. For a pilot, label the feature experimental even if point estimates pass. Keep the final release decision explicit: KEEP_DEFAULT_RULES, EXPERIMENTAL_CLASSICAL_NEWS, or PROMOTE_CLASSICAL_NEWS, supported by measured evidence. Do not switch the global default merely because installation or tests succeeded.

## Required final handoff from the implementer

Provide exact setup/run/evaluation commands validated on this machine, a dependency lock and model manifest, changed files, test results, dataset provenance and review status, P/R/F1 by stage and label, representative errors, measured runtime/memory, known limitations and release decision. State clearly which benchmarks, PDF checks and database tests were actually executed. If blocked on human labels or unavailable legacy models, complete independent work and identify the missing prerequisite without inventing scores.
