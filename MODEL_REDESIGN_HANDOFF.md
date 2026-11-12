# Doc2Graph: extraction redesign and implementation handoff

Prepared 11 November 2026. Audience: Gemini or another engineering reviewer implementing and independently verifying this project.

## 1. Objective and current decision

Build a document-to-graph system that extracts **supported, directed relationships with traceable evidence**, including financial transactions. It must work beyond the supplied dummy PDF. Training, entity extraction and relation prediction should use local learned models, not a generative LLM service doing the extraction at runtime. An LLM may assist development; its unreviewed outputs are not ground truth.

The current extraction-quality problem is **unresolved**. Do not interpret existing code, passing tests, additional edges, or this plan as proof of model quality. Preserve working ingestion, API and graph infrastructure while replacing or correcting the extraction middle in an isolated backend.

Provisional scope: English narrative PDFs, initially corporate finance/M&A plus the five existing general relations. Scanned documents and tables need explicit evaluation; arbitrary domains, arbitrary relations and multilingual extraction are not promised. Verify this scope with the project owner if implementation choices depend on it. Do not silently narrow the application to the dummy PDF.

## 2. Read this evidence before changing code

Workspace: `C:\Users\asus\OneDrive\Desktop\Doc2Graph`. Shell: PowerShell. The working tree contains substantial uncommitted changes from multiple stages. Inventory it; do not reset or overwrite unrelated work.

Read in order:

1. `output/evaluation/REPORT.md` — historical rules benchmark and links to follow-ups.
2. `output/evaluation/LEARNED_REPORT.md` — first trained classifier and old test comparison.
3. `output/evaluation/FOLLOWUP_REPORT.md` — failed production-aware retraining and encoder diagnostic.
4. `output/evaluation/learned_protocol.md`, `robust_protocol.md`, freeze manifests and split-overlap audit.
5. `models/learned_relations.README.md`, `models/encoder_relations.README.md` and training JSON files.

### Recorded results: do not mix evaluation sets

| Experiment | Evaluation population | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Rules | 500 old test documents, production entities | 9.30% | 0.69% | 1.28% |
| First learned classifier | Same old test documents | 6.83% | 10.67% | 8.33% |
| Rules with gold entities | Same old test documents | 42.55% | 3.44% | 6.37% |
| First learned classifier with gold entities | Same old test documents | 27.91% | 33.39% | 30.41% |
| First learned classifier | 500 development documents | 5.07% | 7.67% | 6.10% |
| Pretrained GLiREL adapter | Same 500 development documents | 4.15% | 2.77% | 3.32% |
| Production-aware retraining | Odd-index 250-document development subset | 11.11% | 0.30% | 0.58% |

The last retraining result has just one correct prediction. It is not a successful precision improvement. Its comparison baseline on that exact subset is in the follow-up report.

Important qualifications:

- External labels cover only FOUNDED, EMPLOYED_BY, MEMBER_OF, OWNS and SUBSIDIARY_OF. Acquisition and advisory extraction were not evaluated by those scores.
- The old test split has already been inspected. It is now a regression dataset, not a new untouched holdout.
- Development data has also been inspected repeatedly. Any further results on it are development diagnostics.
- Exact typed entity-mention F1 was about 53% on the old test set. Relation failures remain substantial even with gold entities.
- The external text benchmark does not measure PDF layout/OCR performance. Strict mention alignment can penalize boundary mismatches; report relaxed diagnostics separately without replacing the strict primary metric.
- The encoder's pretraining overlap with benchmark documents is unknown. Its 4% result describes this integration, threshold and mapping, not every possible contextual model.
- Earlier concurrent evaluations collided with a single-process pipeline lock. Invalid files ending in `.lock_conflict.json` must not enter quality comparisons.

## 3. Repository map and what to preserve

| Area | Existing files | Intended treatment |
|---|---|---|
| Orchestration | `src/pipeline.py`, `src/pipeline_lock.py` | Add isolated v2 backend; preserve locking and async persistence separation |
| Loading and cleanup | `src/ingestion/loader.py`, `pdf_cleaner.py`, `preprocessor.py`, `ocr.py` | Retain useful behavior; add source-offset/layout fidelity where missing |
| Entity detection | `src/nlp/ner.py`, `document_entities.py` | Benchmark learned NER and existing recovery independently |
| Coreference and identity | `src/nlp/coref_resolver.py`, `src/graph/entity_resolver.py` | Keep mention identities; audit and reduce premature merging |
| Rules and legacy scoring | `relation_extractor.py`, `predicate_normalizer.py`, `relation_scorer.py` | Preserve baseline; do not hide legacy edges inside v2 |
| Experimental models | `learned_relations.py`, `encoder_relations.py` | Preserve for reproducibility; neither is approved for default use |
| Graph and boundaries | `src/graph/graph_builder.py`, `neo4j_client.py`, `src/api/schemas.py` | Version claims/status representation and migrate compatibly |
| Configuration | `src/config.py`, `config/config.yaml`, `config/relations.yaml` | Add explicit backend/model/schema versions; test actual precedence |
| Evaluation | `scripts/evaluate_external.py`, `evaluate_learned_relations.py`, `evaluate_encoder_relations.py` | Keep historical results reproducible; add a v2 evaluator rather than rewriting metrics to favor it |

Current default is `relation_backend=rules`. Optional values are `learned` and `encoder`. The learned/encoder branch resolves entities before extraction; the rules branch has its own coreference and later resolution flow. Do not describe all branches as identical.

Last recorded full suite: 192 passed, one optional integration test skipped. Rerun it; this is a historical count, not an acceptance substitute.

## 4. Independent audit: verify these hypotheses first

Treat the following as hypotheses, not conclusions that must be confirmed:

1. Entity boundary/type errors and incorrect merges materially damage end-to-end extraction.
2. TF-IDF pair contexts cannot adequately represent direction, cross-sentence relations and transaction state.
3. Gold-entity training versus noisy-entity deployment contributed to the original mismatch.
4. The pretrained adapter may have semantic, preprocessing or runtime issues despite strict parameter loading.
5. The graph boundary may discard distinctions between claims, events and established facts.

Audit 50–100 sampled development errors, stratified across false positives, false negatives, labels and document styles. Record a primary cause and secondary causes. Separate ingestion, NER, coreference/resolution, candidate coverage, label mapping, model scoring, assertion status and graph aggregation errors. Include successful examples to avoid an error-only view.

For GLiREL specifically, check official reference examples, tokenizer/checkpoint compatibility, added-token order, marker offsets, inclusive/exclusive spans, label direction, label-order sensitivity, CPU/GPU agreement, window truncation and scores on unrelated/negated pairs. Strict loading establishes key compatibility only. Do not tune on one toy example until it passes and call that validation.

Deliver `output/evaluation/v2_error_audit.md` with counts, representative evidence, reproducible commands and the highest-impact interventions. Stop adding architecture complexity if a simpler implementation bug explains the failures.

## 5. Proposed architecture: evidence before irreversible consolidation

```text
PDF/text
  -> structured source blocks + normalized text + reversible offset map
  -> typed entity mentions with stable IDs
  -> candidate mention pairs and relevant context
  -> contextual relation + assertion-status predictions + evidence spans
  -> conservative document identity/coreference decisions
  -> aggregate supported claims onto canonical entities/events
  -> versioned graph payload -> existing API/Neo4j path
```

The important invariant is **preserve mention-level evidence and reversible identity decisions**. Moving all resolution after relation extraction is a proposed design to test, not a universal theorem. Document-level models may need soft clusters or coreference information during prediction. Supply those without irreversibly merging unrelated mentions or losing their offsets. Compare mention-first, conservative pre-resolution, and oracle-cluster variants on the same data.

Keep within-document and cross-document identity linking separate. Prefer a duplicate unresolved node over a high-impact false merge. A same-name match alone is insufficient for cross-document identity.

### Data contracts to implement before training

Use versioned Pydantic/dataclass records. Proposed new module: `src/nlp/contracts_v2.py`.

- `Document`: content hash, source name, extraction version, raw/normalized text, block/page map, language and OCR diagnostics.
- `Mention`: mention ID, document ID, exact text, type, normalized character range `[start,end)`, source ranges/page/bounding boxes when available, sentence/block IDs, detector/model version and score.
- `IdentityLink`: two mention IDs, decision, score, method/version and supporting evidence. Keep rejected/uncertain links inspectable.
- `RelationCandidate`: ordered mention IDs, context IDs, token alignment, retrieval reason, and whether either endpoint was truncated.
- `RelationClaim`: mention endpoints, relation type, polarity, modality, lifecycle status, time qualifiers, exact evidence spans, model/version, raw score, calibrated confidence if available, abstention reason and schema version.
- `CanonicalClaim`: canonical endpoints or transaction/event ID, member claim IDs, evidence and aggregation decision. Preserve disagreements.

Example annotation contract, using `Atlas acquired Beacon.`:

```json
{
  "schema_version": "2",
  "document_id": "example-doc",
  "text": "Atlas acquired Beacon.",
  "mentions": [
    {"id": "m1", "start": 0, "end": 5, "text": "Atlas", "type": "ORG"},
    {"id": "m2", "start": 15, "end": 21, "text": "Beacon", "type": "ORG"}
  ],
  "claims": [{
    "subject_mention_id": "m1",
    "object_mention_id": "m2",
    "relation": "ACQUIRED",
    "polarity": "affirmed",
    "modality": "asserted",
    "lifecycle": "completed",
    "evidence_spans": [[0, 22]]
  }]
}
```

This is a schema illustration, not training evidence. Validate offset/text round trips. Do not fake page coordinates for text-only inputs. Normalization must retain a mapping back to source text; otherwise citations will drift.

## 6. Ontology and financial semantics

Create a versioned ontology with definitions, direction, endpoint types, positive/negative examples, overlap rules and evaluation mappings.

| Proposed relation | Direction | Distinction requiring annotation |
|---|---|---|
| FOUNDED | founder → organization | founder versus executive or investor |
| EMPLOYED_BY | person → employer | current, former and proposed appointments |
| MEMBER_OF | member → organization | employment may coexist; do not force mutual exclusivity |
| OWNS | owner → owned entity | full/partial stake, beneficial ownership and time |
| SUBSIDIARY_OF | subsidiary → parent | do not infer from collaboration or investment alone |
| ACQUIRED | acquirer → target | signed agreement is not completed acquisition |
| INVESTED_IN | investor → investee | intent versus executed investment; ownership not automatic |
| ADVISED | adviser → client | identify deal and role; do not infer the other party is the client |
| FINANCED | financier → borrower/transaction | loan, underwriting and advisory roles differ |

This table is a starting proposal. Reconcile it with `config/relations.yaml` before adding labels. Train only labels with adequate reviewed examples; unsupported types must be explicit.

Use separate attributes for polarity (`affirmed/negated/unknown`), modality (`asserted/planned/conditional/rumored/unknown`) and lifecycle (`announced/signed/completed/cancelled/unknown`). Historical completion can remain affirmed/completed with a past time; it is not equivalent to current ownership. “No evidence” is not a negated relation. “No relation” is a candidate classification outcome, not a graph edge.

For amount, currency, stake, date, adviser roles and multiple transactions between the same companies, use a transaction/event object or claim node. Do not collapse all acquisitions between two organizations into one timeless edge. Phase the event schema after the basic claim contract, but do not discard status/evidence in the interim.

## 7. Dataset and annotation plan

### Existing resources

Re-DocRED files and provenance are in `data/external/redocred/`. Training originally had 3,053 documents; one train/dev overlap was excluded. Development and test each contain 500 documents. Keep their licensing, hashes, exclusions and mappings. They provide general relation examples, not complete finance supervision or PDF-layout evaluation.

### New representative data

Start with a reviewed pilot of roughly 100 distinct documents, then expand according to learning curves and per-label support rather than promising a fixed sufficient size. Include transaction announcements, completed-deal notices, cancellations, annual-report prose, employment/ownership disclosures, and documents with no target relations. Add scanned/table-heavy material only with explicit annotations and separate strata.

Track provenance, redistribution rights, domain, company groups, dates, template families and extraction method. Public availability alone does not establish redistribution rights. Do not send private PDFs to external annotation/model services without authorization.

Have annotators mark mentions, document identity clusters, relations, direction, assertion attributes and minimal sufficient evidence. Review a shared subset with two annotators, adjudicate disagreements, and revise guidelines before scaling. LLM/synthetic labels may bootstrap training only if separately identified; they are not independent validation truth.

**Crucial:** an unannotated pair is not necessarily a negative in incompletely annotated corpora. Use explicit no-relation examples or verified exhaustive annotation for the evaluated schema. Audit uncertain negatives; mask them from loss rather than teaching missing positives as false. This is a weakness to investigate in prior production-aware training.

### Splits

- Group related versions, near-duplicates, companies/deals and templates before splitting. Audit exact hashes, normalized text and approximate duplication.
- Allocate separate training, model-selection validation, threshold-calibration and locked-test roles; use grouped cross-validation if data is initially too small.
- Include a temporal/domain holdout where feasible. Prevent synthetic paraphrases of evaluation examples from entering training.
- Freeze split membership and hashes before training. A reviewer should protect final test labels until model and thresholds are frozen.
- Report per-label and negative-document support. Rare labels with insufficient evaluation support remain experimental.
- Do not use the dummy PDF, old development data or old test results as fresh generalization evidence.

## 8. Model development recipe

### Entity recognition and identity

Compare the current spaCy detector against a contextual token/span classifier fine-tuned on reviewed entity annotations. Preserve type vocabulary alignment and support multi-token organizations, abbreviations, nested spans where needed, and OCR corruption diagnostics. Train with representative extraction noise; do not inject arbitrary corruption without checking realism.

Evaluate the resolver independently using pairwise merge precision/recall plus a cluster metric such as B-cubed. Audit high-impact false merges. Treat type compatibility, exact offsets and self-link checks as integrity constraints; they are not learned evidence of a relationship.

### Relation prediction

Implement a contextual ordered-pair classifier with explicit subject/object markers and a no-relation outcome. Start with a reproducible modest encoder that fits available hardware; benchmark before selecting a larger architecture. Multi-label heads are appropriate when labels can coexist. Use loss masks for unknown labels and preserve direction.

Evaluate a document-level architecture with mention aggregation and contextual evidence pooling if cross-sentence errors dominate. [ATLOP](https://github.com/wzhouad/ATLOP) is a reference design for multi-label document relations, not a drop-in finance solution or a promised score. Its published benchmark results are not directly comparable to this pipeline's production-entity metrics.

Train on a mixture of gold mentions and realistic predicted mentions. For predicted-mention training, prefer out-of-fold NER predictions: do not generate unrealistically clean training predictions from a detector trained on the same examples. Include difficult same-type unrelated pairs, reversed arguments, co-mentioned companies, negation, proposals and multiple deals. Track sampling ratios and natural deployment prevalence.

Candidate generation must not require a hand-written relation verb. Measure candidate recall before training the classifier. Start with sentence/paragraph and adjacent-block contexts; add retrieval for distant mentions when needed. Record candidate-budget truncation and missing context rather than silently treating those cases as negatives. A fixed sliding window is a baseline, not full document reasoning.

Train assertion-status heads or a separately validated classifier. If status cannot be determined reliably, retain `unknown`; do not turn every predicted relation into a completed transaction. Evidence selection must identify source spans, not generate supporting text.

### Training controls

Record dataset/split/ontology hashes, tokenizer/checkpoint revision, package versions, seeds, feature/label order, losses, negative sampling, hardware, selected epoch and threshold method. Fit vocabulary/weights only on training data. Early stopping uses validation, never final test. Keep full learning curves and rejected runs.

Use a tiny training subset to verify the model can overfit as a wiring test, then separately measure generalization. Test label permutation, swapped endpoints, renamed entities, length limits, missing artifacts, and identical CPU/GPU decisions within a declared tolerance.

Available hardware was previously detected as an RTX 3050 Laptop GPU with 4 GB VRAM; verify locally. Start with a measured memory/latency smoke test, microbatches and gradient accumulation. Mixed precision or parameter-efficient tuning may help, but do not guarantee a large model fits. [PEFT's token-classification guide](https://huggingface.co/docs/peft/main/task_guides/token-classification-lora) is a reference for adapting the NER stage; it is not a ready-made relation pipeline. Do not silently rent compute or start a costly job.

The downloaded GLiREL checkpoint is licensed CC BY-NC-SA 4.0 according to its [model card](https://huggingface.co/jackboyla/glirel-large-v0#license). Do not treat it as a commercially unrestricted default. Audit code, base-model and weight licenses separately for every candidate.

## 9. Confidence, evidence and graph behavior

Model scores are not automatically probabilities of correctness. Fit calibration only on the calibration split, using deployment-like negatives. Report reliability plots/Brier score or suitable calibration metrics where support allows, and precision–recall/coverage curves. Threshold per label/status only when enough examples support it; otherwise abstain or keep the label experimental.

Store raw score separately from calibrated confidence. Repeated mention predictions are correlated: do not combine them as independent evidence using `1 - product(1-p)` without validation. Deduplicate evidence by claim/document/span while retaining legitimate independent sources.

Do not force edges onto isolated nodes or create RELATED_TO fallbacks to make the graph look complete. Display unresolved mentions and uncertain claims separately. Contradictory or changed claims must coexist with timestamps/provenance rather than silently overwriting each other.

## 10. Evaluation matrix and proposed acceptance gates

Run all candidates on the exact same document sets and mappings. Report errors and rejected documents; preserve gold denominators for failures. Never omit difficult documents because a backend failed.

Required evaluation modes:

1. Gold mentions + gold clusters → relation/status quality ceiling for that model.
2. Gold mentions + predicted clusters → identity-resolution impact.
3. Predicted mentions with exact gold alignment where possible → NER/candidate impact.
4. Full production PDF pipeline → end-to-end quality.
5. Ablations: resolution order, document recovery rules, context retrieval, negative sampling, status head and calibration.

Report strict typed mention P/R/F1; resolver metrics; candidate coverage; directed relation micro/macro and per-label P/R/F1; assertion-status confusion matrices; joint relation+status correctness; evidence-span quality; false edges per negative document; latency, memory, truncation and error rate. Add document-level bootstrap intervals and slice results for prose/OCR/tables, cross-sentence relations and unseen organizations. Relaxed span metrics are diagnostics, not replacements for strict metrics.

**Proposed gates to agree before the next test is opened:**

- Engineering: no unsupported fallback, no lost provenance, no corrupt spans, no silent model-load mismatch; required regression and integration tests pass.
- Research milestone: materially beat matched current baselines, with at least 70% precision and 30% recall on the agreed core scope. This is a proposed development target, not an achieved result.
- Default-promotion target: at least 90% precision and 60% recall overall on a locked representative finance test, with per-label support and uncertainty reported. Require a document-bootstrap lower precision bound of at least 80% and no high-risk slice hidden by the aggregate. These are proposed product targets, not universal standards.
- Proposed/completed and negated/affirmed confusion must have an explicit product-approved tolerance and sufficient test examples. Without it, financial event claims remain experimental even if relation F1 passes.
- Unsupported/rare labels cannot borrow the aggregate score as validation. If support is too small, label them unvalidated.
- Compare at matched precision as well as maximum F1. A model that predicts almost nothing must fail recall/coverage gates.

The previous 50% precision / 20% recall gate was only a minimum screen; it was never production-grade acceptance. Do not lower thresholds or gates after seeing a failed test to claim success.

## 11. Implementation milestones and deliverables

| Milestone | Deliverable | Exit condition |
|---|---|---|
| A — audit | Reproduced baselines, error taxonomy, environment manifest | Causes measured; data/evaluator bugs distinguished from model failures |
| B — contracts | v2 mentions, source mapping, claims and ontology | Round-trip offsets and status-preserving payload tests pass |
| C — data | Reviewed pilot, annotation guide, split manifests | Label definitions adjudicated; leakage and negative-label audit complete |
| D — entity stage | NER/resolver benchmark and conservative adapter | Measurable stage improvement without harmful merge regressions |
| E — relation stage | Trained contextual model, candidate/evidence/status metrics | Gold-entity and production-entity diagnostics pass agreed research gate |
| F — calibration | Frozen model and threshold manifest | Calibration done without final-test access; abstention behavior documented |
| G — integration | `contextual_v2` backend, versioned graph/API adapter | Offline inspection, persistence and compatibility checks pass |
| H — final evaluation | Locked-test report and release decision | Predeclared quality and resource gates pass; otherwise remain experimental |

Proposed files, not existing promises:

```text
src/nlp/contracts_v2.py
src/nlp/contextual_ner.py
src/nlp/contextual_relations.py
src/nlp/claim_aggregation.py
src/ingestion/source_map.py
config/ontology_v2.yaml
scripts/prepare_v2_dataset.py
scripts/train_contextual_ner.py
scripts/train_contextual_relations.py
scripts/calibrate_v2.py
scripts/evaluate_v2.py
tests/test_v2_contracts.py
tests/test_v2_alignment.py
tests/test_v2_pipeline.py
docs/annotation_guide_v2.md
models/contextual_v2/MODEL_CARD.md
```

Prefer smaller modules/PRs with runnable tests over one broad rewrite. Test prediction code with fake model outputs for deterministic contract checks, then test real artifacts separately; do not confuse the former with model accuracy.

Add `contextual_v2` explicitly to settings and offline inspection. Fail clearly when its artifact/schema is missing or incompatible. Do not silently fall back to rules. Preserve existing API behavior for the old backends, and version any new claim/event fields. Add migrations and rollback before changing stored graph structures. No destructive database rebuild is necessary for model experimentation.

The current global file lock prohibits parallel pipeline evaluations in one workspace. Run serially or redesign and test isolation first. Fit/inference scripts must not write evaluation documents into the production Neo4j database. Use a new intermediate-output directory per run.

## 12. Reproduction and operational checklist

Baseline commands, run from the repository root:

```powershell
git status --short
python -m pytest -q -p no:cacheprovider --basetemp output/test_tmp_handoff
python scripts/inspect_document.py Doc2Graph_Corporate_Finance_MA_Dummy.pdf --relation-backend rules --output output/handoff_rules_graph.json
python scripts/evaluate_learned_relations.py --dataset data/external/redocred/dev_revised.json --backend learned --output output/evaluation/handoff_learned_dev.json
```

Use a new `--basetemp` directory if an old one has permissions problems. The encoder additionally requires the optional project-local runtime and cached checkpoint; follow its model documentation. Record failures rather than silently installing different model versions or changing the global environment. Do not expose `.env`, API keys or database credentials in reports.

Historical source hashes no longer necessarily match the current working tree after subsequent integration work. Use saved snapshots where present or explicitly report that a historical snapshot cannot be reproduced. Never claim a hash matched without checking it.

Every implementation handoff should include: changed-file list, commands, model/data hashes, before/after results on matched sets, failure cases, label coverage, runtime costs, what remains unresolved, and an explicit release decision. Keep the default unchanged until acceptance is demonstrated.

## 13. Instructions to the reviewing/building model

Challenge this plan. Verify claims against code and reports. In particular, do not assume “resolve after relations” always wins, do not assume unannotated pairs are negatives, and do not assume a pretrained encoder solves domain shift.

First return an audit of this plan: confirmed facts, incorrect assumptions, missing requirements, and changes you recommend. Then implement milestone A and the smallest evidence-supported next milestone. Continue routine reversible engineering autonomously. If reviewed labels or a genuinely new test set are missing, build the contracts, tools and annotation workflow, then state that missing data blocks a quality claim. Do not fabricate annotation quality or claim training completed from code changes alone.

Do not add PDF-specific entity names, relation trigger lists, arbitrary graph-completion edges or an LLM extraction fallback to satisfy metrics. Integrity checks and a declared ontology are allowed; they must not masquerade as learned semantic improvement.

Success means a measured, reproducible improvement on unseen representative documents with defensible evidence and status—not a more connected graph, a larger model, or a passing unit-test count.
