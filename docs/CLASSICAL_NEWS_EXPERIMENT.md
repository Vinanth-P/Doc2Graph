# Experiment Specification: Classical Pretrained ML for English News & Biography

## 1. Overview & Objective
This experiment investigates building an isolated English news/biography document-to-graph backend using verified, pretrained, non-neural statistical machine learning models.
The primary candidate stack is **Stanford CRF NER** and the legacy **Stanford Supervised Relation Extractor** (Roth & Yih linear/MaxEnt classifier), backed by a classical non-neural constituency parser (English PCFG).

This investigation is governed by strict feasibility gates: model compatibility and empirical extraction quality must be demonstrated on authentic news/biography text before integrating into the production application or database.

## 2. Invariants & Scope Boundaries
1. **Strict Non-Neural Path**:
   - Zero transformers, zero neural parsers (`depparse`), zero neural coreference, zero Stanza neural models, and zero LLM calls in this backend's inference path.
   - Permitted annotators: `tokenize`, `ssplit`, `pos` (MaxEnt left3words), `lemma`, `ner` (3-class CRF: PERSON, ORGANIZATION, LOCATION), `parse` (classical English PCFG), and `relation` (legacy supervised relation extractor).
2. **Domain Isolation**:
   - The news/biography domain is completely isolated from the existing corporate finance domain.
   - The default production backend remains `rules` (`settings.relation_backend = "rules"`).
   - All existing finance models, artifacts, evaluation reports, and tests are strictly preserved.
   - News extraction metrics are **never** compared to finance/Re-DocRED metrics as a matched baseline.
3. **No Fabricated Labels**:
   - Machine-generated outputs will never be claimed as human-reviewed truth.
   - Tests passing execution checks will never be presented as proof of extraction accuracy.
   - If human-reviewed gold annotations are unavailable for held-out evaluation, the evaluation tooling will be delivered while reporting the benchmark as unreviewed.

## 3. Product Ontology & Mapping Proposal
The target domain covers sentence-local relationships between People, Organizations, and Locations:

| Native Stanford Label | Proposed Graph Label | Subject Type | Object Type | Direction Semantics |
|-----------------------|----------------------|--------------|-------------|---------------------|
| `Work_For` | `WORKS_FOR` | `PERSON` | `ORGANIZATION` | Person is employed by Organization |
| `Live_In` | `LIVES_IN` | `PERSON` | `LOCATION` | Person lives/resided in Location |
| `OrgBased_In` | `BASED_IN` | `ORGANIZATION` | `LOCATION` | Organization is based in Location |
| `Located_In` | `LOCATED_IN` | `LOCATION` | `LOCATION` | Location is located within Location |
| `None` | (No Edge) | Any | Any | Absence of supported semantic relation |

### Exclusions & Conservative Guardrails
- **Directionality**: Never infer direction from mention order. If argument types do not match the ontology, reject the edge.
- **No Over-Interpretation**: `WORKS_FOR` is never remapped to ownership, founding, or advisory.
- **Unsupported Classes**: Acquisitions, investments, funding, parent-subsidiary, and family relations are outside this backend's scope.
- **Assertion Status**: Without a verified assertion classifier, polarity, modality, and lifecycle default to `"unknown"`. Predictions are presented in the UI as "extracted relationship hypotheses", not real-world confirmed facts.
- **Document Scope**: Entity resolution is initially document-scoped; equal surface strings alone do not establish global identity across distinct entities or documents.

## 4. Pinned Dependency & Component Inventory

### A. Java Runtime Environment (Isolated Portable Runtime)
- **Component**: Eclipse Temurin OpenJDK 17 LTS (HotSpot)
- **Version**: `17.0.11+9`
- **Distribution**: Windows x64 ZIP (`OpenJDK17U-jdk_x64_windows_hotspot_17.0.11_9.zip`)
- **Source**: Official Eclipse Adoptium release (`adoptium/temurin17-binaries`)
- **Direct URL**: `https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.11%2B9/OpenJDK17U-jdk_x64_windows_hotspot_17.0.11_9.zip`
- **Content-Length**: `190,109,030 bytes`
- **Target Location**: `.runtime/jdk-17.0.11+9/`
- **License**: GNU General Public License v2 with Classpath Exception (GPLv2+CE)

### B. CoreNLP Libraries & Models (Official Maven Central Release)
- **Component**: Stanford CoreNLP Library & Models
- **Version**: `4.5.8` (released December 2024)
- **Source**: Official Maven Central Repository (`edu.stanford.nlp`)
- **License**: GNU General Public License v3 (GPLv3)
- **Artifacts**:
  1. `stanford-corenlp-4.5.8.jar` (`9,333,318 bytes`)
     - SHA1: `34f07a1262d08a54c86cb31797c5e7b2b00d4023` (from `stanford-corenlp-4.5.8.jar.sha1`)
     - URL: `https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/4.5.8/stanford-corenlp-4.5.8.jar`
  2. `stanford-corenlp-4.5.8-models.jar` (`474,409,531 bytes`)
     - SHA1: `cf45258f7009dfbd46452594a973a5a7b8e19e07` (from `stanford-corenlp-4.5.8-models.jar.sha1`)
     - URL: `https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/4.5.8/stanford-corenlp-4.5.8-models.jar`
  3. Supporting Runtime Jars (Maven Central):
     - `joda-time-2.13.0.jar`
     - `xom-1.3.9.jar`
     - `ejml-core-0.39.jar`, `ejml-ddense-0.39.jar`, `ejml-simple-0.39.jar`
     - `slf4j-api-1.7.36.jar`, `slf4j-simple-1.7.36.jar`
     - `commons-lang3-3.14.0.jar`
     - `jakarta.json-1.1.6.jar`

### C. Specific Model Resource Paths Inside Jar
- **NER**: `edu/stanford/nlp/models/ner/english.all.3class.distsim.crf.ser.gz` (Feature-based CRF)
- **Parser**: `edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz` (Classical non-neural PCFG)
- **POS**: `edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger`
- **Relation Classifier**: `edu/stanford/nlp/models/supervised_relation_extractor/roth_relation_model_pipelineNER.ser`

## 5. Feasibility Gates & Project Targets

| Metric | Target Gate | Action on Failure |
|--------|------------:|-------------------|
| Runtime Loading & Non-DL Audit | 100% Pass | Halt integration immediately; report missing resources or neural leaks. |
| Native Relation Annotation Output | Verified | Halt integration if relation extractor fails to emit structured spans/labels. |
| UTF-16 Code-Unit to Unicode Code-Point Mapping | 100% Accuracy | Fix span mapping before benchmark evaluation. |
| Exact Entity Micro F1 (3-Class) | >= 85.0% | Review NER errors; if fundamental, stop dependent relation pipeline. |
| End-to-End Relation Precision | >= 80.0% | Halt promotion to production; mark experimental or reject. |
| End-to-End Relation Recall | >= 50.0% | Evaluate coverage; record false negatives. |
| End-to-End Relation Micro F1 | >= 60.0% | Required for candidate promotion consideration. |
| Unhandled Exceptions / Lost Evidence | 0 | Fail-closed runtime error handling. |

## 6. Execution Plan & Milestones Status
1. **Milestone 0: Inventory and Specification** (Completed)
   - Created this experiment specification, inventorying dependencies, models, ontology mapping, and predeclared feasibility gates.
2. **Milestone 1: Portable Non-Neural Runtime Stack & Bridge** (Completed)
   - Verified OpenJDK 17 portable runtime and CoreNLP 4.5.8 Maven Central jars.
   - Compiled lightweight Java JSONL bridge (`StanfordClassicalBridge.java`) and verified non-neural execution.
   - Verified UTF-16 code units to Unicode code points span mapper with emoji support.
   - All 7 contract tests passed in `tests/test_classical_news_contracts.py`.
3. **Milestone 2: Dataset Preparation & Review Workflow** (Completed)
   - Implemented `scripts/prepare_news_review.py`.
   - Created frozen splits (`dev`: 30 docs, `locked_test`: 30 docs, `challenge`: 15 docs) with strict group isolation and SHA256 integrity in `data/classical_news/splits/split_manifest.json`.
   - Generated review packets with `label_status` (`positive`, `explicit_negative`, `unknown`) and reviewer tracking in `data/classical_news/review_packets/`.
4. **Milestone 3: Feasibility Benchmarking** (Completed)
   - Implemented `scripts/evaluate_classical_news.py`.
   - Evaluated exact entity extraction (3-class CRF) and end-to-end directed relation extraction across all 75 documents.
   - Computed document-level 95% bootstrap confidence intervals and recorded results in `output/classical_news_evaluation.json`.
5. **Milestone 4 & 5: Backend & Graph Semantics Integration** (Completed)
   - Implemented `src/nlp/classical_news.py`, `src/nlp/stanford_classical_client.py`, and `config/ontology_classical_news.yaml`.
   - Connected `classical_news` to `src/pipeline.py` with document-scoped entity resolution, explicit claim IDs, and unknown assertion tracking.
   - Verified integration in `tests/test_classical_news_integration.py`.
6. **Milestone 6: Regression & Feasibility Gate Decision** (Completed)
   - Ran full test suite: **274 passed, 1 skipped in 17.57s**.

## 7. Empirical Benchmark Results & Feasibility Decision

### Empirical Evaluation Summary (`output/classical_news_evaluation.json`)

| Split | Metric Category | Precision | Recall | Micro F1 | 95% Bootstrap CI (F1) | Feasibility Gates Status |
|---|---|---|---|---|---|---|
| **Dev** (30 docs) | Exact NER (3-class) | 82.68% | 78.07% | **80.32%** | [74.5%, 85.8%] | Failed target (>= 85.0%) |
| | End-to-End Relations | 48.00% | 11.01% | **17.91%** | [8.8%, 27.3%] | Failed target (>= 60.0%) |
| **Locked Test** (30 docs) | Exact NER (3-class) | 91.80% | 89.60% | **90.69%** | [86.7%, 94.4%] | Passed target (>= 85.0%) |
| | End-to-End Relations | 74.42% | 28.07% | **40.76%** | [27.8%, 53.6%] | Failed target (>= 60.0%) |
| **Challenge** (15 docs) | Exact NER (3-class) | 90.00% | 96.43% | **93.10%** | [87.5%, 97.7%] | Passed target (>= 85.0%) |
| | End-to-End Relations | 30.77% | 11.11% | **16.33%** | [3.8%, 29.6%] | Failed target (>= 60.0%) |

### Per-Relation Analysis (Locked Test Split)
- **`WORKS_FOR`**: **F1: 90.91%** (Precision: 100.0%, Recall: 83.33%, Support: 30, TP: 25, FP: 0, FN: 5). The pretrained model demonstrates excellent performance on explicit corporate/organizational employment relations.
- **`BASED_IN`**: **F1: 15.09%** (Precision: 100.0%, Recall: 8.16%, Support: 49, TP: 4, FP: 0, FN: 45). High precision but very low recall; the legacy Roth model fails to trigger on standard corporate headquarters phrases.
- **`LOCATED_IN`**: **F1: 20.51%** (Precision: 100.0%, Recall: 11.43%, Support: 35, TP: 4, FP: 0, FN: 31). High precision but severe recall degradation.
- **`LIVES_IN`**: **F1: 0.00%** (Precision: 0.0%, Recall: 0.0%, Support: 0, TP: 0, FP: 8, FN: 0). Generates spurious predictions for individuals mentioned in locations.

### Feasibility Gate Decision
- **Gate Outcome**: **FAILED** (End-to-End Relation Micro F1 on locked test is 40.76%, well below the predeclared 60.0% threshold; Recall is 28.07% vs >= 50.0% target).
- **Production Recommendation**: **`KEEP_DEFAULT_RULES`**
  - As mandated by `docs/CLASSICAL_NEWS_BUILD_GUIDE.md`, the production default remains `"rules"`.
  - The `classical_news` backend is preserved in the repository as an isolated, verified experimental backend (`settings.relation_backend = "classical_news"`), but is **not promoted** to the default production pipeline.

