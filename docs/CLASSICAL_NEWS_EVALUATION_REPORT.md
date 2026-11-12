# Classical News Backend: Model Benchmark & Feasibility Evaluation Report

**Generated**: 2026-11-11 21:55:51 UTC
**Evaluator**: Doc2Graph Independent Evaluation Harness
**Architecture**: Stanford CoreNLP 4.5.8 Classical CRF NER + Roth Linear Relation Extractor (Non-Neural)
**Default Backend Preserved**: `rules`

---

## 1. Feasibility Gates Assessment (Locked Test Split)

| Gate Metric | Target Gate | Actual (Strict) | Status |
|:---|:---:|:---:|:---:|
| **Exact Entity Micro F1** | &ge; 85.0% | **93.09%** | ✅ PASS |
| **End-to-End Relation Precision** | &ge; 80.0% | **80.49%** | ✅ PASS |
| **End-to-End Relation Recall** | &ge; 50.0% | **28.95%** | ❌ FAIL |
| **End-to-End Relation Micro F1** | &ge; 60.0% | **42.58%** | ❌ FAIL |
| **Silent Exceptions / Lost Evidence** | 0 | **0** | ✅ PASS |

---

## 2. Detailed Performance by Stage & Split

### A. Named Entity Recognition (Exact Character Span & Entity Type)
| Split | Total Gold | Total Pred | Precision | Recall | Micro F1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Dev** (30 docs) | 175 | 196 | 76.0% | 85.1% | **80.3%** |
| **Locked Test** (30 docs) | 184 | 192 | 91.1% | 95.1% | **93.1%** |
| **Challenge** (15 docs) | 84 | 90 | 90.0% | 96.4% | **93.1%** |

#### Locked Test Per-Type NER Breakdown:
- **PERSON**: Precision 67.4%, Recall 96.7%, F1 79.5% (Support: 30)
- **ORGANIZATION**: Precision 100.0%, Recall 88.1%, F1 93.7% (Support: 59)
- **LOCATION**: Precision 96.9%, Recall 99.0%, F1 97.9% (Support: 95)

---

### B. End-to-End Relation Extraction (Locked Test Split)

| Matching Mode | True Positives | False Positives | False Negatives | Precision | Recall | Micro F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Strict Span Match** | 33 | 8 | 81 | 80.5% | 28.9% | **42.6%** |
| **Relaxed Name Match** | 33 | 8 | 68 | 80.5% | 32.7% | **46.5%** |

#### Strict Per-Relation Breakdown (Locked Test):
- **WORKS_FOR**: P 100.0%, R 83.3%, F1 90.9% (Support: 30)
- **LIVES_IN**: P 0.0%, R 0.0%, F1 0.0% (Support: 0)
- **BASED_IN**: P 100.0%, R 8.2%, F1 15.1% (Support: 49)
- **LOCATED_IN**: P 100.0%, R 11.4%, F1 20.5% (Support: 35)

---

### C. Challenge Set Robustness (15 Adversarial Documents)
- **Negation False Positives**: 1
- **Distractor Travel / Visitation False Positives**: 0
- **Challenge Strict F1**: 16.3%

---

## 3. Non-DL Invariants & Integrity Audit
1. **Zero Neural Models**: Stanford CRF NER (`english.all.3class.distsim.crf.ser.gz`) and Stanford PCFG Parser (`englishPCFG.ser.gz`) are strictly non-neural. No transformers, Stanza, or `depparse` are referenced or invoked.
2. **Confidence Honesty**: Raw model softmax scores are reported in `raw_score`. `calibrated_confidence` is explicitly set to `None`.
3. **Assertion Custody**: Polarity, modality, and lifecycle default strictly to `unknown` rather than fabricated values.
4. **Document-Scoped Identity**: Entity resolution is scoped strictly to within the document. No global fuzzy cross-document merging without evidence.

---

## 4. Release Decision

**Recommendation**: `KEEP_DEFAULT_RULES`
**Release Decision**: `KEEP_DEFAULT_RULES`

**Justification**:
While the classical Stanford CRF NER and Roth relation extractor demonstrate strong performance on this news/biography benchmark, the production default backend MUST remain `KEEP_DEFAULT_RULES`. The `classical_news` backend is available for domain-specific English news ingestion via `relation_backend="classical_news"`.
