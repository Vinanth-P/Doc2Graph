# Milestone H: Final Evaluation Report and Release Decision

**Date:** 2026-11-11T18:05:44Z  
**Evaluated Backend:** `contextual_v2`  
**Dataset:** `regression_dev.jsonl` (100 documents)

## 1. Executive Summary & Release Decision

> [!IMPORTANT]
> **Release Decision:** `KEEP_DEFAULT_RULES`  
> **Recommendation:** Keep 'rules' as the default production backend. Maintain 'contextual_v2' as an opt-in experimental backend until independently sealed corporate finance holdout is annotated.

The contextual neural relation model (`contextual_v2`) successfully implements typed directional relation extraction with entity marker pooling, assertion classification (`polarity`, `modality`, `lifecycle`), strict masked loss supervision, and calibrated thresholds.

However, in accordance with the predeclared acceptance gates in `MODEL_REDESIGN_HANDOFF.md`, **the default production backend remains `rules`**. The `contextual_v2` model is maintained as an isolated, opt-in backend for research and pilot evaluation.

---

## 2. Evaluation Results (Oracle Entity Ceiling)

| Metric | Point Estimate | Bootstrap 95% CI | Baseline (Oracle Dev) |
|---|---|---|---|
| **Precision** | **4.98%** | [2.68%, 7.82%] | 24.55% |
| **Recall** | **24.80%** | [14.47%, 37.67%] | 31.16% |
| **F1 Score** | **8.30%** | [4.68%, 12.80%] | 27.46% |
| **True Positives** | 31 | - | 191 |
| **False Positives** | 591 | - | 587 |
| **False Negatives** | 94 | - | 422 |

*Average Latency:* 32.8 ms / document.

---

## 3. Acceptance Gate Audit

| Gate | Target Requirement | Achieved | Status |
|---|---|---|---|
| **Engineering Gate** | No lost provenance, round-trip spans, 0 corrupt IDs, all unit tests pass | 236+ passed tests, 0 corrupt IDs | **PASSED** |
| **Research Gate** | Beat baseline with P >= 70% and R >= 30% | P = 4.98%, R = 24.80% | **FAILED** |
| **Default Promotion Gate** | P >= 90%, R >= 60%, P_low >= 80%, sealed holdout, >=100 finance docs | Evaluated on pilot fixture; holdout not yet independently sealed | **FAILED** |

### Audit Notes:
- Research gate (P>=70%, R>=30%) not met: achieved P=4.98%, R=24.80%.
- Default-promotion gate failed: Performance (P=4.98%, R=24.80%) below 90%/60% target; Bootstrap 95% lower precision bound (2.68%) below 80%; Independently sealed release holdout not yet available; Finance review support (5) below 100 documents.
