# Model Card: Contextual Relation Model v2 (`contextual_v2`)

## Model Summary
- **Model Name:** Contextual Relation Classifier v2 (`contextual_v2`)
- **Version:** 2.0.0
- **Task:** Directional, typed multi-label relation extraction and assertion attribute classification (`polarity`, `modality`, `lifecycle`).
- **Architecture:** Bidirectional recurrent contextual sequence encoder with explicit entity markers (`[SUBJ_START] ... [SUBJ_END]`, `[OBJ_START] ... [OBJ_END]`), directional asymmetric span pooling (`[h_subj; h_obj; h_subj - h_obj; h_subj * h_obj; h_cls]`), multi-label binary relation head, and 3 assertion heads.
- **Hardware Profile:** Trained locally on RTX 3050 Laptop GPU (CUDA). Runs fully offline on CPU or GPU without external network access.

## Ontological Scope
Evaluates 9 directed relations:
1. `FOUNDED` (founder -> organization)
2. `EMPLOYED_BY` (person -> employer)
3. `MEMBER_OF` (member -> organization)
4. `OWNS` (owner -> owned_entity)
5. `SUBSIDIARY_OF` (subsidiary -> parent)
6. `ACQUIRED` (acquirer -> target)
7. `INVESTED_IN` (investor -> investee)
8. `ADVISED` (adviser -> client)
9. `FINANCED` (financier -> borrower)

Assertion attributes:
- `polarity`: `affirmed`, `negated`, `unknown`
- `modality`: `asserted`, `planned`, `conditional`, `rumored`, `unknown`
- `lifecycle`: `announced`, `signed`, `completed`, `cancelled`, `unknown`

## Training Controls & Loss Masking
- **Training Data:** Prepared Re-DocRED v2 (`schema_version: 2.dataset.1`) and reviewed finance annotations (`schema_version: 2.dataset.2`).
- **Supervision Gating:** Unannotated pairs are strictly masked (`loss_mask = False`). Loss is computed via `BCEWithLogitsLoss` exclusively over supervised positions (`loss_mask = True`). Missing positives are never taught as negative zeros.
- **Directionality:** Entity marker positions and vector difference ensure asymmetric handling of `(A, B)` vs `(B, A)`.

## Calibration & Abstention Policy
- Calibration was performed strictly on the isolated calibration split (`data/prepared/v2_redocred/calibration.jsonl`, 315 documents, 12,111 candidate instances).
- The locked test split (`regression_test.jsonl` / `locked_test.jsonl`) was strictly protected and never accessed during training or threshold fitting.
- Finance-specific relations (`ACQUIRED`, `INVESTED_IN`, `ADVISED`, `FINANCED`) that lack exhaustive general corpus calibration support are designated `experimental_low_support` and enforce high-confidence abstention.
