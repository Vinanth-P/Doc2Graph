# Model improvement follow-up — 11 November 2026

## Outcome

**The extraction-quality problem is not fixed. Neither new candidate passed validation, and neither was made the default.** Fewer false edges alone are not an improvement when most correct edges disappear too. No PDF-specific relation rules or LLM inference calls were added in this follow-up.

## Production-aware retraining

The first learned model was trained and thresholded on gold entities but deployed on noisy extracted entities. To address that mismatch, `scripts/train_robust_relations.py` trained new weights using both production and annotated entity-pair contexts. All negative pairs were retained; balanced class weighting was removed. Exact typed span alignment labels unmatched/ambiguous production entities as negatives.

Training used 3,052 documents and 158,223 pairs. The upstream train/dev overlap (`Lark Force`) remained excluded. One production training document (`Eugène Olivier`) was rejected by the existing English-language gate; the rejection was recorded. Development had no rejections. Gold-entity training examples for that document remain included.

Even-index development documents selected thresholds with at least 50% precision and five unique correct predictions per label. Only MEMBER_OF qualified; the other four labels abstained. On the separate odd-index development subset, precision did not hold and recall collapsed:

| Same 250 validation documents | Correct | False positives | Missed | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Previous learned model | 27 | 428 | 306 | 5.93% | 8.11% | 6.85% |
| Production-aware retraining | 1 | 8 | 332 | 11.11% | 0.30% | 0.58% |

The lower false-positive count is chiefly abstention, not a successful fix. The weights are preserved as `models/robust_relations.joblib`; full metadata is in `models/robust_relations.training.json`. They have not replaced the original artifact. Protocol: [robust_protocol.md](robust_protocol.md).

## Pretrained contextual encoder

The second candidate uses [GLiREL](https://github.com/jackboyla/GLiREL), a local discriminative DeBERTa relation encoder. Its pinned checkpoint is `jackboyla/glirel-large-v0` revision `40a523e12a8432d6da364cf2a195a28755ff04d3`. Strict state-dictionary loading passed, ruling out missing/unexpected checkpoint keys; that does not prove semantic quality or complete runtime equivalence to its training environment.

The adapter corrects founder direction (person to organization), handles inclusive input versus exclusive output token spans, retains evidence, rejects canonical self-links, and enforces schema types. Evaluation used a fixed 0.75 score threshold, bounded overlapping token windows, production NER/resolution, and the same five-label metric. All 500 development documents completed with **zero errors**.

| Same 500 development documents | Correct | False positives | Missed | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Previous learned model | 47 | 880 | 566 | 5.07% | 7.67% | 6.10% |
| Pretrained encoder | 17 | 393 | 596 | 4.15% | 2.77% | 3.32% |

The encoder is worse on this diagnostic. Its simple smoke test also produced incorrect links. It remains an explicitly experimental backend, not a recommended replacement. See [raw results and false-positive evidence](encoder_dev.json), [smoke test](encoder_probe.json), and [runtime documentation](../../models/encoder_relations.README.md). The checkpoint model card specifies CC BY-NC-SA 4.0; it is not a commercial deployment recommendation.

## Interpretation and practical state

Code validation: **192 tests passed, 1 optional integration test skipped**. `git diff --check` reported no whitespace errors. Passing implementation tests does not override the poor extraction metrics above.

These are previously observed development documents, not a fresh holdout. The pretrained encoder's training overlap with this benchmark is unknown. The old test split was not used for further threshold tuning or rerun for these failed candidates. Neither candidate meets the earlier 50% precision / 20% recall gate, and no finance-specific polarity/modality validation has been established.

The application still defaults to `rules`; that backend's poor external performance has not been resolved either. The new training/evaluation infrastructure makes failures measurable and reproducible, but it would be misleading to describe the graph extractor as fixed or reliable on new PDFs. A successful replacement still needs stronger entity recognition/resolution and relation learning, with validation covering negative, proposed, conditional and cross-sentence relationships.

Reproduction:

```powershell
python scripts/train_robust_relations.py
python scripts/evaluate_encoder_relations.py
python -m pytest -q -p no:cacheprovider --basetemp output/test_tmp_followup
```

The first command retrains the separate robust artifact. The encoder command needs the project-local optional runtime and cached checkpoint; it performs no model downloads. The model and cache directories are excluded from Git where appropriate. Earlier model artifacts and reports are retained.
