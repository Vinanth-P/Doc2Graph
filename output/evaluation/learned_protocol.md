# Learned candidate protocol

Before inspecting the test split:

- Train a supervised entity-pair classifier on the Re-DocRED training split (3,053 source documents), excluding duplicate train documents and any title/text overlaps with dev. No test data is loaded by training code. Record exclusions explicitly.
- Use the already inspected 500-document development split for per-label threshold selection only. Fit vocabulary, IDF and weights on training data only.
- Keep the same five mapped labels and entity-type constraints as the external evaluation. Candidate generation is independent of dependency/verb patterns and includes all ordered, type-compatible entity pairs, including cross-sentence pairs.
- Mask entity identities in context. Use up to two closest mention-pair contexts, subject/object role markers, type and sentence distance. Train TF-IDF word 1-3 grams and one-vs-rest logistic regression, C=4, balanced class weights, fixed seed. Retain all positive training pairs and sample at most max(20, 8 times positives) negative pairs per document. Development and test candidates are not sampled.
- Choose each threshold on development F1 with at least three true positives and threshold >= 0.05; no eligible threshold disables that label. Keep unsupported gold relationships in the recall denominator.
- Freeze the artifact, features, thresholds and inference code before downloading/evaluating the test split. Do not tune or retrain after examining test results.
- Evaluate both rules and learned candidate through the same production NER/entity-resolution pipeline. Also report gold-entity diagnostic results, exact entity metrics, per-label P/R/F1, errors and coverage. Evaluate all 500 test documents, including documents without mapped positive facts.
- Audit test titles/text for train/dev overlap without refitting. Report the full split and a disjoint subset separately if overlaps exist; use the disjoint subset for generalization claims.
- Promotion: an overall F1 gain alone is insufficient to make the candidate the default. Require precision >= 50% and recall >= 20% end-to-end on the five mapped types before even considering default promotion; finance-specific polarity/modality and unseen-PDF validation remain required. Otherwise keep the candidate explicitly opt-in and document limitations.

This is a lightweight classical supervised baseline, not a transformer or an LLM service. It trains relationship labels from text, unlike the original scorer which merely scores rule-generated triples. New labels such as acquisition/advisory are outside this training/evaluation scope. No claim of calibrated probabilities or robust handling of negated/planned financial events is made.
