# First implemented SemEval SVM baseline

Implemented a strict marker/span parser, binary lexical features, linear SVMs, development search, full-training refit, test evaluation and marked-pair prediction CLI. This is a UTD-inspired lexical baseline; POS, dependency, WordNet, semantic-role and external-knowledge features are not implemented yet.

Data: all 8,000 official training examples and 2,717 official test examples, obtained from the JoelNiklaus/SemEval2010Task8 release mirror. The archive provenance and SHA256 are in data/external/semeval2010/download.json. That mirror hosts the original task archive including the Perl scorer. Dataset licensing terms were not independently established in this build; retain original release notices and establish permitted redistribution before packaging data.

Development uses one fixed StratifiedGroupKFold partition, grouping exact identical text. The 19 directed labels are severely imbalanced: Entity-Destination(e2,e1) has only one training example, which triggers sklearn's split warning. The exact memberships and all 16 searched configurations are saved in models/utd_svm/development.json. This is not a cross-validation performance estimate. Near-duplicate grouping beyond exact text is not implemented.

The best development configuration was the direct 19-class baseline, C=0.1, balanced class weights, at 76.72% macro F1. The UTD-style two-stage alternative reached 76.71% on the same development partition. These point estimates are effectively tied; no statistical superiority is claimed. The final selected direct classifier was refit on all 8,000 training examples before test evaluation.

Official scorer test result: **78.64% direction-aware nine-family macro F1**. Accuracy including Other: **73.94%**. Python macro F1: 0.7864064445816946, agreeing with the official scorer's displayed rounding. The official output is output/utd_svm_test/official_score.txt; metrics and ID-aligned predictions are saved alongside it. No test-driven tuning was performed in this build. Subsequent tuning must disclose that this test result has now been inspected.

These results concern supplied noun pairs, not automatic PDF or graph extraction. The model is not a faithful full UTD reproduction, and its score does not establish improvement over the unrelated finance/news pipelines. Production defaults were not changed.

## Commands (PowerShell, repository root)

```powershell
python scripts/predict_utd_svm.py 'The <e1>water</e1> is in the <e2>bottle</e2>.'
pytest -q tests/test_utd_svm.py --basetemp=output/pytest_utd_build
```

To reproduce training/evaluation, use new output directories; scripts refuse to overwrite previous models/results:

```powershell
python scripts/train_utd_svm.py --output models/utd_svm_reproduction
python scripts/evaluate_utd_svm.py --model models/utd_svm_reproduction --output output/utd_svm_reproduction
perl data/external/semeval2010/semeval2010_task8_scorer-v1.2.pl output/utd_svm_reproduction/predictions.txt output/utd_svm_reproduction/gold.txt
```

Remaining work: stronger linguistic features with development-only ablations, richer feature/resource manifests, scorer parity on adversarial fixtures beyond the actual test predictions, confidence intervals and independently reviewed automatic candidate extraction. Current code tests cover marker validation, repeated phrase offsets/Unicode and wrong-direction penalties. No live graph, PDF integration or neural component was added.
