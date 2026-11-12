# Copy/paste into Gemini Flash

Implement `docs/UTD_SVM_BUILD_GUIDE.md` in this repository. Read it fully, inspect applicable repository instructions and preserve unrelated changes. Our agreed scope is classical non-DL relation classification using ALL SemEval-2010 Task 8 relation families and all official training examples, with no finance/news restriction.

Target UTD's feature-based SVM design: relation-family classification followed by relation-specific direction classifiers. First read the original paper and create a feature/resource reproduction ledger. Start with a runnable lexical baseline, then add available non-neural linguistic/semantic features with measured ablations. If original resources are missing, explicitly call the result UTD-inspired; never claim faithful reproduction or the paper's 82.19% score without evidence.

Use official training data for development selection, freeze the configuration, refit on all 8,000 official training examples, then evaluate on the untouched 2,717 official test examples with the official direction-aware nine-family macro-F1 scorer. Do not fit vocabulary or thresholds on test data. Do not fabricate or automatically certify gold annotations.

Begin authorized reversible work after a short inventory. Download permitted data/resources as needed and record provenance. Build the parser, feature pipeline, training, marked-pair prediction CLI, scorer verification, model artifacts and tests. Measure results instead of promising a target. Keep automatic PDF noun-pair extraction as a separately evaluated later phase; the benchmark supplies the noun pairs.

Do not change the production default based solely on classification scores. Finish with actual runnable commands, changed files, model/resource hashes, feature coverage, development and final-test results, test output and explicit limitations. Report missing dependencies or human-review requirements without inventing completion. Full instructions and acceptance criteria are in the guide.
