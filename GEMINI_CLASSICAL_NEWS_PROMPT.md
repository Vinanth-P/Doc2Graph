# Handoff prompt: classical ML news-domain pivot

Read `docs/CLASSICAL_NEWS_BUILD_GUIDE.md` completely, then inspect the repository and applicable instructions. Implement its staged plan for an isolated English news/biography backend using pretrained non-neural Stanford CRF NER and the legacy supervised relation extractor. The domain change is intentional; do not continue finance neural training as the main task.

First verify the pinned release actually supplies a working pretrained relation model and that every inference component is non-neural. Produce native model outputs and an independent benchmark before full pipeline/UI integration. Do not substitute OpenIE, KBP, a transformer, Stanza neural models or an LLM without discussing the scope change.

Preserve unrelated edits, finance artifacts, old reports, user data and the default backend. Read the latest review in `docs/CONTEXTUAL_V2_CORRECTIVE_FOLLOWUP.md` for identity, custody and storage pitfalls relevant to reused code. Reproduce and fix those boundaries where used, without expanding into unrelated retraining.

Begin authorized reversible implementation after a brief inventory. Download official dependencies as needed, pin versions/hashes/licenses, and record actual resource requirements. Implement the adapter and evaluation tooling while collecting annotation requirements. Never claim machine-generated labels are human-reviewed, invent quality results, or treat passing tests as extraction accuracy.

Follow the guide's feasibility gates. If pretrained model compatibility or measured quality fails, report it and stop the dependent integration work; finish independent useful tasks. Return exact runnable commands, per-stage and per-label P/R/F1, tests, provenance, error examples and an explicit release decision. Mark every unrun benchmark or integration test as unrun.
