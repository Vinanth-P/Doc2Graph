# Experimental Stanford + SVM pipeline

Enable `RELATION_BACKEND=semeval_svm` for the application, or select `--relation-backend semeval_svm` in scripts/inspect_document.py. Rules remains the default. `UTD_SVM_MODEL_PATH` can override models/utd_svm. The selected local model must match its manifest hash. Only load trusted local joblib models.

Flow: existing text/PDF ingestion -> Stanford classical sentence segmentation, CRF named entities and PCFG noun phrases -> lexical SVM -> document-scoped concept nodes and directed relation edges. The SVM backend branches before the legacy spaCy preprocessing/extraction path. The shared Stanford bridge also computes legacy Roth annotations for its news backend; SVM predictions do not consume them.

The Java bridge must be recompiled after source updates. On this workspace:

```powershell
& '.runtime/jdk17/bin/javac.exe' -encoding UTF-8 -cp '.runtime/stanford_classical/*;.runtime/stanford_classical/lib/*' -d java/stanford-classical-bridge/build java/stanford-classical-bridge/src/StanfordClassicalBridge.java
python scripts/inspect_document.py example.pdf --relation-backend semeval_svm --output output/svm_graph.json
```

Restart running application workers after recompiling so persistent Java processes use the new protocol. An old bridge without noun_phrases fails explicitly. Java and model dependencies use the existing Stanford runtime installation.

Candidate policy: innermost parsed NP constituents containing noun-tagged tokens, with leading determiners removed. Evaluate non-overlapping pairs in sentence order, at most 200 characters apart and 100 pairs per sentence. Native direction labels determine graph endpoints; Other emits no edge. Pipeline results expose evaluated/skipped pair counts. These are general candidate rules, not validated extraction thresholds. The simple policy can include semantically unhelpful phrases such as “part”; sample text produced an extra part-whole edge.

Nodes use document+source-span identity, avoiding global name-only merging. CRF types are retained for exactly matching named phrases; ordinary phrases use CONCEPT. Sentence chunk IDs match the actual persisted Stanford sentence chunks. Edges retain the native directed label, whole-sentence evidence and stable claim identity. All nine relation families are whitelisted for storage. Confidence is null because this model supplies uncalibrated predictions; API/storage boundary schemas accept it. No arbitrary probability is assigned. Assertions remain unknown. This may require frontend consumers to handle null confidence explicitly.

Live text smoke test succeeded for water/bottle and wheel/car. Code tests validate source offsets, schema compatibility and a mocked graph-write query. A live Neo4j round trip and separately labeled PDF extraction benchmark have not been performed. The existing inherited bridge still has blocking reads without enforced request deadlines; startup/request robustness is an outstanding limitation.

The 78.64% official SemEval score measures supplied-pair classification. It does not measure this automatic candidate generator, PDF extraction, entity detection or end-to-end graph accuracy. Keep this backend experimental until those stages have reviewed evaluation data.
