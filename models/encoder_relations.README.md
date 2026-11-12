# Experimental pretrained relation encoder

This backend runs [GLiREL](https://github.com/jackboyla/GLiREL) locally. It is a discriminative DeBERTa relation encoder, not a generative LLM service. It has not passed the project's quality gate and must not be treated as a validated replacement for the default extractor.

- Checkpoint: `jackboyla/glirel-large-v0`, revision `40a523e12a8432d6da364cf2a195a28755ff04d3`.
- Runtime tested: GLiREL 1.2.1, sentencepiece 0.2.1, seqeval 1.2.2, loguru 0.7.3, win32-setctime 1.2.1, with the project's existing Torch/Transformers installation. Sentencepiece 0.2.3 was incompatible with the installed Protobuf runtime.
- Dependencies and checkpoint cache live in `.runtime/`, excluded from Git. These are not committed or automatically downloaded during inference.
- Loading validates all checkpoint parameter keys strictly. Runtime inference is offline. The GPU is used when available, otherwise CPU; missing weights/dependencies cause an explicit error.
- The adapter uses bounded 180-token contexts with 40-token overlap. Input entity spans are inclusive; returned spans are exclusive. Founder is mapped person-to-organization. Schema constraints and canonical self-edge rejection remain enforced.
- Fixed score threshold: 0.75. Scores are not calibrated truth probabilities. Only the same five evaluated relation types are mapped. This does not implement acquisition or advisory extraction.
- Pretraining overlap with Re-DocRED has not been established. Development results are diagnostic, not an independent holdout claim. The model still relies on the existing NER/entity-resolution pipeline.

Existing installation can be inspected without changing Neo4j:

```powershell
python scripts/inspect_document.py Doc2Graph_Corporate_Finance_MA_Dummy.pdf --relation-backend encoder --output output/finance_graph_encoder.json
```

The original `rules` default is unchanged. `RELATION_BACKEND=encoder` explicitly selects this experimental backend; it is not recommended for production. Consult `output/evaluation/encoder_dev.json` and the follow-up report before using its predictions. The [checkpoint model card](https://huggingface.co/jackboyla/glirel-large-v0#license) specifies **CC BY-NC-SA 4.0**, including a noncommercial restriction; this checkpoint is not a commercial deployment recommendation.
