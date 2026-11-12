"""Offline smoke test of the cached, pinned discriminative relation encoder."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.runtime/glirel'))
os.environ['HF_HOME'] = str(ROOT / '.runtime/huggingface')
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['USE_TF'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
import json
import torch
from glirel import GLiREL

torch.set_num_threads(2)
repo = 'jackboyla/glirel-large-v0'
revision = '40a523e12a8432d6da364cf2a195a28755ff04d3'
print(json.dumps({'repo': repo, 'revision': revision}), flush=True)
model = GLiREL.from_pretrained(repo, revision=revision, map_location='cpu', strict=True, local_files_only=True)
model.eval()
tokens = ['Alice', 'founded', 'Acme', '.', 'Bob', 'works', 'for', 'Globex', '.']
ner = [[0, 0, 'PERSON', 'Alice'], [2, 2, 'ORG', 'Acme'],
       [4, 4, 'PERSON', 'Bob'], [7, 7, 'ORG', 'Globex']]
with torch.inference_mode():
    predictions = model.predict_relations(tokens, ['founder', 'employer', 'member of', 'owned by', 'parent organization'],
                                          threshold=.5, ner=ner, top_k=-1)
report = {'repo': repo, 'revision': revision, 'predictions': predictions}
(ROOT / 'output/evaluation/encoder_probe.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report), flush=True)
