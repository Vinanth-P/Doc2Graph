"""Optional pretrained discriminative relation encoder; no generative API calls."""
from contextlib import nullcontext
from functools import lru_cache
import os
from pathlib import Path
import sys
from src.nlp.predicate_normalizer import NormalizedTriple, load_relations_config
from src.ingestion.preprocessor import get_nlp

REPOSITORY = 'jackboyla/glirel-large-v0'
REVISION = '40a523e12a8432d6da364cf2a195a28755ff04d3'
# Dataset predicates have head->tail directions, independent of project naming.
LABELS = {'founder': ('FOUNDED', False), 'employer': ('EMPLOYED_BY', False),
          'member of': ('MEMBER_OF', False), 'owned by': ('OWNS', True),
          'parent organization': ('SUBSIDIARY_OF', False)}


def token_windows(chunks, entities, width=180, overlap=40):
    """Bound context size while preserving exact entity offsets and provenance."""
    nlp = get_nlp()
    tokens, token_chunks, spans = [], [], []
    by_chunk = {}
    for entity in entities:
        if entity.source != 'noun_phrase':
            by_chunk.setdefault(entity.chunk_id, []).append(entity)
    for chunk in chunks:
        doc = nlp.make_doc(chunk.text)
        offset = len(tokens)
        tokens.extend(t.text for t in doc)
        token_chunks.extend([chunk.chunk_id] * len(doc))
        for entity in by_chunk.get(chunk.chunk_id, []):
            span = doc.char_span(entity.start_char, entity.end_char, alignment_mode='strict')
            if span is not None:
                spans.append((offset + span.start, offset + span.end, entity))
    for start in range(0, len(tokens), width - overlap):
        end = min(start + width, len(tokens))
        mapping = {(a - start, b - start): ent for a, b, ent in spans if start <= a < b <= end}
        if len(mapping) >= 2:
            yield tokens[start:end], mapping, tuple(dict.fromkeys(token_chunks[start:end]))
        if end == len(tokens):
            break


class EncoderRelationExtractor:
    def __init__(self, model, threshold=.75, device='cpu'):
        self.model, self.threshold, self.device = model, threshold, device
        self.config = load_relations_config()

    def extract(self, chunks, entities, id_map):
        import torch
        by_id = {c.chunk_id: c for c in chunks}
        best = {}
        for tokens, mapping, chunk_ids in token_windows(chunks, entities):
            ner = [[a, b - 1, e.type, e.text] for (a, b), e in sorted(mapping.items())]
            context = torch.autocast('cuda', dtype=torch.float16) if self.device == 'cuda' else nullcontext()
            with torch.inference_mode(), context:
                predictions = self.model.predict_relations(tokens, list(LABELS), ner=ner,
                                                            threshold=self.threshold, top_k=-1)
            for prediction in predictions:
                head = mapping.get(tuple(prediction['head_pos']))
                tail = mapping.get(tuple(prediction['tail_pos']))
                if head is None or tail is None or prediction['label'] not in LABELS:
                    continue
                relation, reverse = LABELS[prediction['label']]
                subject, obj = (tail, head) if reverse else (head, tail)
                source, target = (id_map.get(e.entity_id, e.entity_id) for e in (subject, obj))
                if source == target or [subject.type, obj.type] not in self.config[relation]['entity_type_pairs']:
                    continue
                key = source, relation, target
                score = float(prediction['score'])
                if key not in best or score > best[key][0]:
                    best[key] = score, subject, obj, chunk_ids
        triples = []
        for (_, relation, _), (score, subject, obj, chunk_ids) in best.items():
            evidence = ' '.join(by_id[cid].text for cid in chunk_ids)
            for cid in chunk_ids:
                triples.append(NormalizedTriple(relation, subject, obj, 'glirel_encoder', cid,
                                                by_id[cid].sent_idx, score, evidence, 'glirel_encoder'))
        return triples


@lru_cache(maxsize=1)
def load_encoder_extractor():
    root = Path(__file__).resolve().parents[2]
    runtime = root / '.runtime/glirel'
    if runtime.exists():
        sys.path.insert(0, str(runtime))
    os.environ.setdefault('HF_HOME', str(root / '.runtime/huggingface'))
    os.environ.setdefault('USE_TF', '0')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    try:
        import torch
        from glirel import GLiREL
    except ImportError as exc:
        raise RuntimeError('Encoder dependencies are missing. See models/encoder_relations.README.md.') from exc
    torch.set_num_threads(2)
    model = GLiREL.from_pretrained(REPOSITORY, revision=REVISION, local_files_only=True, map_location='cpu', strict=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval().to(device)
    return EncoderRelationExtractor(model, device=device)
