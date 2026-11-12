"""Experimental automatic noun-pair extraction with the trained lexical SVM."""
import hashlib
import itertools
import json
from pathlib import Path
import joblib
from src.graph.graph_builder import ResolvedEdge, build_resolved_entities
from src.ingestion.preprocessor import PreprocessedDoc, SentenceChunk
from src.nlp.ner import Entity
from src.nlp.stanford_classical_client import get_shared_stanford_client

GRAPH_RELATIONS = {
    'Cause-Effect': 'CAUSE_EFFECT', 'Instrument-Agency': 'INSTRUMENT_AGENCY',
    'Product-Producer': 'PRODUCT_PRODUCER', 'Content-Container': 'CONTENT_CONTAINER',
    'Entity-Origin': 'ENTITY_ORIGIN', 'Entity-Destination': 'ENTITY_DESTINATION',
    'Component-Whole': 'COMPONENT_WHOLE', 'Member-Collection': 'MEMBER_COLLECTION',
    'Message-Topic': 'MESSAGE_TOPIC',
}


class DocumentScopedResolver:
    async def resolve_against_graph(self, entity, session):
        return None


def extract_pipeline(raw_bytes, filename, text, model_dir, client=None):
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir/'manifest.json').read_text(encoding='utf-8'))
    model_path = model_dir/'model.joblib'
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != manifest['model_sha256']:
        raise ValueError('SVM checkpoint hash mismatch')
    model = joblib.load(model_path)  # trusted local training artifact
    annotation = (client or get_shared_stanford_client()).annotate(text)
    if 'noun_phrases' not in annotation:
        raise RuntimeError('Stanford bridge needs recompilation: noun_phrases missing')
    doc_id = hashlib.sha256(raw_bytes).hexdigest()
    chunks, entities, edges, diagnostics = [], [], [], []
    for sentence in annotation['sentences']:
        si, start, end = sentence['index'], sentence['char_start'], sentence['char_end']
        cid = f'{doc_id}:{si}'
        chunks.append(SentenceChunk(cid, doc_id, si, text[start:end], start, end))
        phrases = sorted({(p['char_start'], p['char_end']) for p in annotation['noun_phrases']
                          if p['sentence_index'] == si})
        if any(not start <= a < b <= end for a,b in phrases):
            raise ValueError('Noun phrase outside source sentence')
        # Bound pair growth explicitly; report skipped pairs rather than hide truncation.
        pairs = list(itertools.combinations(phrases, 2))
        considered = [(a,b) for a,b in pairs if a[1] <= b[0] and b[0]-a[1] <= 200][:100]
        diagnostics.append(dict(sentence=si, candidate_pairs=len(pairs), evaluated_pairs=len(considered),
                                skipped_pairs=len(pairs)-len(considered)))
        by_span = {}
        for a,b in phrases:
            eid = f'{doc_id}:{a}:{b}'
            matching = [e for e in annotation.get('entities',[]) if e['char_start']==a and e['char_end']==b]
            etype = {'PERSON':'PERSON','PEOPLE':'PERSON','ORGANIZATION':'ORG','LOCATION':'GPE'}.get(
                matching[0]['type'] if matching else '', 'CONCEPT')
            ent = Entity(eid,text[a:b],text[a:b],etype,a,b,'stanford_pcfg',cid,sentence=text[start:end])
            entities.append(ent)
            by_span[(a,b)] = ent
        if not considered:
            continue
        records = [dict(text=text[start:end],e1=[a[0]-start,a[1]-start],e2=[b[0]-start,b[1]-start])
                   for a,b in considered]
        for (a,b), label in zip(considered, model.predict(records)):
            if label == 'Other':
                continue
            family, direction = label.split('(')
            subject, obj = (a,b) if direction == 'e1,e2)' else (b,a)
            claim_id = hashlib.sha256(f'{doc_id}:{subject}:{label}:{obj}:{digest}'.encode()).hexdigest()
            edges.append(ResolvedEdge(GRAPH_RELATIONS[family],by_span[subject].entity_id,
                by_span[obj].entity_id,[cid],['semeval_svm_experimental'],None,
                raw_predicate=label,evidence=text[start:end],claim_id=claim_id))
    id_map = {e.entity_id:e.entity_id for e in entities}
    resolved, entity_to_chunks = build_resolved_entities(entities,id_map)
    mentions = {e.entity_id:[dict(id=e.entity_id,text=e.text,start=e.start_char,end=e.end_char,
                                chunk_id=e.chunk_id,source=e.source)] for e in entities}
    return dict(preprocessed=PreprocessedDoc(doc_id,filename,'en',text,chunks),chunks=chunks,
        filename=filename,all_entities=entities,id_map=id_map,resolved_entities=resolved,
        normalized=[],mentions_by_id=mentions,resolver=DocumentScopedResolver(),resolved_edges=edges,
        relation_backend='semeval_svm',candidate_diagnostics=diagnostics,
        evaluation_scope='experimental automatic noun pairs; 78.64% applies only to supplied pairs')
