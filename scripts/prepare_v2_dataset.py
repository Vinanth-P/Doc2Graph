"""Prepare conservative Re-DocRED v2 supervision without fabricating negatives.

This builds a development artifact, not reviewed finance data or a fresh test set.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from scripts.evaluate_external import reconstruct, gold_facts
from src.nlp.dataset_v2 import pair_supervision, development_split


def record_keys(record):
    return record['title'].strip().casefold(), ' '.join(reconstruct(record)[0].casefold().split())


def convert_record(record, origin, index, ontology):
    text, sentences, starts, mentions = reconstruct(record)
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    document_id = f'redocred:{origin}:{index}:{content_hash[:12]}'
    groups = {str(i): [] for i in range(len(record['vertexSet']))}
    typed = {str(i): set() for i in range(len(record['vertexSet']))}
    output_mentions = []
    for i, mention in enumerate(mentions):
        mid = f'{document_id}:m{i}'
        groups[str(mention['cluster'])].append(mid)
        if mention['type']:
            typed[str(mention['cluster'])].add(mention['type'])
        output_mentions.append({'id': mid, 'document_id': document_id, 'text': mention['text'],
                                'start': mention['start'], 'end': mention['end'],
                                'type': mention['type'] or 'OTHER', 'sentence_idx': mention['sent_id'],
                                'detector': 'dataset_annotation'})
    relations = list(ontology['relations'])
    positives = gold_facts(record)
    candidates = []
    for source in groups:
        for target in groups:
            if source == target:
                continue
            supported = any([a, b] in config['entity_type_pairs'] for config in ontology['relations'].values()
                            for a in typed[source] for b in typed[target])
            if not supported and not any(s == source and t == target for s, _, t in positives):
                continue
            candidates.append({'subject_group': source, 'object_group': target,
                               'type_supported': supported,
                               **pair_supervision({source}, {target}, relations, positives),
                               'assertion_targets': {'polarity': None, 'modality': None, 'lifecycle': None},
                               'assertion_loss_mask': {'polarity': False, 'modality': False, 'lifecycle': False}})
    return {'schema_version': '2.dataset.1', 'document_id': document_id, 'source': origin,
            'source_index': index, 'title': record['title'], 'content_hash': content_hash,
            'text': text, 'sentences': [{'start': start, 'text': sentence} for start, sentence in zip(starts, sentences)],
            'mentions': output_mentions, 'annotated_groups': groups,
            'gold_relations': sorted(positives), 'candidates': candidates,
            'supervision_scope': 'document_entity_group; not a claim that every mention pair expresses the relation',
            'annotation_provenance': 'upstream Re-DocRED; no new human finance review',
            'evidence_annotation_status': 'not converted; do not train evidence selection from full document text'}


def prepare(args):
    ontology = yaml.safe_load(args.ontology.read_text(encoding='utf-8'))
    sources = {name: args.data_dir / f'{name}_revised.json' for name in ('train', 'dev', 'test')}
    records = {name: json.loads(path.read_text(encoding='utf-8')) for name, path in sources.items()}
    blocked_titles, blocked_texts = set(), set()
    for name in ('dev', 'test'):
        for record in records[name]:
            title, text = record_keys(record)
            blocked_titles.add(title); blocked_texts.add(text)
    if args.output.exists():
        raise FileExistsError('Use a new output directory; existing datasets are immutable')
    args.output.mkdir(parents=True)
    counts, assignment, exclusions = Counter(), [], []
    handles = {name: (args.output / f'{name}.jsonl').open('w', encoding='utf-8')
               for name in ('train', 'validation', 'calibration', 'regression_dev', 'regression_test')}
    try:
        for origin, rows in records.items():
            for index, record in enumerate(rows):
                title, text = record_keys(record)
                if origin == 'train':
                    if title in blocked_titles or text in blocked_texts:
                        exclusions.append({'source_index': index, 'title': record['title'],
                                           'reason': 'title/text overlap with regression data or earlier training record'})
                        continue
                    blocked_titles.add(title); blocked_texts.add(text)
                    split = development_split(title)
                else:
                    split = f'regression_{origin}'
                converted = convert_record(record, origin, index, ontology)
                handles[split].write(json.dumps(converted, ensure_ascii=False) + '\n')
                counts[split] += 1
                for candidate in converted['candidates']:
                    counts['candidates'] += 1
                    counts['positive_labels'] += candidate['targets'].count(1)
                    counts['explicit_negative_labels'] += candidate['targets'].count(0)
                    counts['unknown_labels'] += candidate['targets'].count(None)
                assignment.append({'document_id': converted['document_id'], 'split': split,
                                   'group_key': title, 'content_hash': converted['content_hash']})
    finally:
        for handle in handles.values():
            handle.close()
    manifest = {'schema_version': '2.dataset.1', 'seed': 'doc2graph-v2-20261111',
                'label_order': list(ontology['relations']), 'counts': dict(counts),
                'source_hashes': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
                'ontology_sha256': hashlib.sha256(args.ontology.read_bytes()).hexdigest(),
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'masking_code_sha256': hashlib.sha256(Path('src/nlp/dataset_v2.py').read_bytes()).hexdigest(),
                'output_hashes': {name: hashlib.sha256((args.output / f'{name}.jsonl').read_bytes()).hexdigest() for name in handles},
                'exclusions': exclusions, 'assignments': assignment,
                'release_holdout_available': False,
                'limitations': ['Old dev/test are regression-only; training pool previously used in experiments.',
                               'Exact title/text dedup only; semantic company/deal/template overlap remains unaudited.',
                               'All absent labels are unknown. No explicit negative or finance assertion supervision.',
                               'Document group facts must not be broadcast as local mention-pair evidence.',
                               'Not sufficient for production training or a finance quality claim.']}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'counts': dict(counts), 'excluded': len(exclusions), 'release_holdout_available': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/external/redocred'))
    parser.add_argument('--ontology', type=Path, default=Path('config/ontology_v2.yaml'))
    parser.add_argument('--output', type=Path, default=Path('data/prepared/v2_redocred'))
    prepare(parser.parse_args())
