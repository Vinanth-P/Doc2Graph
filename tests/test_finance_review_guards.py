import copy
import hashlib
import json
from pathlib import Path
import pytest
import yaml
from pydantic import ValidationError
from src.nlp.contracts_v2 import ReviewedDocumentPayload
from src.nlp.deduplication import analyze_duplicates_and_groups
from src.nlp.dataset_v2 import finance_split
from scripts.import_reviewed_finance import convert_reviewed_document, import_finance_data


def fixture():
    doc = json.loads(Path('data/finance_pilot/pilot_reviewed_annotations.json').read_text(encoding='utf-8'))[0]
    # Unit-test attestation only; never exported as a real reviewed corpus.
    doc['data_origin'] = 'human_reviewed_source'
    return doc


def convert(doc):
    ontology = yaml.safe_load(Path('config/ontology_v2.yaml').read_text())
    return convert_reviewed_document(doc, doc['doc_group'], ontology)


@pytest.mark.parametrize('decision', ['positive', 'explicit_negative'])
def test_pending_annotation_never_supervises(decision):
    doc = fixture()
    doc['relation_annotations'] = [doc['relation_annotations'][0]]
    doc['relation_annotations'][0].update(decision=decision, review_status='pending_human_review', reviewer='proposal_engine_unreviewed')
    output = convert(doc)
    assert not output['gold_relations'] and not output['reviewed_negatives']
    assert all(not any(c['loss_mask']) for c in output['candidates'])
    assert all(not c['assertions_by_relation'] for c in output['candidates'])


@pytest.mark.parametrize('field,value', [('review_status', 'pending_human_review'), ('data_origin', 'unverified')])
def test_document_review_gate_masks_claimed_reviewed_annotations(field, value):
    doc = fixture(); doc[field] = value
    assert all(not any(c['loss_mask']) for c in convert(doc)['candidates'])


def test_reviewed_document_requires_timestamp():
    doc = fixture(); doc['review_timestamp'] = None
    with pytest.raises(ValidationError, match='timestamp'):
        ReviewedDocumentPayload.model_validate(doc)


def test_fixture_requires_explicit_testing_mode(tmp_path):
    with pytest.raises(ValueError, match='fixtures require'):
        import_finance_data(Path('data/finance_pilot/pilot_reviewed_annotations.json'),
                            Path('config/ontology_v2.yaml'), tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_each_relation_retains_its_own_status_evidence_and_reviewer():
    doc = fixture()
    acquisition = doc['relation_annotations'][0]
    acquisition.update(modality='planned', lifecycle='announced')
    ownership = copy.deepcopy(acquisition)
    ownership.update(annotation_id='ownership', relation='OWNS', reviewer='second_reviewer',
                     modality='asserted', lifecycle='unknown')
    doc['relation_annotations'] = [acquisition, ownership]
    candidate = next(c for c in convert(doc)['candidates'] if c['subject_group'] == 'm1' and c['object_group'] == 'm2')
    assert candidate['assertions_by_relation']['ACQUIRED']['targets']['lifecycle'] == 'announced'
    assert candidate['assertions_by_relation']['OWNS']['targets']['lifecycle'] is None
    assert candidate['supervision_by_relation']['OWNS']['reviewer'] == 'second_reviewer'
    assert candidate['supervision_by_relation']['ACQUIRED']['reviewer'] == acquisition['reviewer']


def test_duplicate_relation_statuses_cannot_silently_overwrite():
    doc = fixture()
    a = doc['relation_annotations'][0]
    b = copy.deepcopy(a); b.update(annotation_id='other_event', lifecycle='cancelled')
    doc['relation_annotations'] = [a, b]
    with pytest.raises(ValidationError, match='requires adjudication'):
        ReviewedDocumentPayload.model_validate(doc)


def test_excluded_duplicate_preserves_group_bridge():
    records = [{'document_id': 'a', 'text': 'same content', 'doc_group': 'g1'},
               {'document_id': 'b', 'text': 'same content', 'doc_group': 'g2'},
               {'document_id': 'c', 'text': 'entirely unrelated words', 'doc_group': 'g2'}]
    kept, exclusions, groups = analyze_duplicates_and_groups(records, near_threshold=1)
    assert len(kept) == 2 and len(exclusions) == 1
    assert groups['a'] == groups['b'] == groups['c']


def test_duplicate_ids_and_invalid_threshold_fail():
    record = {'document_id': 'a', 'text': 'x', 'doc_group': 'g'}
    with pytest.raises(ValueError, match='Duplicate document IDs'):
        analyze_duplicates_and_groups([record, record])
    with pytest.raises(ValueError, match='threshold'):
        analyze_duplicates_and_groups([record], near_threshold=0)


def test_split_uses_documented_60_15_15_10_buckets():
    seed = 'test'
    for i in range(300):
        group = str(i)
        bucket = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest(), 16) % 100
        expected = 'train' if bucket < 60 else 'validation' if bucket < 75 else 'calibration' if bucket < 90 else 'locked_test'
        assert finance_split(group, seed) == expected
