import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import yaml
from src.nlp.dataset_v2 import pair_supervision, align_mentions, development_split
from src.nlp.contracts_v2 import MentionRecord
from scripts.prepare_v2_dataset import convert_record, prepare
from scripts.generate_audit_report import analyze


def test_unknown_labels_and_unmatched_pairs_are_masked():
    positives = {('a', 'OWNS', 'b')}
    assert pair_supervision({'a'}, {'b'}, ['OWNS', 'ACQUIRED'], positives) == {
        'targets': [1, None], 'loss_mask': [True, False]}
    for source in (set(), {'a', 'c'}):
        assert pair_supervision(source, {'b'}, ['OWNS'], positives)['loss_mask'] == [False]
    assert pair_supervision({'b'}, {'a'}, ['OWNS'], positives)['targets'] == [None]


def test_only_reviewed_negatives_receive_zero_and_conflicts_fail():
    negative = {('a', 'OWNS', 'b')}
    assert pair_supervision({'a'}, {'b'}, ['OWNS'], (), negative)['targets'] == [0]
    with pytest.raises(ValueError, match='Contradictory'):
        pair_supervision({'a'}, {'b'}, ['OWNS'], negative, negative)


def test_alignment_preserves_unmatched_and_ambiguous_mentions():
    def m(mid, start=0):
        return MentionRecord(id=mid, document_id='d', text='A', type='ORG', start=start, end=start+1)
    predicted = [m('p'), m('q', 2)]
    aligned = align_mentions(predicted, [m('a'), m('b')])
    assert aligned == {'p': frozenset({'a', 'b'}), 'q': frozenset()}
    assert [m.id for m in predicted] == ['p', 'q']


def record(title='Example'):
    return {'title': title, 'sents': [['Alice', 'founded', 'Acme', '.']],
            'vertexSet': [[{'name': 'Alice', 'sent_id': 0, 'pos': [0, 1], 'type': 'PER'}],
                          [{'name': 'Acme', 'sent_id': 0, 'pos': [2, 3], 'type': 'ORG'}]],
            'labels': [{'h': 1, 't': 0, 'r': 'P112', 'evidence': [0]}]}


def test_conversion_direction_unknown_finance_and_no_mention_merging():
    ontology = yaml.safe_load(Path('config/ontology_v2.yaml').read_text())
    converted = convert_record(record(), 'train', 0, ontology)
    assert converted['gold_relations'] == [('0', 'FOUNDED', '1')]
    labels = list(ontology['relations'])
    pair = next(c for c in converted['candidates'] if c['subject_group'] == '0' and c['object_group'] == '1')
    assert pair['targets'][labels.index('FOUNDED')] == 1
    assert pair['targets'][labels.index('ACQUIRED')] is None
    assert not any(pair['assertion_loss_mask'].values())
    assert len(converted['mentions']) == 2
    for mention in converted['mentions']:
        assert converted['text'][mention['start']:mention['end']] == mention['text']


def test_preparation_excludes_regression_overlap_and_refuses_overwrite(tmp_path):
    for source, records in [('train', [record()]), ('dev', [record()]), ('test', [])]:
        (tmp_path / f'{source}_revised.json').write_text(json.dumps(records))
    args = SimpleNamespace(data_dir=tmp_path, ontology=Path('config/ontology_v2.yaml'), output=tmp_path / 'prepared')
    prepare(args)
    manifest = json.loads((args.output / 'manifest.json').read_text())
    assert len(manifest['exclusions']) == 1
    assert manifest['counts']['explicit_negative_labels'] == 0
    assert manifest['release_holdout_available'] is False
    assert all(a['split'] == 'regression_dev' for a in manifest['assignments'])
    with pytest.raises(FileExistsError):
        prepare(args)
    assert development_split('same') == development_split('same')


def test_audit_uses_set_intersection_not_difference_of_tp_totals():
    a, b = ['a', 'OWNS', 'b'], ['c', 'OWNS', 'd']
    data = {'documents_detail': [{'gold': [a, b], 'end_to_end': {'predicted': [a]},
                                 'gold_entities': {'predicted': [b]}, 'index': 0, 'title': 'x'}]}
    counts, _ = analyze(data)
    assert counts['fn_recovered_by_oracle'] == 1
    assert counts['e2e_tp_missing_in_oracle'] == 1
