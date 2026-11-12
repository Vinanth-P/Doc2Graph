import pytest
from src.nlp.semeval_dataset import parse_marked
from src.nlp.utd_svm import features, score


def test_repeated_phrase_offsets_and_unicode():
    r = parse_marked('🧪 bottle: <e1>bottle</e1> contains <e2>water</e2>.')
    assert r['text'][slice(*r['e1'])] == 'bottle'
    assert r['e1'][0] == 10
    assert r['text'][slice(*r['e2'])] == 'water'
    assert features(r)['e1=bottle'] == 1


@pytest.mark.parametrize('text',['<e1>x</e1>', '<e1><e2>x</e2></e1>', '<e1></e1><e2>x</e2>'])
def test_bad_markers(text):
    with pytest.raises(ValueError):
        parse_marked(text)


def test_direction_error_counts_as_fp_and_fn():
    r = score(['Cause-Effect(e1,e2)'],['Cause-Effect(e2,e1)'])
    assert r['per_family']['Cause-Effect']['fp'] == 1
    assert r['per_family']['Cause-Effect']['fn'] == 1
    assert r['macro_f1'] == 0
