from src.nlp.encoder_relations import EncoderRelationExtractor, token_windows
from src.nlp.ner import Entity
from src.ingestion.preprocessor import SentenceChunk


def fixture():
    text = 'Alice founded Acme.'
    chunk = SentenceChunk('c', 'd', 0, text, 0, len(text))
    entities = [Entity('a', 'Alice', 'Alice', 'PERSON', 0, 5, 'ner', 'c', text),
                Entity('b', 'Acme', 'Acme', 'ORG', 14, 18, 'ner', 'c', text)]
    return [chunk], entities, {'a': 'alice', 'b': 'acme'}


def test_token_offsets_are_exact_and_output_spans_are_exclusive():
    chunks, entities, _ = fixture()
    tokens, mapping, provenance = next(token_windows(chunks, entities))
    assert tokens == ['Alice', 'founded', 'Acme', '.']
    assert mapping[0, 1].text == 'Alice' and mapping[2, 3].text == 'Acme'
    assert provenance == ('c',)


def test_founder_direction_and_inclusive_ner_input():
    class Model:
        def predict_relations(self, tokens, labels, **kwargs):
            assert kwargs['ner'] == [[0, 0, 'PERSON', 'Alice'], [2, 2, 'ORG', 'Acme']]
            return [{'head_pos': [0, 1], 'tail_pos': [2, 3], 'label': 'founder', 'score': .9}]
    triples = EncoderRelationExtractor(Model()).extract(*fixture())
    assert len(triples) == 1
    assert triples[0].relation_type == 'FOUNDED'
    assert triples[0].subject.text == 'Alice' and triples[0].object.text == 'Acme'


def test_encoder_does_not_connect_resolved_self_pairs():
    class Model:
        def predict_relations(self, *args, **kwargs):
            return [{'head_pos': [0, 1], 'tail_pos': [2, 3], 'label': 'founder', 'score': .9}]
    chunks, entities, _ = fixture()
    assert EncoderRelationExtractor(Model()).extract(chunks, entities, {'a': 'same', 'b': 'same'}) == []
