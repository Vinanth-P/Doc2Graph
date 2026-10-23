"""Tests for Named Entity Recognition (NER) pipeline."""
import pytest
from src.nlp.ner import NERRunner, regex_entities, GazetteerMatcher


def test_regex_entities_date_money():
    text = "On Jan 15, 2024, the company paid $50 million via contact@tesla.com."
    ents = regex_entities(text, chunk_id="chunk_1")
    types = {e.type for e in ents}
    assert "DATE" in types
    assert "MONEY" in types
    assert "EMAIL" in types


def test_gazetteer_matcher(tmp_path):
    # Setup test gazetteer file
    gaz_dir = tmp_path / "gazetteers"
    gaz_dir.mkdir()
    (gaz_dir / "organizations.txt").write_text("OpenAI\nDeepMind\nAnthropic")

    matcher = GazetteerMatcher(gazetteer_dir=gaz_dir)
    ents = matcher.match("DeepMind was acquired by Google.", chunk_id="chunk_2")
    assert len(ents) == 1
    assert ents[0].canonical_name.lower() == "deepmind"
    assert ents[0].type == "ORG"


def test_ner_runner_spacy(nlp):
    runner = NERRunner()
    text = "Elon Musk works at Tesla in California."
    doc = nlp(text)
    entities = runner.run(doc, chunk_id="chunk_3", raw_text=text)

    ent_map = {e.text: e.type for e in entities}
    assert "Elon Musk" in ent_map
    assert ent_map["Elon Musk"] == "PERSON"
    assert "Tesla" in ent_map
    assert ent_map["Tesla"] == "ORG"
    assert "California" in ent_map
    assert ent_map["California"] == "GPE"


def test_gazetteer_finds_multiple_repeated_and_lowercase_names(tmp_path):
    (tmp_path / "organizations.txt").write_text("Google\nDeepMind\nMicrosoft")
    text = "google acquired DeepMind. Google contacted Microsoftt."
    entities = GazetteerMatcher(tmp_path).match(text, "d:0")
    assert [(e.text, e.canonical_name) for e in entities] == [
        ("google", "Google"), ("DeepMind", "Deepmind"),
        ("Google", "Google"), ("Microsoftt", "Microsoft"),
    ]
    assert len({e.entity_id for e in entities}) == 4
    assert all(text[e.start_char:e.end_char] == e.text for e in entities)

