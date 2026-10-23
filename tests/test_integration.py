"""End-to-end pipeline integration test."""
import pytest
from src.ingestion.preprocessor import preprocess, build_windows
from src.nlp.ner import NERRunner
from src.nlp.coref_resolver import RuleBasedCorefResolver
from src.nlp.relation_extractor import DepPatternExtractor
from src.nlp.predicate_normalizer import PredicateNormalizer
from src.nlp.relation_scorer import RelationScorer
from src.graph.entity_resolver import EntityResolver
from src.graph.graph_builder import build_resolved_entities, build_resolved_edges


def test_end_to_end_nlp_pipeline(nlp, sample_text):
    """Verify full NLP pipeline runs sequentially from text to resolved graph components."""
    raw_bytes = sample_text.encode("utf-8")
    preprocessed = preprocess(raw_bytes, "sample.txt")
    chunks = preprocessed.chunks

    assert len(chunks) >= 3

    # 1. NER
    ner = NERRunner()
    all_entities = []
    entities_by_chunk = {}
    for chunk in chunks:
        doc = nlp(chunk.text)
        ents = ner.run(doc, chunk.chunk_id, chunk.text)
        entities_by_chunk[chunk.chunk_id] = ents
        all_entities.extend(ents)

    assert len(all_entities) > 0

    # 2. Coreference
    coref = RuleBasedCorefResolver()
    coref_map = coref.resolve(chunks, entities_by_chunk)
    entity_by_id = {e.entity_id: e for e in all_entities}

    # 3. Relation Extraction across sliding windows
    extractor = DepPatternExtractor()
    normalizer = PredicateNormalizer()
    scorer = RelationScorer()
    windows = build_windows(chunks, window_size=3, stride=2)

    raw_triples = []
    for window_chunks, center_chunk in windows:
        window_text = " ".join(c.text for c in window_chunks)
        window_doc = nlp(window_text)
        window_entities = []
        for wc in window_chunks:
            window_entities.extend(entities_by_chunk.get(wc.chunk_id, []))

        triples = extractor.extract(
            doc=window_doc,
            entities=window_entities,
            center_chunk_id=center_chunk.chunk_id,
            center_sent_idx=center_chunk.sent_idx,
            coref_map=coref_map,
            entity_by_id=entity_by_id,
            window_chunks=window_chunks,
        )
        raw_triples.extend(triples)

    assert len(raw_triples) > 0

    # 4. Predicate Normalization & Scoring
    normalized = []
    for triple in raw_triples:
        nt = normalizer.normalize(triple)
        if nt is not None:
            nt.confidence = scorer.score(triple)
            normalized.append(nt)

    # 5. Entity Resolution
    resolver = EntityResolver(threshold=0.85)
    id_map = resolver.resolve_within_doc(all_entities)

    # 6. Graph Payload Assembly
    resolved_entities, entity_to_chunks = build_resolved_entities(all_entities, id_map)
    resolved_edges = build_resolved_edges(normalized, id_map)

    assert len(resolved_entities) > 0
    assert len(resolved_edges) > 0

    edge_types = {e.relation_type for e in resolved_edges}
    # Expected relations from sample_text
    assert {"FOUNDED", "ACQUIRED", "LOCATED_IN"} <= edge_types
    facts = {(t.subject.canonical_name, t.relation_type, t.object.canonical_name, t.sent_idx)
             for t in normalized}
    assert ("Elon Musk", "FOUNDED", "Tesla", 0) in facts
    assert ("Elon Musk", "RELATED_TO", "Spacex", 1) in facts
    assert ("Tesla", "LOCATED_IN", "Austin", 2) in facts
    assert ("Google", "ACQUIRED", "Deepmind", 3) in facts
