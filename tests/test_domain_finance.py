"""Domain regression tests for Corporate Finance, M&A, and Venture Capital."""
from pathlib import Path
import pytest

from src.ingestion.loader import load_document
from src.ingestion.preprocessor import preprocess, build_windows, get_nlp
from src.nlp.ner import NERRunner
from src.nlp.coref_resolver import RuleBasedCorefResolver
from src.nlp.relation_extractor import DepPatternExtractor
from src.nlp.predicate_normalizer import PredicateNormalizer
from src.nlp.relation_scorer import RelationScorer
from src.graph.entity_resolver import EntityResolver
from src.graph.graph_builder import build_resolved_entities, build_resolved_edges


def _extract_graph_from_text(text: str, filename: str):
    raw_bytes = text.encode("utf-8")
    preprocessed = preprocess(raw_bytes, filename, text=text)
    chunks = preprocessed.chunks
    nlp = get_nlp()
    ner = NERRunner()

    all_entities = []
    entities_by_chunk = {}
    for chunk in chunks:
        doc = nlp(chunk.text)
        ents = ner.run(doc, chunk.chunk_id, chunk.text)
        entities_by_chunk[chunk.chunk_id] = ents
        all_entities.extend(ents)

    coref = RuleBasedCorefResolver()
    coref_map = coref.resolve(chunks, entities_by_chunk)

    extractor = DepPatternExtractor()
    normalizer = PredicateNormalizer(include_extended_relations=True)
    scorer = RelationScorer()
    windows = build_windows(chunks, window_size=3, stride=2)

    norm_triples = []
    for wc, cc in windows:
        wdoc = nlp(" ".join(c.text for c in wc))
        wents = [e for c in wc for e in entities_by_chunk.get(c.chunk_id, [])]
        raw = extractor.extract(
            wdoc, wents, cc.chunk_id, cc.sent_idx,
            coref_map, {e.entity_id: e for e in all_entities}, wc,
        )
        for t in raw:
            nt = normalizer.normalize(t)
            if nt is not None:
                nt.confidence = scorer.score(t)
                if nt.relation_type != "RELATED_TO" or nt.confidence >= 0.60:
                    norm_triples.append(nt)

    resolver = EntityResolver()
    id_map = resolver.resolve_within_doc(all_entities)
    resolved_entities, _ = build_resolved_entities(all_entities, id_map)
    resolved_edges = build_resolved_edges(norm_triples, id_map)

    return resolved_entities, resolved_edges


def test_tech_venture_and_ma_extraction():
    path = Path("data/raw/tech_venture_and_ma_report.txt")
    assert path.exists(), "Test data file must exist"
    text = path.read_text(encoding="utf-8")

    entities, edges = _extract_graph_from_text(text, path.name)

    # Entities and edges must form a connected graph
    assert len(entities) >= 15
    assert len(edges) >= 10

    edge_tuples = {(e.source_id, e.relation_type, e.target_id) for e in edges}
    edge_types = {e.relation_type for e in edges}

    # Core relations must be asserted
    assert "INVESTED_IN" in edge_types
    assert "ACQUIRED" in edge_types
    assert "LEADS" in edge_types
    assert "PARTNER_WITH" in edge_types


def test_private_equity_and_buyouts_extraction():
    path = Path("data/raw/private_equity_and_buyouts.txt")
    assert path.exists(), "Test data file must exist"
    text = path.read_text(encoding="utf-8")

    entities, edges = _extract_graph_from_text(text, path.name)

    assert len(entities) >= 12
    assert len(edges) >= 8

    edge_types = {e.relation_type for e in edges}
    assert "ACQUIRED" in edge_types
    assert "INVESTED_IN" in edge_types
    assert "LEADS" in edge_types


def test_pdf_extraction_matches_text():
    """Verify that pdfplumber extracts clean text matching the raw text from the generated PDF."""
    pdf_path = Path("data/raw/tech_venture_and_ma_report.pdf")
    assert pdf_path.exists()
    content = pdf_path.read_bytes()
    extracted_text = load_document(content, "pdf")

    assert "Microsoft" in extracted_text
    assert "OpenAI" in extracted_text
    assert "Stripe" in extracted_text
    assert "Sequoia Capital" in extracted_text
    # Must not contain replacement or binary control characters
    assert "\ufffd" not in extracted_text

