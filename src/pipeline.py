"""Pipeline orchestrator — runs all stages sequentially with file lock."""
from __future__ import annotations

import logging
from pathlib import Path

import spacy

from src import pipeline_lock
from src.config import get_settings
from src.ingestion.loader import load_document
from src.ingestion.pdf_cleaner import clean_pdf_text
from src.ingestion.preprocessor import preprocess, build_windows
from src.nlp.ner import NERRunner
from src.nlp.coref_resolver import RuleBasedCorefResolver
from src.nlp.relation_extractor import DepPatternExtractor
from src.nlp.predicate_normalizer import PredicateNormalizer
from src.nlp.relation_scorer import RelationScorer
from src.graph.entity_resolver import EntityResolver
from src.graph.graph_builder import (
    GraphBuilder,
    build_resolved_entities,
    build_resolved_edges,
)
from src.graph.neo4j_client import get_driver

logger = logging.getLogger(__name__)

# Shared spaCy model across pipeline (loaded once per process)
_nlp: spacy.Language | None = None


def _get_nlp() -> spacy.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def run_pipeline(
    raw_bytes: bytes,
    filename: str,
    ftype: str,
    job_id: str | None = None,
) -> dict:
    """
    Execute the full ingestion pipeline synchronously.

    Acquires the pipeline file lock before running. Designed to be called
    from a background thread (via asyncio.run_in_executor) to avoid blocking
    the event loop.

    Returns a summary dict with entity/edge counts.
    """
    import asyncio

    settings = get_settings()

    pipeline_lock.acquire()
    try:
        logger.info("[%s] Starting pipeline for %s", job_id, filename)

        # ── Phase 1: Ingestion ────────────────────────────────────────────
        text = load_document(raw_bytes, ftype)
        if ftype == "pdf":
            text = clean_pdf_text(text)
        preprocessed = preprocess(raw_bytes, filename)
        chunks = preprocessed.chunks
        logger.info("[%s] %d sentences extracted", job_id, len(chunks))

        # ── Phase 2a: NER ─────────────────────────────────────────────────
        nlp = _get_nlp()
        ner = NERRunner()
        all_entities = []
        entities_by_chunk: dict[str, list] = {}

        for chunk in chunks:
            doc = nlp(chunk.text)
            chunk_entities = ner.run(doc, chunk.chunk_id, chunk.text)
            entities_by_chunk[chunk.chunk_id] = chunk_entities
            all_entities.extend(chunk_entities)

        logger.info("[%s] %d raw entities extracted", job_id, len(all_entities))

        # ── Phase 2b: Coreference ─────────────────────────────────────────
        coref = RuleBasedCorefResolver()
        coref_map = coref.resolve(chunks, entities_by_chunk)

        # ── Phase 2c: Relation Extraction ─────────────────────────────────
        extractor = DepPatternExtractor()
        normalizer = PredicateNormalizer()
        scorer = RelationScorer()

        windows = build_windows(
            chunks,
            window_size=settings.window_size,
            stride=settings.window_stride,
        )

        raw_triples = []
        for window_chunks, center_chunk in windows:
            window_text = " ".join(c.text for c in window_chunks)
            window_doc = nlp(window_text)

            # Collect entities across the window
            window_entities = []
            for wc in window_chunks:
                window_entities.extend(entities_by_chunk.get(wc.chunk_id, []))

            triples = extractor.extract(
                doc=window_doc,
                entities=window_entities,
                center_chunk_id=center_chunk.chunk_id,
                center_sent_idx=center_chunk.sent_idx,
            )
            raw_triples.extend(triples)

        # Deduplicate by (sent_idx, subj_id, pred, obj_id)
        seen_keys: set = set()
        deduped_triples = []
        for t in raw_triples:
            key = (t.sent_idx, t.subject.entity_id, t.predicate, t.object.entity_id)
            if key not in seen_keys:
                seen_keys.add(key)
                deduped_triples.append(t)

        # Score and normalize
        scores = scorer.score_batch(deduped_triples)
        normalized = normalizer.normalize_batch(deduped_triples)
        for nt, score in zip(normalized, scores):
            nt.confidence = score

        logger.info("[%s] %d triples extracted", job_id, len(normalized))

        # ── Phase 3: Entity Resolution ────────────────────────────────────
        resolver = EntityResolver(threshold=settings.resolution_threshold)
        id_map = resolver.resolve_within_doc(all_entities)

        # ── Phase 4: Graph Write ──────────────────────────────────────────
        resolved_entities, entity_to_chunks = build_resolved_entities(all_entities, id_map)
        resolved_edges = build_resolved_edges(normalized, id_map)

        # Cross-document resolution + graph write run in a new event loop
        async def _write():
            driver = await get_driver(
                settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password
            )
            builder = GraphBuilder()
            async with driver.session() as session:
                await builder.write_document(session, filename, preprocessed.doc_id)
                await builder.write_chunks(session, chunks)

                # Cross-doc resolution: check existing nodes before creating new ones
                final_entities = []
                cross_doc_map: dict[str, str] = {}
                for ent in resolved_entities:
                    existing_id = await resolver.resolve_against_graph(ent, session)
                    if existing_id:
                        cross_doc_map[ent.canonical_id] = existing_id
                    else:
                        final_entities.append(ent)

                # Update id_map with cross-doc merges
                full_id_map = {**id_map}
                for old_id, new_id in cross_doc_map.items():
                    for k, v in full_id_map.items():
                        if v == old_id:
                            full_id_map[k] = new_id

                # Re-build edges with updated ID map
                final_edges = build_resolved_edges(normalized, full_id_map)

                await builder.write_entities(session, final_entities)
                await builder.write_mentions(session, entity_to_chunks)
                await builder.write_edges(session, final_edges)

            return len(final_entities), len(final_edges)

        n_entities, n_edges = asyncio.run(_write())
        logger.info("[%s] Written %d entities, %d edges", job_id, n_entities, n_edges)

        return {
            "doc_id": preprocessed.doc_id,
            "sentences": len(chunks),
            "entities": n_entities,
            "edges": n_edges,
        }

    finally:
        pipeline_lock.release()
