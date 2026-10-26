"""Pipeline orchestrator — runs all stages sequentially with file lock."""
from __future__ import annotations

import logging
import json
from pathlib import Path

import spacy

from src import pipeline_lock
from src.config import get_settings
from src.ingestion.loader import load_document
from src.ingestion.pdf_cleaner import clean_pdf_text
from src.ingestion.preprocessor import preprocess, build_windows, get_nlp
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
from src.graph.neo4j_client import get_driver, create_driver

logger = logging.getLogger(__name__)


def _write_intermediate_graph(
    path: Path,
    entities,
    entity_to_chunks: dict[str, list[str]],
    edges,
    mentions_by_id: dict[str, list[dict]],
) -> None:
    """Persist the graph payload before Neo4j so NLP and storage can be separated."""
    entity_by_id = {entity.canonical_id: entity for entity in entities}
    relationships = []
    for edge in edges:
        relationship = edge.to_dict()
        source = entity_by_id.get(edge.source_id)
        target = entity_by_id.get(edge.target_id)
        relationship["source_type"] = source.type if source else None
        relationship["target_type"] = target.type if target else None
        relationships.append(relationship)
    payload = {
        "entities": [
            {
                **entity.to_dict(),
                "mentions": mentions_by_id.get(entity.canonical_id, []),
            }
            for entity in entities
        ],
        "relationships": relationships,
        "entity_to_chunks": entity_to_chunks,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def _get_nlp() -> spacy.Language:
    # Share the tokenizer and parser used for preprocessing.
    return get_nlp()


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
        preprocessed = preprocess(raw_bytes, filename, text=text)
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
        entity_by_id = {e.entity_id: e for e in all_entities}
        if getattr(settings, "debug_pipeline", False):
            logger.debug("[%s] coreference links: %d", job_id, len(coref_map))

        # ── Phase 2c: Relation Extraction ─────────────────────────────────
        extractor = DepPatternExtractor()
        normalizer = PredicateNormalizer(include_extended_relations=True)
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
                coref_map=coref_map,
                entity_by_id=entity_by_id,
                window_chunks=window_chunks,
            )
            raw_triples.extend(triples)

        # Preserve preposition context and polarity until normalization.
        seen_keys: set = set()
        deduped_triples = []
        for t in raw_triples:
            key = (t.chunk_id, t.subject.entity_id, t.predicate, t.prep,
                   t.object.entity_id, t.has_negation)
            if key not in seen_keys:
                seen_keys.add(key)
                deduped_triples.append(t)

        # Score and normalize
        normalized = []
        for triple in deduped_triples:
            nt = normalizer.normalize(triple)
            if nt is not None:
                nt.confidence = scorer.score(triple)
                nt.evidence = next(
                    (chunk.text for chunk in chunks if chunk.chunk_id == nt.chunk_id),
                    "",
                )
                normalized.append(nt)

        # RELATED_TO is retained only for backwards-compatible direct
        # normalizer callers. It is not an asserted relation in the graph.
        normalized = [t for t in normalized if t.relation_type != "RELATED_TO"]
        if getattr(settings, "debug_pipeline", False):
            logger.debug(
                "[%s] raw triples=%d, supported triples=%d",
                job_id, len(deduped_triples), len(normalized),
            )
        logger.info("[%s] %d supported triples extracted", job_id, len(normalized))

        # ── Phase 3: Entity Resolution ────────────────────────────────────
        resolver = EntityResolver(threshold=settings.resolution_threshold)
        id_map = resolver.resolve_within_doc(all_entities)

        # ── Phase 4: Graph Write ──────────────────────────────────────────
        resolved_entities, _ = build_resolved_entities(all_entities, id_map)
        mentions_by_id: dict[str, list[dict]] = {}
        for ent in all_entities:
            canonical_id = id_map.get(ent.entity_id, ent.entity_id)
            mentions_by_id.setdefault(canonical_id, []).append({
                "id": ent.entity_id,
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "sentence": ent.sentence,
                "source": ent.source,
                "chunk_id": ent.chunk_id,
            })

        # Cross-document resolution + graph write run in a new event loop
        async def _write():
            driver = create_driver(
                settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password
            )
            try:
                builder = GraphBuilder()
                async with driver.session() as session:
                    await builder.write_document(session, filename, preprocessed.doc_id)
                    await builder.write_chunks(session, chunks)

                    # Cross-doc resolution: check existing nodes before creating new ones
                    cross_doc_map: dict[str, str] = {}
                    for ent in resolved_entities:
                        existing_id = await resolver.resolve_against_graph(ent, session)
                        if existing_id:
                            cross_doc_map[ent.canonical_id] = existing_id

                    # Update id_map with cross-doc merges
                    full_id_map = {**id_map}
                    for old_id, new_id in cross_doc_map.items():
                        for k, v in full_id_map.items():
                            if v == old_id:
                                full_id_map[k] = new_id

                    # Rebuild all payloads with the same IDs, including aliases
                    # and mentions for entities that already exist in the graph.
                    final_entities, entity_to_chunks = build_resolved_entities(all_entities, full_id_map)
                    final_edges = build_resolved_edges(normalized, full_id_map)
                    valid_ids = {entity.canonical_id for entity in final_entities}
                    final_edges = [
                        edge for edge in final_edges
                        if edge.source_id in valid_ids and edge.target_id in valid_ids
                    ]
                    final_mentions: dict[str, list[dict]] = {}
                    for old_id, mentions in mentions_by_id.items():
                        final_mentions.setdefault(full_id_map.get(old_id, old_id), []).extend(mentions)
                    _write_intermediate_graph(
                        Path(getattr(settings, "intermediate_graph_dir", "data/processed"))
                        / f"{preprocessed.doc_id}.json",
                        final_entities,
                        entity_to_chunks,
                        final_edges,
                        final_mentions,
                    )

                    await builder.write_entities(session, final_entities)
                    await builder.write_mentions(session, entity_to_chunks)
                    await builder.write_edges(session, final_edges)

                return len(resolved_entities) - len(cross_doc_map), len(final_edges)
            finally:
                await driver.close()

        n_entities, n_edges = asyncio.run(_write())
        logger.info("[%s] Written %d entities, %d edges", job_id, n_entities, n_edges)

        return {
            "doc_id": preprocessed.doc_id,
            "sentences": len(chunks),
            "entities": len(resolved_entities),
            "new_entities": n_entities,
            "edges": n_edges,
        }

    finally:
        pipeline_lock.release()
