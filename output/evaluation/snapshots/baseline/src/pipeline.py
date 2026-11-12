"""Pipeline orchestrator — runs all stages sequentially with file lock."""
from __future__ import annotations
import asyncio

import logging
import json
from pathlib import Path

import spacy

from src import pipeline_lock
from src.api.schemas import ResolvedEntityPayload, ResolvedEdgePayload
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
from src.graph.neo4j_client import create_driver

logger = logging.getLogger(__name__)


def _remove_document_title_entities(
    chunks,
    entities_by_chunk: dict[str, list],
) -> dict[str, list]:
    """Keep report titles as document metadata, not semantic graph entities."""
    filtered: dict[str, list] = {}
    first_chunk_id = chunks[0].chunk_id if chunks else None
    for chunk in chunks:
        entities = entities_by_chunk.get(chunk.chunk_id, [])
        if chunk.chunk_id != first_chunk_id or "\n" not in chunk.text:
            filtered[chunk.chunk_id] = entities
            continue
        title_end = chunk.text.index("\n")
        filtered[chunk.chunk_id] = [
            entity for entity in entities if entity.start_char >= title_end
        ]
    return filtered


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


def run_pipeline_nlp(
    raw_bytes: bytes,
    filename: str,
    ftype: str,
    job_id: str | None = None,
) -> dict:
    """
    Execute the CPU-bound NLP phases of the pipeline synchronously.

    Acquires the pipeline file lock before running. Designed to be called
    from a background thread (via asyncio.run_in_executor) to avoid blocking
    the event loop.

    Returns a dict of NLP results consumed by run_pipeline_write().
    The async Neo4j write phase MUST run on the FastAPI event loop to avoid
    'Future attached to a different loop' errors from the async neo4j driver.
    """
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
        entities_by_chunk: dict[str, list] = {}

        for chunk, doc in zip(chunks, nlp.pipe(c.text for c in chunks)):
            chunk_entities = ner.run(doc, chunk.chunk_id, chunk.text)
            entities_by_chunk[chunk.chunk_id] = chunk_entities

        entities_by_chunk = _remove_document_title_entities(chunks, entities_by_chunk)
        all_entities = [
            entity for chunk_entities in entities_by_chunk.values()
            for entity in chunk_entities
        ]
        logger.info("[%s] %d raw entities extracted", job_id, len(all_entities))

        # ── Phase 2b: Coreference ─────────────────────────────────────────
        coref = RuleBasedCorefResolver()
        coref_map = coref.resolve(chunks, entities_by_chunk)
        entity_by_id = {e.entity_id: e for e in all_entities}
        if getattr(settings, "debug_pipeline", False):
            logger.debug("[%s] coreference links: %d", job_id, len(coref_map))

        # ── Phase 2c: Relation Extraction ─────────────────────────────────
        extractor = DepPatternExtractor()
        relations_path = getattr(settings, "relations_config_path", "config/relations.yaml")
        normalizer = PredicateNormalizer(relations_path, include_extended_relations=True)
        scorer = RelationScorer(relations_path)

        windows = build_windows(
            chunks,
            window_size=settings.window_size,
            stride=settings.window_stride,
        )

        raw_triples = []
        window_docs = nlp.pipe(" ".join(c.text for c in window) for window, _ in windows)
        for (window_chunks, center_chunk), window_doc in zip(windows, window_docs):

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

        # Relation extraction may add a source-supported concept entity when
        # a supported verb's object is a noun phrase rather than a named entity.
        entity_by_id = {e.entity_id: e for e in all_entities}
        for triple in raw_triples:
            for entity in (triple.subject, triple.object):
                if entity.entity_id not in entity_by_id:
                    all_entities.append(entity)
                    entity_by_id[entity.entity_id] = entity

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
        chunk_text = {chunk.chunk_id: chunk.text for chunk in chunks}
        for triple in deduped_triples:
            nt = normalizer.normalize(triple)
            if nt is not None:
                nt.confidence = scorer.score(triple)
                nt.evidence = chunk_text.get(nt.chunk_id, "")
                normalized.append(nt)

        # Retain canonical relations, plus high-confidence RELATED_TO edges
        # (>= min_related_confidence, default 0.60) so novel phrasing connects nodes.
        min_related_conf = getattr(settings, "min_related_confidence", 0.60)
        normalized = [
            t for t in normalized
            if t.relation_type != "RELATED_TO" or t.confidence >= min_related_conf
        ]
        if getattr(settings, "debug_pipeline", False):
            logger.debug(
                "[%s] raw triples=%d, supported triples=%d",
                job_id, len(deduped_triples), len(normalized),
            )
        logger.info("[%s] %d supported triples extracted", job_id, len(normalized))

        # ── Phase 3: Entity Resolution ────────────────────────────────────
        resolver = EntityResolver(threshold=settings.resolution_threshold)
        id_map = resolver.resolve_within_doc(all_entities)

        # Build resolved entities and mentions map for graph write
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

        return {
            "preprocessed": preprocessed,
            "chunks": chunks,
            "filename": filename,
            "all_entities": all_entities,
            "id_map": id_map,
            "resolved_entities": resolved_entities,
            "normalized": normalized,
            "mentions_by_id": mentions_by_id,
            "resolver": resolver,
        }

    finally:
        pipeline_lock.release()


async def run_pipeline_write(nlp_result: dict, job_id: str | None = None) -> dict:
    """
    Execute the async Neo4j write phase using results from run_pipeline_nlp.

    Must be awaited from the FastAPI event loop so the async neo4j driver is
    bound to the correct loop (avoids 'Future attached to a different loop').
    """
    settings = get_settings()

    preprocessed = nlp_result["preprocessed"]
    chunks = nlp_result["chunks"]
    filename = nlp_result["filename"]
    all_entities = nlp_result["all_entities"]
    id_map = nlp_result["id_map"]
    resolved_entities = nlp_result["resolved_entities"]
    normalized = nlp_result["normalized"]
    mentions_by_id = nlp_result["mentions_by_id"]
    resolver = nlp_result["resolver"]

    driver = create_driver(
        settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password
    )
    try:
        builder = GraphBuilder()
        async with driver.session() as session:
            # Cross-doc resolution: check existing nodes before creating new ones
            cross_doc_map: dict[str, str] = {}
            for ent in resolved_entities:
                existing_id = await resolver.resolve_against_graph(ent, session)
                if existing_id:
                    cross_doc_map[ent.canonical_id] = existing_id

            # Update id_map with cross-doc merges
            full_id_map = {
                mention_id: cross_doc_map.get(canonical_id, canonical_id)
                for mention_id, canonical_id in id_map.items()
            }

            # Rebuild all payloads with the same IDs, including aliases
            # and mentions for entities that already exist in the graph.
            final_entities, entity_to_chunks = build_resolved_entities(all_entities, full_id_map)
            final_edges = build_resolved_edges(normalized, full_id_map)
            valid_ids = {entity.canonical_id for entity in final_entities}
            final_edges = [
                edge for edge in final_edges
                if edge.source_id in valid_ids and edge.target_id in valid_ids
            ]
            # Boundary validation: enforce strict Pydantic schemas before storage
            validated_entities = []
            for ent in final_entities:
                ResolvedEntityPayload(**ent.to_dict())
                validated_entities.append(ent)

            validated_edges = []
            for edge in final_edges:
                ResolvedEdgePayload(**edge.to_dict())
                validated_edges.append(edge)

            final_mentions: dict[str, list[dict]] = {}
            for old_id, mentions in mentions_by_id.items():
                final_mentions.setdefault(full_id_map.get(old_id, old_id), []).extend(mentions)
            _write_intermediate_graph(
                Path(getattr(settings, "intermediate_graph_dir", "data/processed"))
                / f"{preprocessed.doc_id}.json",
                validated_entities,
                entity_to_chunks,
                validated_edges,
                final_mentions,
            )

            await builder.write_document_graph_atomic(
                session, preprocessed.doc_id, filename, chunks,
                validated_entities, entity_to_chunks, validated_edges,
            )

            n_new = len(resolved_entities) - len(cross_doc_map)
            n_edges = len(final_edges)

        logger.info("[%s] Written %d entities, %d edges", job_id, n_new, n_edges)
        return {
            "doc_id": preprocessed.doc_id,
            "sentences": len(chunks),
            "entities": len(resolved_entities),
            "new_entities": n_new,
            "edges": n_edges,
        }
    finally:
        await driver.close()


def run_pipeline(
    raw_bytes: bytes,
    filename: str,
    ftype: str,
    job_id: str | None = None,
) -> dict:
    """
    Synchronous full pipeline entry point (runs NLP and then writes to Neo4j).

    For background execution in FastAPI, prefer calling run_pipeline_nlp in a thread
    and awaiting run_pipeline_write on the main event loop to avoid loop conflicts.
    """
    nlp_result = run_pipeline_nlp(raw_bytes, filename, ftype, job_id=job_id)
    return asyncio.run(run_pipeline_write(nlp_result, job_id=job_id))
