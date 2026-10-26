"""Graph builder — writes entities, edges, and Chunk provenance nodes to Neo4j.

Key design decisions (from v5/v6 plan):
  - MERGE on :Entity {id}, then SET :Label (avoids compound-MERGE constraint conflict)
  - Edge MERGE + apoc.coll.toSet union for idempotent chunk_ids on retry
  - evidence_count derived from size(r.chunk_ids), never stored as a separate counter
  - Chunk IDs are deterministic: "<doc_hash>:<sent_idx>"
  - mention_count is NOT stored — computed from MENTIONED_IN traversal on query
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from neo4j import AsyncSession

from src.type_map import TYPE_TO_LABEL
from src.nlp.ner import Entity
from src.nlp.predicate_normalizer import NormalizedTriple
from src.ingestion.preprocessor import SentenceChunk
from src.graph.entity_resolver import compute_initials_key, _tokenize


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ResolvedEntity:
    """An entity after intra-doc resolution — has a stable canonical ID."""
    canonical_id: str
    name: str
    type: str
    aliases: list[str]
    initials_key: str

    @property
    def canonical_name(self) -> str:
        return self.name

    def to_dict(self) -> dict:
        return {
            "id": self.canonical_id,
            "name": self.name,
            "type": self.type,
            "aliases": sorted(set(self.aliases)),
            "initials_key": self.initials_key,
        }


@dataclass
class ResolvedEdge:
    """A normalized triple ready for graph write, with evidence."""
    relation_type: str
    source_id: str
    target_id: str
    chunk_ids: list[str]
    extractors: list[str]
    confidence: float
    raw_predicate: str = ""
    evidence: str = ""

    def to_dict(self) -> dict:
        return {
            "relation_type": self.relation_type,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "chunk_ids": self.chunk_ids,
            "extractors": list(set(self.extractors)),
            "confidence": self.confidence,
            "raw_predicate": self.raw_predicate,
            "evidence": self.evidence,
        }


# ── Builder ───────────────────────────────────────────────────────────────────

class GraphBuilder:
    """Writes a processed document's graph data to Neo4j."""

    # Canonical relation types (from relations.yaml) + catch-all
    CANONICAL_TYPES = [
        "MEMBER_OF", "EMPLOYED_BY", "ACQUIRED",
        "LOCATED_IN", "FOUNDED", "INVESTED_IN", "PARTNER_WITH",
        "RELATED_TO", "LEADS", "PROVIDED", "FUNDED", "DEVELOPED",
        "USES", "STORES_DATA_IN",
    ]

    async def clear_document(self, session: AsyncSession, doc_id: str) -> None:
        """Remove evidence owned by a document before replacing its graph."""
        await session.run(
            """
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE any(cid IN coalesce(r.chunk_ids, [])
                WHERE cid STARTS WITH $doc_id + ':')
            DELETE r
            """,
            doc_id=doc_id,
        )
        await session.run(
            """
            MATCH (c:Chunk {doc_id: $doc_id})
            DETACH DELETE c
            """,
            doc_id=doc_id,
        )
        await session.run(
            """
            MATCH (entity:Entity)
            WHERE NOT (entity)-[:MENTIONED_IN]->()
            DETACH DELETE entity
            """
        )
        await session.run(
            "MATCH (d:Document {id: $doc_id}) DETACH DELETE d",
            doc_id=doc_id,
        )

    async def write_document(self, session: AsyncSession, filename: str, doc_id: str) -> None:
        await session.run(
            """
            MERGE (d:Document {id: $id})
            ON CREATE SET d.filename = $filename, d.processed_at = datetime()
            """,
            id=doc_id, filename=filename,
        )

    async def write_chunks(self, session: AsyncSession, chunks: list[SentenceChunk]) -> None:
        chunk_dicts = [
            {
                "id": c.chunk_id,
                "text": c.text,
                "doc_id": c.doc_id,
                "sent_idx": c.sent_idx,
                "start_char": c.start_char,
                "end_char": c.end_char,
            }
            for c in chunks
        ]
        await session.run(
            """
            UNWIND $chunks AS c
            MERGE (ch:Chunk {id: c.id})
            ON CREATE SET ch.text = c.text, ch.doc_id = c.doc_id,
                          ch.sent_idx = c.sent_idx,
                          ch.start_char = c.start_char, ch.end_char = c.end_char
            WITH ch, c
            MATCH (d:Document {id: c.doc_id})
            MERGE (ch)-[:FROM_DOC]->(d)
            """,
            chunks=chunk_dicts,
        )

    async def write_entities(
        self, session: AsyncSession, entities: list[ResolvedEntity]
    ) -> None:
        """MERGE entity on :Entity {id}, then SET secondary type label.
        Groups by type so each query interpolates the correct label from the whitelist.
        """
        by_type: dict[str, list[dict]] = defaultdict(list)
        for e in entities:
            if e.type in TYPE_TO_LABEL:
                by_type[e.type].append(e.to_dict())

        for etype, group in by_type.items():
            label = TYPE_TO_LABEL[etype]  # from whitelist — safe to interpolate
            await session.run(
                f"""
                UNWIND $entities AS e
                MERGE (n:Entity {{id: e.id}})
                ON CREATE SET n.name = e.name, n.type = e.type,
                              n.aliases = e.aliases,
                              n.initials_key = e.initials_key
                ON MATCH SET  n.aliases = apoc.coll.toSet(
                                  coalesce(n.aliases, []) + e.aliases),
                              n.name = e.name,
                              n.type = e.type,
                              n.initials_key = e.initials_key
                SET n:{label}
                """,
                entities=group,
            )

    async def write_mentions(
        self,
        session: AsyncSession,
        entity_to_chunks: dict[str, list[str]],
    ) -> None:
        """Write MENTIONED_IN edges from Entity to Chunk nodes."""
        rows = [
            {"entity_id": eid, "chunk_id": cid}
            for eid, chunk_ids in entity_to_chunks.items()
            for cid in chunk_ids
        ]
        if rows:
            await session.run(
                """
                UNWIND $rows AS r
                MATCH (e:Entity {id: r.entity_id})
                MATCH (c:Chunk   {id: r.chunk_id})
                MERGE (e)-[:MENTIONED_IN]->(c)
                """,
                rows=rows,
            )

    async def write_edges(
        self, session: AsyncSession, edges: list[ResolvedEdge]
    ) -> None:
        """Idempotent edge write — union chunk_ids, max confidence, derive count."""
        for rel_type in self.CANONICAL_TYPES:
            group = [e.to_dict() for e in edges if e.relation_type == rel_type]
            if not group:
                continue
            # rel_type is from the hardcoded whitelist above — safe to interpolate
            await session.run(
                f"""
                UNWIND $edges AS e
                MATCH (a:Entity {{id: e.source_id}})
                MATCH (b:Entity {{id: e.target_id}})
                MERGE (a)-[r:{rel_type}]->(b)
                ON CREATE SET r.chunk_ids    = e.chunk_ids,
                              r.extractors   = e.extractors,
                              r.confidence   = e.confidence,
                              r.raw_predicate = e.raw_predicate,
                              r.evidence     = e.evidence
                ON MATCH SET  r.chunk_ids    = apoc.coll.toSet(
                                  coalesce(r.chunk_ids, []) + e.chunk_ids),
                              r.extractors   = apoc.coll.toSet(
                                  coalesce(r.extractors, []) + e.extractors),
                              r.confidence   = CASE
                                  WHEN e.confidence > r.confidence
                                  THEN e.confidence ELSE r.confidence END
                              ,r.evidence = CASE
                                  WHEN e.confidence >= coalesce(r.confidence, 0)
                                  THEN e.evidence ELSE r.evidence END
                SET r.evidence_count = size(r.chunk_ids)
                """,
                edges=group,
            )


def build_resolved_entities(
    entities: list[Entity],
    id_map: dict[str, str],    # entity_id → canonical_id (from resolver)
) -> tuple[list[ResolvedEntity], dict[str, list[str]]]:
    """Collapse entities by canonical ID and collect their chunk occurrences."""
    canonical: dict[str, ResolvedEntity] = {}
    entity_to_chunks: dict[str, list[str]] = defaultdict(list)

    for ent in entities:
        cid = id_map.get(ent.entity_id, ent.entity_id)
        entity_to_chunks[cid].append(ent.chunk_id)

        if cid not in canonical:
            canonical[cid] = ResolvedEntity(
                canonical_id=cid,
                name=ent.canonical_name,
                type=ent.type,
                aliases=[ent.text],
                initials_key=compute_initials_key(ent.canonical_name),
            )
        else:
            # Accumulate aliases
            if ent.text not in canonical[cid].aliases:
                canonical[cid].aliases.append(ent.text)
            current = canonical[cid].name
            current_key = (len(_tokenize(current)), len(current), current.casefold())
            candidate_key = (
                len(_tokenize(ent.canonical_name)),
                len(ent.canonical_name),
                ent.canonical_name.casefold(),
            )
            if candidate_key > current_key:
                canonical[cid].name = ent.canonical_name
                canonical[cid].initials_key = compute_initials_key(ent.canonical_name)

    return list(canonical.values()), dict(entity_to_chunks)


def build_resolved_edges(
    triples: list[NormalizedTriple],
    id_map: dict[str, str],
    min_confidence: float = 0.0,
) -> list[ResolvedEdge]:
    """Collapse duplicate triples (same subj/rel/obj) by merging chunk_ids."""
    # Dedup key: (canonical_source_id, relation_type, canonical_target_id)
    edge_map: dict[tuple, ResolvedEdge] = {}

    for t in triples:
        if t.confidence < min_confidence:
            continue
        src_id = id_map.get(t.subject.entity_id, t.subject.entity_id)
        tgt_id = id_map.get(t.object.entity_id,  t.object.entity_id)
        key = (src_id, t.relation_type, tgt_id)

        if key in edge_map:
            e = edge_map[key]
            if t.chunk_id not in e.chunk_ids:
                e.chunk_ids.append(t.chunk_id)
            e.confidence = max(e.confidence, t.confidence)
            if t.extractor and t.extractor not in e.extractors:
                e.extractors.append(t.extractor)
            if t.confidence >= e.confidence and t.evidence:
                e.evidence = t.evidence
        else:
            edge_map[key] = ResolvedEdge(
                relation_type=t.relation_type,
                source_id=src_id,
                target_id=tgt_id,
                chunk_ids=[t.chunk_id],
                extractors=[t.extractor or "dep_pattern"],
                confidence=t.confidence,
                raw_predicate=t.raw_predicate,
                evidence=t.evidence,
            )

    return list(edge_map.values())
