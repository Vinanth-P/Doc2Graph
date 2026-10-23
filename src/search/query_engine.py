"""Full-text search and Cypher query engine."""
from __future__ import annotations

from neo4j import AsyncDriver


class QueryEngine:
    def __init__(self, driver: AsyncDriver):
        self._driver = driver

    async def search_entities(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search over entity names and aliases."""
        escaped = _lucene_escape(query)
        async with self._driver.session() as s:
            result = await s.run(
                """
                CALL db.index.fulltext.queryNodes('entity_fulltext', $q)
                YIELD node, score
                RETURN node.id AS id, node.name AS name,
                       node.type AS type, node.aliases AS aliases,
                       score
                ORDER BY score DESC LIMIT $limit
                """,
                q=escaped, limit=limit,
            )
            return [r.data() async for r in result]

    async def get_entity_relations(self, entity_id: str) -> dict:
        """Return all relationships for an entity node."""
        async with self._driver.session() as s:
            result = await s.run(
                """
                MATCH (e:Entity {id: $id})
                OPTIONAL MATCH (e)-[r]->(other:Entity)
                RETURN e.id AS id, e.name AS name, e.type AS type,
                       collect({
                           type: type(r),
                           target_id: other.id,
                           target_name: other.name,
                           confidence: r.confidence,
                           evidence_count: r.evidence_count
                       }) AS outgoing
                """,
                id=entity_id,
            )
            rows = [r.data() async for r in result]
            return rows[0] if rows else {}

    async def shortest_path(
        self, source_id: str, target_id: str, max_hops: int = 5
    ) -> list[dict]:
        """Find the shortest path between two entities."""
        async with self._driver.session() as s:
            result = await s.run(
                """
                MATCH p = shortestPath(
                    (a:Entity {id: $src})-[*1..$max_hops]-(b:Entity {id: $tgt})
                )
                UNWIND nodes(p) AS n
                RETURN n.id AS id, n.name AS name, n.type AS type
                """,
                src=source_id, tgt=target_id, max_hops=max_hops,
            )
            return [r.data() async for r in result]

    async def get_graph_stats(self) -> dict:
        """Overall graph statistics."""
        async with self._driver.session() as s:
            result = await s.run(
                """
                MATCH (e:Entity) WITH count(e) AS entity_count
                MATCH ()-[r]->() WITH entity_count, count(r) AS edge_count
                MATCH (d:Document) WITH entity_count, edge_count, count(d) AS doc_count
                RETURN entity_count, edge_count, doc_count
                """
            )
            rows = [r.data() async for r in result]
            return rows[0] if rows else {}


def _lucene_escape(text: str) -> str:
    specials = r'\+-&&||!(){}[]^"~*?:/'
    return "".join(f"\\{c}" if c in specials else c for c in text)
