"""Full-text search and Cypher query engine."""
from __future__ import annotations

from neo4j import AsyncDriver


class QueryEngine:
    def __init__(self, driver: AsyncDriver):
        self._driver = driver

    async def search_entities(self, query: str, limit: int = 20, doc_id: str | None = None) -> list[dict]:
        """Full-text search over entity names and aliases."""
        escaped = _lucene_escape(query)
        async with self._driver.session() as s:
            result = await s.run(
                """
                CALL db.index.fulltext.queryNodes('entity_fulltext', $q)
                YIELD node, score
                WHERE $doc_id IS NULL OR EXISTS {
                    MATCH (node)-[:MENTIONED_IN]->(:Chunk {doc_id: $doc_id})
                }
                RETURN node.id AS id, node.name AS name,
                       node.type AS type, node.aliases AS aliases,
                       score
                ORDER BY score DESC LIMIT $limit
                """,
                q=escaped, limit=limit, doc_id=doc_id,
            )
            return [r.data() async for r in result]

    async def get_entity_relations(self, entity_id: str, doc_id: str | None = None) -> dict:
        """Return all relationships for an entity node."""
        async with self._driver.session() as s:
            result = await s.run(
                """
                MATCH (e:Entity {id: $id})
                WHERE $doc_id IS NULL OR EXISTS {
                    MATCH (e)-[:MENTIONED_IN]->(:Chunk {doc_id: $doc_id})
                }
                OPTIONAL MATCH (e)-[r]->(other:Entity)
                WHERE $doc_id IS NULL OR any(cid IN coalesce(r.chunk_ids, [])
                    WHERE cid STARTS WITH $doc_id + ':')
                RETURN e.id AS id, e.name AS name, e.type AS type,
                       collect(CASE WHEN other IS NOT NULL THEN {
                           type: type(r),
                           target_id: other.id,
                           target_name: other.name,
                           confidence: r.confidence,
                           evidence_count: CASE WHEN $doc_id IS NULL THEN r.evidence_count
                               ELSE size([cid IN coalesce(r.chunk_ids, [])
                                   WHERE cid STARTS WITH $doc_id + ':']) END
                       } END) AS outgoing
                """,
                id=entity_id, doc_id=doc_id,
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

    async def get_graph_stats(self, doc_id: str | None = None) -> dict:
        """Overall graph statistics."""
        async with self._driver.session() as s:
            result = await s.run(
                """
                OPTIONAL MATCH (e:Entity)
                WHERE $doc_id IS NULL OR EXISTS {
                    MATCH (e)-[:MENTIONED_IN]->(:Chunk {doc_id: $doc_id})
                }
                WITH count(e) AS entity_count
                OPTIONAL MATCH (:Entity)-[r]->(:Entity)
                WHERE $doc_id IS NULL OR any(cid IN coalesce(r.chunk_ids, [])
                    WHERE cid STARTS WITH $doc_id + ':')
                WITH entity_count, count(r) AS edge_count
                OPTIONAL MATCH (d:Document)
                WHERE $doc_id IS NULL OR d.id = $doc_id
                RETURN entity_count, edge_count, count(d) AS doc_count
                """,
                doc_id=doc_id,
            )
            rows = [r.data() async for r in result]
            if rows and rows[0].get("entity_count") is not None:
                return rows[0]
            return {"entity_count": 0, "edge_count": 0, "doc_count": 0}

    async def get_visualization_data(
        self, limit: int = 150, entity_id: str | None = None, doc_id: str | None = None
    ) -> dict:
        """Fetch nodes and edges formatted for D3.js force-directed graph."""
        async with self._driver.session() as s:
            # Keep the stored direction even when focusing an incoming edge.
            query = """
                MATCH (a:Entity)-[r]->(b:Entity)
                WHERE ($eid IS NULL OR a.id = $eid OR b.id = $eid)
                  AND ($doc_id IS NULL OR any(cid IN coalesce(r.chunk_ids, [])
                      WHERE cid STARTS WITH $doc_id + ':'))
                RETURN a.id AS src_id, a.name AS src_name, a.type AS src_type,
                       type(r) AS rel_type, r.confidence AS confidence,
                       CASE WHEN $doc_id IS NULL THEN
                           coalesce(r.evidence_count, size(coalesce(r.chunk_ids, [])))
                       ELSE size([cid IN coalesce(r.chunk_ids, [])
                           WHERE cid STARTS WITH $doc_id + ':']) END AS evidence_count,
                       b.id AS tgt_id, b.name AS tgt_name, b.type AS tgt_type
                LIMIT $limit
                """
            params = {"limit": limit, "eid": entity_id, "doc_id": doc_id}

            result = await s.run(query, **params)
            rows = [r.data() async for r in result]

            nodes_dict: dict[str, dict] = {}
            links: list[dict] = []
            for r in rows:
                if r["src_id"] not in nodes_dict:
                    nodes_dict[r["src_id"]] = {
                        "id": r["src_id"],
                        "name": r["src_name"],
                        "type": r["src_type"],
                    }
                if r["tgt_id"] not in nodes_dict:
                    nodes_dict[r["tgt_id"]] = {
                        "id": r["tgt_id"],
                        "name": r["tgt_name"],
                        "type": r["tgt_type"],
                    }
                links.append({
                    "source": r["src_id"],
                    "target": r["tgt_id"],
                    "type": r["rel_type"],
                    "confidence": r.get("confidence", 1.0),
                    "evidence_count": r.get("evidence_count", 1),
                })

            # Include isolated entities (or an isolated focused entity), but
            # only when mentioned in the selected document.
            node_query = await s.run(
                """
                MATCH (n:Entity)
                WHERE ($eid IS NULL OR n.id = $eid)
                  AND ($doc_id IS NULL OR EXISTS {
                      MATCH (n)-[:MENTIONED_IN]->(:Chunk {doc_id: $doc_id})
                  })
                RETURN n.id AS id, n.name AS name, n.type AS type LIMIT $limit
                """,
                limit=limit, eid=entity_id, doc_id=doc_id,
            )
            async for nr in node_query:
                ndata = nr.data()
                if ndata["id"] not in nodes_dict:
                    nodes_dict[ndata["id"]] = ndata

            return {"nodes": list(nodes_dict.values()), "links": links}


def _lucene_escape(text: str) -> str:
    specials = r'\+-&&||!(){}[]^"~*?:/'
    return "".join(f"\\{c}" if c in specials else c for c in text)
