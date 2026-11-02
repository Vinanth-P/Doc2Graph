"""GDS PageRank and degree centrality ranking."""
from __future__ import annotations

import uuid
from neo4j import AsyncDriver


class RankingEngine:
    def __init__(self, driver: AsyncDriver):
        self._driver = driver

    async def top_entities_by_degree(self, limit: int = 10) -> list[dict]:
        """Return top entities by number of relationships (degree centrality)."""
        async with self._driver.session() as s:
            result = await s.run(
                """
                MATCH (e:Entity)-[r]-(:Entity)
                RETURN e.id AS id, e.name AS name, e.type AS type,
                       count(r) AS degree
                ORDER BY degree DESC LIMIT $limit
                """,
                limit=limit,
            )
            return [r.data() async for r in result]

    async def top_entities_by_pagerank(self, limit: int = 10) -> list[dict]:
        """Run GDS PageRank on the entity graph and return top entities."""
        graph_name = f"entity_graph_{uuid.uuid4().hex[:8]}"
        async with self._driver.session() as s:
            try:
                await s.run(
                    """
                    CALL gds.graph.project($graph_name, 'Entity', '*')
                    YIELD graphName
                    """,
                    graph_name=graph_name,
                )
                result = await s.run(
                    """
                    CALL gds.pageRank.stream($graph_name)
                    YIELD nodeId, score
                    MATCH (e:Entity) WHERE id(e) = nodeId
                    RETURN e.id AS id, e.name AS name, e.type AS type, score
                    ORDER BY score DESC LIMIT $limit
                    """,
                    graph_name=graph_name,
                    limit=limit,
                )
                return [r.data() async for r in result]
            finally:
                try:
                    await s.run(
                        "CALL gds.graph.drop($graph_name, false) YIELD graphName",
                        graph_name=graph_name,
                    )
                except Exception:
                    pass
