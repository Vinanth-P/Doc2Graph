"""GDS PageRank and degree centrality ranking."""
from __future__ import annotations

from neo4j import AsyncDriver


class RankingEngine:
    def __init__(self, driver: AsyncDriver):
        self._driver = driver

    async def top_entities_by_degree(self, limit: int = 10) -> list[dict]:
        """Return top entities by number of relationships (degree centrality)."""
        async with self._driver.session() as s:
            result = await s.run(
                """
                MATCH (e:Entity)-[r]-()
                RETURN e.id AS id, e.name AS name, e.type AS type,
                       count(r) AS degree
                ORDER BY degree DESC LIMIT $limit
                """,
                limit=limit,
            )
            return [r.data() async for r in result]

    async def top_entities_by_pagerank(self, limit: int = 10) -> list[dict]:
        """Run GDS PageRank on the entity graph and return top entities."""
        async with self._driver.session() as s:
            # GDS in-memory projection
            await s.run(
                """
                CALL gds.graph.project.cypher(
                    'entity_graph',
                    'MATCH (e:Entity) RETURN id(e) AS id',
                    'MATCH (a:Entity)-[r]->(b:Entity) RETURN id(a) AS source, id(b) AS target'
                ) YIELD graphName
                """
            )
            result = await s.run(
                """
                CALL gds.pageRank.stream('entity_graph')
                YIELD nodeId, score
                MATCH (e:Entity) WHERE id(e) = nodeId
                RETURN e.id AS id, e.name AS name, e.type AS type, score
                ORDER BY score DESC LIMIT $limit
                """,
                limit=limit,
            )
            rows = [r.data() async for r in result]
            # Drop the projection
            await s.run("CALL gds.graph.drop('entity_graph', false) YIELD graphName")
            return rows
