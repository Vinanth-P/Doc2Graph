"""Neo4j async client — connection management, schema init, batched writes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession


_driver: AsyncDriver | None = None


async def get_driver(uri: str, user: str, password: str) -> AsyncDriver:
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    return _driver


async def close_driver() -> None:
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


async def run_schema(driver: AsyncDriver, cypher_path: str = "scripts/setup_neo4j.cypher") -> None:
    """Execute all statements in setup_neo4j.cypher (idempotent — uses IF NOT EXISTS)."""
    cypher = Path(cypher_path).read_text()
    async with driver.session() as session:
        for stmt in cypher.split(";"):
            stmt = stmt.strip()
            if stmt:
                await session.run(stmt)


async def batched_run(
    session: AsyncSession,
    query: str,
    params: dict[str, Any],
) -> None:
    """Run a parameterized Cypher query."""
    await session.run(query, **params)
