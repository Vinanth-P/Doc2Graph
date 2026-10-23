"""Neo4j async client — connection management, schema init, batched writes."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession


_driver_by_loop: dict[asyncio.AbstractEventLoop | None, AsyncDriver] = {}


def create_driver(uri: str, user: str, password: str) -> AsyncDriver:
    """Create a new driver instance tied to the caller's active event loop."""
    return AsyncGraphDatabase.driver(uri, auth=(user, password))


async def get_driver(uri: str, user: str, password: str) -> AsyncDriver:
    """Get or create singleton driver for the current event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    driver = _driver_by_loop.get(loop)
    if driver is None:
        driver = create_driver(uri, user, password)
        _driver_by_loop[loop] = driver
    return driver


async def close_driver() -> None:
    """Close the driver registered for the current event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    driver = _driver_by_loop.pop(loop, None)
    if driver is not None:
        await driver.close()


async def close_all_drivers() -> None:
    """Close all drivers across all event loops."""
    for driver in list(_driver_by_loop.values()):
        try:
            await driver.close()
        except Exception:
            pass
    _driver_by_loop.clear()


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
