"""Graph exploration routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.schemas import EntityDetailResponse, GraphStatsResponse, PathNode
from src.config import Settings, get_settings
from src.graph.neo4j_client import get_driver
from src.search.query_engine import QueryEngine
from src.search.ranking import RankingEngine

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/entities/{entity_id}", response_model=EntityDetailResponse)
async def get_entity(entity_id: str, doc_id: str | None = Query(None, min_length=1), settings: Settings = Depends(get_settings)):
    try:
        driver = await get_driver(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        engine = QueryEngine(driver)
        data = await engine.get_entity_relations(entity_id, doc_id=doc_id)
        if not data:
            raise HTTPException(404, f"Entity {entity_id!r} not found")
        return EntityDetailResponse(**data)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Error fetching entity details for %s: %s", entity_id, exc)
        raise HTTPException(503, f"Neo4j database unavailable: {exc}")


@router.get("/path", response_model=list[PathNode])
async def get_path(
    source: str = Query(..., description="Source entity ID"),
    target: str = Query(..., description="Target entity ID"),
    max_hops: int = Query(5, ge=1, le=10),
    settings: Settings = Depends(get_settings),
):
    try:
        driver = await get_driver(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        engine = QueryEngine(driver)
        nodes = await engine.shortest_path(source, target, max_hops)
        return [PathNode(**n) for n in nodes]
    except Exception as exc:
        logger.warning("Error finding path: %s", exc)
        return []


@router.get("/stats", response_model=GraphStatsResponse)
async def get_stats(doc_id: str | None = Query(None, min_length=1), settings: Settings = Depends(get_settings)):
    try:
        driver = await get_driver(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        engine = QueryEngine(driver)
        stats = await engine.get_graph_stats(doc_id=doc_id)
        return GraphStatsResponse(
            entity_count=stats.get("entity_count", 0),
            edge_count=stats.get("edge_count", 0),
            doc_count=stats.get("doc_count", 0),
        )
    except Exception as exc:
        logger.warning("Failed to fetch graph stats: %s", exc)
        return GraphStatsResponse(entity_count=0, edge_count=0, doc_count=0)


@router.get("/visualization")
async def get_visualization(
    limit: int = Query(150, ge=1, le=500),
    entity_id: str | None = Query(None, description="Center entity ID for subgraph"),
    doc_id: str | None = Query(None, min_length=1, description="Only evidence from this document"),
    settings: Settings = Depends(get_settings),
):
    """Return nodes and links for D3.js visualization."""
    try:
        driver = await get_driver(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        engine = QueryEngine(driver)
        return await engine.get_visualization_data(limit=limit, entity_id=entity_id, doc_id=doc_id)
    except Exception as exc:
        logger.warning("Failed to fetch visualization data: %s", exc)
        return {"nodes": [], "links": []}


@router.delete("/clear")
async def clear_graph(settings: Settings = Depends(get_settings)):
    """Delete all nodes and relationships from Neo4j."""
    try:
        driver = await get_driver(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        async with driver.session() as s:
            await s.run("MATCH (n) DETACH DELETE n")
        return {"status": "ok", "message": "Graph cleared"}
    except Exception as exc:
        logger.error("Failed to clear graph: %s", exc)
        raise HTTPException(500, f"Failed to clear graph: {exc}")


@router.get("/hubs")
async def get_hubs(
    limit: int = Query(10, ge=1, le=50),
    settings: Settings = Depends(get_settings),
):
    driver = await get_driver(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    ranking = RankingEngine(driver)
    return await ranking.top_entities_by_degree(limit)
