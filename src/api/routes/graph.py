"""Graph exploration routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.schemas import EntityDetailResponse, GraphStatsResponse, PathNode
from src.config import Settings, get_settings
from src.graph.neo4j_client import get_driver
from src.search.query_engine import QueryEngine
from src.search.ranking import RankingEngine

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/entities/{entity_id}", response_model=EntityDetailResponse)
async def get_entity(entity_id: str, settings: Settings = Depends(get_settings)):
    driver = await get_driver(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    engine = QueryEngine(driver)
    data = await engine.get_entity_relations(entity_id)
    if not data:
        raise HTTPException(404, f"Entity {entity_id!r} not found")
    return EntityDetailResponse(**data)


@router.get("/path", response_model=list[PathNode])
async def get_path(
    source: str = Query(..., description="Source entity ID"),
    target: str = Query(..., description="Target entity ID"),
    max_hops: int = Query(5, ge=1, le=10),
    settings: Settings = Depends(get_settings),
):
    driver = await get_driver(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    engine = QueryEngine(driver)
    nodes = await engine.shortest_path(source, target, max_hops)
    return [PathNode(**n) for n in nodes]


@router.get("/stats", response_model=GraphStatsResponse)
async def get_stats(settings: Settings = Depends(get_settings)):
    driver = await get_driver(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    engine = QueryEngine(driver)
    stats = await engine.get_graph_stats()
    return GraphStatsResponse(
        entity_count=stats.get("entity_count", 0),
        edge_count=stats.get("edge_count", 0),
        doc_count=stats.get("doc_count", 0),
    )


@router.get("/hubs")
async def get_hubs(
    limit: int = Query(10, ge=1, le=50),
    settings: Settings = Depends(get_settings),
):
    driver = await get_driver(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    ranking = RankingEngine(driver)
    return await ranking.top_entities_by_degree(limit)
