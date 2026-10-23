"""Search routes — full-text entity search."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.api.schemas import SearchResult
from src.config import Settings, get_settings
from src.graph.neo4j_client import get_driver
from src.search.query_engine import QueryEngine

router = APIRouter(prefix="/api/search", tags=["search"])


@router.get("", response_model=list[SearchResult])
async def search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    settings: Settings = Depends(get_settings),
):
    """Full-text search over entity names and aliases."""
    driver = await get_driver(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    engine = QueryEngine(driver)
    results = await engine.search_entities(q, limit)
    return [SearchResult(**r) for r in results]
