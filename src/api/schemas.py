"""Pydantic v2 request/response schemas."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel


class UploadResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    id: str
    status: str
    filename: Optional[str] = None
    created_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None


class EntityResponse(BaseModel):
    id: str
    name: str
    type: str
    aliases: list[str] = []


class RelationResponse(BaseModel):
    type: str
    target_id: str
    target_name: str
    confidence: float
    evidence_count: int


class EntityDetailResponse(BaseModel):
    id: str
    name: str
    type: str
    outgoing: list[RelationResponse] = []


class SearchResult(BaseModel):
    id: str
    name: str
    type: str
    aliases: list[str] = []
    score: float = 0.0


class PathNode(BaseModel):
    id: str
    name: str
    type: str


class GraphStatsResponse(BaseModel):
    entity_count: int = 0
    edge_count: int = 0
    doc_count: int = 0
