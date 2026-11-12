"""Pydantic v2 request/response schemas & pipeline/graph validation boundaries."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ── API Request / Response Schemas ──────────────────────────────────────────

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


# ── Pipeline / Graph Boundary Validation Schemas ────────────────────────────

class ResolvedEntityPayload(BaseModel):
    """Strict schema for entities entering intermediate storage and Neo4j."""
    id: str = Field(..., min_length=1, description="Canonical entity ID")
    name: str = Field(..., min_length=1, description="Primary entity display name")
    type: str = Field(..., min_length=1, description="Entity type")
    aliases: list[str] = Field(default_factory=list)
    initials_key: str = ""

    @field_validator("id", "name")
    @classmethod
    def reject_corrupt_or_binary_content(cls, v: str) -> str:
        if "\ufffd" in v or "%PDF" in v or "\x00" in v:
            raise ValueError(f"Corrupt or raw binary content detected: {v!r}")
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty or whitespace only")
        return cleaned

    @field_validator("type")
    @classmethod
    def validate_entity_type(cls, v: str) -> str:
        from src.type_map import TYPE_TO_LABEL, SPACY_TO_INTERNAL
        allowed = set(TYPE_TO_LABEL) | set(SPACY_TO_INTERNAL.values())
        if v not in allowed:
            raise ValueError(f"Entity type {v!r} is not an allowed entity type")
        return v


class ResolvedEdgePayload(BaseModel):
    """Strict schema for normalized relation edges entering intermediate storage and Neo4j."""
    relation_type: str = Field(..., min_length=1)
    source_id: str = Field(..., min_length=1)
    target_id: str = Field(..., min_length=1)
    chunk_ids: list[str] = Field(default_factory=list)
    extractors: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    raw_predicate: str = ""
    evidence: str = ""

    @field_validator("relation_type")
    @classmethod
    def validate_relation_type(cls, v: str) -> str:
        from src.graph.graph_builder import GraphBuilder
        if v not in GraphBuilder.CANONICAL_TYPES:
            raise ValueError(f"Relation type {v!r} is not in canonical whitelist")
        return v

    @field_validator("source_id", "target_id")
    @classmethod
    def validate_entity_ids(cls, v: str) -> str:
        if "\ufffd" in v or "\x00" in v or not v.strip():
            raise ValueError(f"Invalid entity ID reference: {v!r}")
        return v.strip()

    @field_validator("raw_predicate", "evidence")
    @classmethod
    def reject_corrupt_text(cls, v: str) -> str:
        if "\ufffd" in v or "\x00" in v:
            raise ValueError(f"Corrupt text detected in edge payload: {v!r}")
        return v


class DocumentGraphPayload(BaseModel):
    """Composite payload validating all document entities, edges, and provenance."""
    doc_id: str = Field(..., min_length=1)
    filename: str = Field(..., min_length=1)
    entities: list[ResolvedEntityPayload]
    edges: list[ResolvedEdgePayload]
    mentions: dict[str, list[str]] = Field(default_factory=dict)
