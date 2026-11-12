"""API-key authentication dependency (optional — disabled when api_key is empty)."""
from __future__ import annotations

from fastapi import Security, HTTPException, Depends
from fastapi.security import APIKeyHeader

from src.config import Settings, get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    key: str | None = Security(_api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    """Dependency — raises 401 if API key is set and doesn't match."""
    if not settings.api_key:
        return   # auth disabled
    if key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
