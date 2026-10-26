"""Configuration — pydantic-settings v2 with YAML overlay."""
from __future__ import annotations

import os
from typing import Literal

import yaml
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j"

    # Pipeline
    pipeline_mode: Literal["fast", "accurate"] = "accurate"
    max_upload_mb: int = 50
    window_size: int = 3
    window_stride: int = 2
    debug_pipeline: bool = False
    intermediate_graph_dir: str = "data/processed"

    # Entity resolution
    resolution_threshold: float = 0.85

    # Auth (empty = disabled)
    api_key: str = ""

    @model_validator(mode="after")
    def _merge_yaml(self) -> "Settings":
        path = os.environ.get("CONFIG_YAML", "config.yaml")
        if os.path.exists(path):
            with open(path) as f:
                overrides: dict = yaml.safe_load(f) or {}
            _SECRETS = {"neo4j_password", "api_key"}
            for k, v in overrides.items():
                if k in self.model_fields and k not in _SECRETS:
                    setattr(self, k, v)
        return self


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
