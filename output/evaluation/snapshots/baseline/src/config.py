"""Configuration — pydantic-settings v2 with YAML overlay and canonical model paths."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

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

    # Canonical Model & Whitelist Paths
    relation_scorer_model_path: str = "models/relation_scorer.joblib"
    entity_resolver_model_path: str = "models/entity_resolver.joblib"
    relations_config_path: str = "config/relations.yaml"

    # Entity resolution
    resolution_threshold: float = 0.85
    min_related_confidence: float = 0.60

    # Auth (empty = disabled)
    api_key: str = ""

    @model_validator(mode="after")
    def _merge_yaml(self) -> "Settings":
        # Resolve YAML path: check env var, then config/config.yaml, then legacy config.yaml
        path_str = os.environ.get("CONFIG_YAML")
        if not path_str:
            if os.path.exists("config/config.yaml"):
                path_str = "config/config.yaml"
            elif os.path.exists("config.yaml"):
                path_str = "config.yaml"

        if path_str and os.path.exists(path_str):
            with open(path_str, "r", encoding="utf-8") as f:
                overrides: dict = yaml.safe_load(f) or {}
            _SECRETS = {"neo4j_password", "api_key"}
            for k, v in overrides.items():
                if k in self.model_fields and k not in _SECRETS:
                    # Let explicit environment variables take precedence over YAML values
                    if k.upper() not in os.environ:
                        setattr(self, k, v)
        return self


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
