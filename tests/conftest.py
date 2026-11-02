"""Shared test fixtures — spaCy nlp instance, temp paths, and offline Neo4j guards."""
from __future__ import annotations

import os
import pytest
from src.ingestion.preprocessor import get_nlp


@pytest.fixture(scope="session")
def nlp():
    return get_nlp()


@pytest.fixture
def sample_text():
    return (
        "Elon Musk founded Tesla in 2003. "
        "He also leads SpaceX. "
        "Tesla is headquartered in Austin, Texas. "
        "Google acquired DeepMind in 2014."
    )


@pytest.fixture(autouse=True)
def mock_neo4j_lifespan_when_offline(monkeypatch):
    """Bypass Neo4j schema initialization during unit tests when Neo4j is offline."""
    if os.environ.get("DOC2GRAPH_NEO4J_TESTS") != "1":
        from src.api import app as api_app
        async def _noop(*args, **kwargs):
            return None
        monkeypatch.setattr(api_app, "run_schema", _noop)
