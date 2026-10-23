"""Shared test fixtures — spaCy nlp instance, temp paths."""
from __future__ import annotations

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
