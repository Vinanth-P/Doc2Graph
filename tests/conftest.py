"""Shared test fixtures — spaCy nlp instance, temp paths."""
from __future__ import annotations

import pytest
import spacy


@pytest.fixture(scope="session")
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture
def sample_text():
    return (
        "Elon Musk founded Tesla in 2003. "
        "He also leads SpaceX. "
        "Tesla is headquartered in Austin, Texas. "
        "Google acquired DeepMind in 2014."
    )
