"""Sentence provenance and window coverage regressions."""
import pytest

from src.ingestion.preprocessor import SentenceChunk, build_windows, preprocess


def test_trimmed_sentence_offsets_match_source(monkeypatch):
    monkeypatch.setattr("src.ingestion.preprocessor.detect_language", lambda text: "en")
    text = "Google acquired Microsoft.\n\n  Elon Musk founded Tesla.\n  Tesla is in Austin."
    result = preprocess(text.encode(), "offsets.txt")
    assert len(result.chunks) >= 3
    for chunk in result.chunks:
        assert result.raw_text[chunk.start_char:chunk.end_char] == chunk.text


@pytest.mark.parametrize("size,stride", [(0, 1), (-1, 1), (3, 0), (3, -1), (1, 2)])
def test_invalid_windows_fail_before_processing(size, stride):
    with pytest.raises(ValueError):
        build_windows([], size, stride)


@pytest.mark.parametrize("count", range(8))
@pytest.mark.parametrize("size,stride", [(1, 1), (3, 1), (3, 2), (3, 3)])
def test_windows_cover_every_sentence(count, size, stride):
    chunks = [SentenceChunk(f"d:{i}", "d", i, "Text.", 0, 5) for i in range(count)]
    windows = build_windows(chunks, size, stride)
    assert {c.chunk_id for window, _ in windows for c in window} == {c.chunk_id for c in chunks}
    assert all(center in window for window, center in windows)
