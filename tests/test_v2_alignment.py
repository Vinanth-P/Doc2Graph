"""Tests for source offset alignment, Unicode NFKC mapping, and span preservation."""
import pytest
import unicodedata
from src.ingestion.source_map import SourceMap


@pytest.mark.parametrize('raw', ['Cafe\u0301', '\u1100\u1161\u11a8', 'a\u0315\u0300', '\ufb03'])
def test_normalization_matches_whole_string_nfkc(raw):
    normalized, mapping = SourceMap.create(raw)
    assert normalized == unicodedata.normalize('NFKC', raw)
    assert mapping.normalized_to_source(0, len(normalized)) == (0, len(raw))


def test_partial_ligature_span_covers_original_character():
    normalized, mapping = SourceMap.create('\ufb03')
    assert normalized == 'ffi'
    for start in range(3):
        assert mapping.normalized_to_source(start, start + 1) == (0, 1)
    assert mapping.source_to_normalized(0, 1) == (0, 3)


def test_combining_mark_and_collapsed_whitespace_cover_contributors():
    _, mapping = SourceMap.create('e\u0301')
    assert mapping.source_to_normalized(1, 2) == (0, 1)
    normalized, mapping = SourceMap.create('A\t\r\n\u00a0 B', normalize_whitespace=True)
    assert normalized == 'A B'
    assert mapping.normalized_to_source(1, 2) == (1, 6)
    assert mapping.source_to_normalized(3, 4) == (1, 2)


def test_empty_map_and_out_of_bounds():
    _, mapping = SourceMap.create('')
    assert mapping.source_to_normalized(0, 0) == (0, 0)
    assert mapping.normalized_to_source(0, 0) == (0, 0)
    with pytest.raises(IndexError):
        mapping.normalized_to_source(0, 1)


def test_identity_source_map():
    text = "Simple plain text without modifications."
    norm, sm = SourceMap.create(text, normalize_unicode=False, normalize_whitespace=False)
    assert norm == text
    
    raw_s, raw_e = sm.normalized_to_source(7, 12)
    assert text[raw_s:raw_e] == "plain"
    
    norm_s, norm_e = sm.source_to_normalized(7, 12)
    assert norm[norm_s:norm_e] == "plain"


def test_unicode_normalization_source_map():
    # \uFB03 is 'ffi' ligature in Unicode
    raw_text = "The o\uFB03ce is in London."
    norm_text, sm = SourceMap.create(raw_text, normalize_unicode=True)
    assert "office" in norm_text
    
    # "office" in norm_text starts at index 4 and ends at index 10 (length 6)
    idx = norm_text.index("office")
    norm_s, norm_e = idx, idx + len("office")
    
    raw_s, raw_e = sm.normalized_to_source(norm_s, norm_e)
    # in raw_text, "o\uFB03ce" has length 4 (o, \uFB03, c, e)
    assert raw_text[raw_s:raw_e] == "o\uFB03ce"


def test_whitespace_collapsing_source_map():
    raw_text = "Atlas   acquired   Beacon."
    norm_text, sm = SourceMap.create(raw_text, normalize_unicode=True, normalize_whitespace=True)
    assert norm_text == "Atlas acquired Beacon."
    
    beacon_norm_start = norm_text.index("Beacon")
    beacon_norm_end = beacon_norm_start + len("Beacon")
    
    raw_s, raw_e = sm.normalized_to_source(beacon_norm_start, beacon_norm_end)
    assert raw_text[raw_s:raw_e] == "Beacon"
