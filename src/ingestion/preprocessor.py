"""Preprocessor — sentence splitting, normalization, SHA-256 hashing, language detection.

This module is the *single* sentence splitter used by the entire pipeline.
spaCy's parser is used instead of Punkt to keep sentence indices
consistent with the NER and dep-parse stages (which also use spaCy).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import spacy
from spacy.symbols import ORTH
from langdetect import detect, DetectorFactory, LangDetectException
from src.ingestion.text_validation import validate_text

# Make langdetect deterministic
DetectorFactory.seed = 0

# Shared spaCy model (loaded once at module level)
_nlp: spacy.Language | None = None


def get_nlp() -> spacy.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
        # Some mixed-case names ending in capitals (e.g. SpaceX.) are
        # tokenized as abbreviations, hiding the sentence-ending period.
        # Known single-word organizations are names, not abbreviations.
        gazetteer = Path(__file__).resolve().parents[2] / "data/gazetteers/organizations.txt"
        if gazetteer.exists():
            for line in gazetteer.read_text(encoding="utf-8").splitlines():
                name = line.strip()
                if name.isalpha() and len(name) > 1:
                    for surface in {name, name.upper()}:
                        _nlp.tokenizer.add_special_case(
                            surface + ".", [{ORTH: surface}, {ORTH: "."}],
                        )
    return _nlp


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SentenceChunk:
    """A single sentence extracted from a document."""
    chunk_id: str        # "<doc_hash>:<sent_idx>" — deterministic
    doc_id: str          # SHA-256 of the raw document bytes
    sent_idx: int        # 0-based sentence index within the document
    text: str            # sentence text (original, never rewritten)
    start_char: int      # character offset in the full document text
    end_char: int


@dataclass
class PreprocessedDoc:
    doc_id: str
    filename: str
    language: str
    raw_text: str
    chunks: list[SentenceChunk] = field(default_factory=list)


# ── Utilities ─────────────────────────────────────────────────────────────────

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_text(text: str) -> str:
    """Unicode NFKC normalization + collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_language(text: str) -> str:
    """Return ISO 639-1 language code. Returns 'unknown' on failure."""
    try:
        return detect(text[:2000])
    except LangDetectException:
        return "unknown"


# ── Main API ──────────────────────────────────────────────────────────────────

def preprocess(
    raw_bytes: bytes,
    filename: str,
    text: str | None = None,
) -> PreprocessedDoc:
    """Full preprocessing pipeline: normalize → detect language → split sentences.

    Raises:
        ValueError: if detected language is not English.
    """
    doc_id = sha256_hex(raw_bytes)
    # Use extracted/cleaned text if provided (e.g. from PDF/DOCX loader), else decode bytes
    if text is None:
        from src.ingestion.loader import load_txt
        text = load_txt(raw_bytes)
    validate_text(text)
    text = normalize_text(text)

    # Language gate
    lang = detect_language(text)
    if lang != "en":
        raise ValueError(
            f"Document language '{lang}' is not supported. English only."
        )

    # Sentence splitting via spaCy (same model as NER + dep-parse)
    nlp = get_nlp()
    doc = nlp(text)

    chunks: list[SentenceChunk] = []
    for idx, sent in enumerate(doc.sents):
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        chunk_id = f"{doc_id}:{idx}"
        chunks.append(SentenceChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            sent_idx=idx,
            text=sent_text,
            start_char=sent.start_char,
            end_char=sent.end_char,
        ))

    return PreprocessedDoc(
        doc_id=doc_id,
        filename=filename,
        language=lang,
        raw_text=text,
        chunks=chunks,
    )


def build_windows(
    chunks: list[SentenceChunk],
    window_size: int = 3,
    stride: int = 2,
) -> list[tuple[list[SentenceChunk], SentenceChunk]]:
    """Slide a window over sentence chunks.

    Returns a list of (window_chunks, center_chunk) tuples.
    Triples extracted from a window are attributed to the center chunk's ID.
    Stride 2 means each sentence is covered by ~1.5 windows on average,
    vs stride 1 which would cover each sentence ~window_size times.
    """
    windows = []
    i = 0
    while i < len(chunks):
        window = chunks[i : i + window_size]
        center_idx = len(window) // 2
        center = window[center_idx]
        windows.append((window, center))
        i += stride
    return windows
