"""Document loaders — PDF, DOCX, TXT.

All loaders return raw text as a single string. The preprocessor is
responsible for sentence splitting, normalization, and chunking.
"""
from __future__ import annotations

import io
from pathlib import Path
from src.ingestion.text_validation import validate_text


# Minimum ratio of printable ASCII/Latin characters required to consider
# pdfplumber's extraction trustworthy. Below this threshold the page is
# treated as symbol-encoded or scanned and re-extracted via OCR.
_MIN_LATIN_RATIO = 0.60


def _latin_ratio(text: str) -> float:
    """Return the fraction of characters that are printable Latin/ASCII."""
    if not text:
        return 0.0
    latin = sum(
        1 for c in text
        if "\x20" <= c <= "\x7e"           # basic ASCII printable
        or "\xa0" <= c <= "\xff"            # Latin-1 supplement
        or c in ("\n", "\r", "\t")
    )
    return latin / len(text)


def load_pdf(content: bytes) -> str:
    """Extract text from a PDF byte string using pdfplumber (layout-aware).

    For each page, pdfplumber's extracted text is accepted only when the
    fraction of printable Latin characters is above ``_MIN_LATIN_RATIO``.
    Pages that fall below the threshold (symbol-encoded fonts, scanned pages,
    or purely graphical pages) fall back to Tesseract OCR so that garbled
    glyph sequences are never passed to the NLP pipeline.
    """
    import pdfplumber

    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text and _latin_ratio(page_text) >= _MIN_LATIN_RATIO:
                text_parts.append(page_text)
            else:
                # Scanned page or symbol-encoded font — fall back to OCR
                from src.ingestion.ocr import ocr_page_image
                img = page.to_image(resolution=300).original
                ocr_text = ocr_page_image(img)
                if ocr_text.strip():
                    text_parts.append(ocr_text)
    return "\n\n".join(text_parts)


def load_docx(content: bytes) -> str:
    """Extract text from a DOCX byte string using python-docx."""
    from docx import Document

    doc = Document(io.BytesIO(content))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def load_txt(content: bytes) -> str:
    """Decode Unicode text or Windows-1252 without silently replacing bytes."""
    try:
        if content.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = content.decode("utf-16")
        else:
            text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = content.decode("cp1252")
        except UnicodeDecodeError as exc:
            raise ValueError("Text encoding is unsupported. Save the document as UTF-8.") from exc
    return validate_text(text)


def load_document(content: bytes, ftype: str) -> str:
    """Dispatch to the correct loader based on detected file type."""
    loaders = {"pdf": load_pdf, "docx": load_docx, "txt": load_txt}
    loader = loaders.get(ftype)
    if loader is None:
        raise ValueError(f"Unsupported file type: {ftype!r}")
    return validate_text(loader(content))
