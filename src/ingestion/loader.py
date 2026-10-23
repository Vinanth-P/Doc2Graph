"""Document loaders — PDF, DOCX, TXT.

All loaders return raw text as a single string. The preprocessor is
responsible for sentence splitting, normalization, and chunking.
"""
from __future__ import annotations

import io
from pathlib import Path


def load_pdf(content: bytes) -> str:
    """Extract text from a PDF byte string using pdfplumber (layout-aware)."""
    import pdfplumber

    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
            else:
                # Scanned page — fall back to OCR
                from src.ingestion.ocr import ocr_page_image
                img = page.to_image(resolution=300).original
                text_parts.append(ocr_page_image(img))
    return "\n\n".join(text_parts)


def load_docx(content: bytes) -> str:
    """Extract text from a DOCX byte string using python-docx."""
    from docx import Document

    doc = Document(io.BytesIO(content))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def load_txt(content: bytes) -> str:
    """Decode a plain-text file. Tries UTF-8 then latin-1."""
    # Strip UTF-8 BOM if present
    if content.startswith(b"\xef\xbb\xbf"):
        content = content[3:]
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def load_document(content: bytes, ftype: str) -> str:
    """Dispatch to the correct loader based on detected file type."""
    loaders = {"pdf": load_pdf, "docx": load_docx, "txt": load_txt}
    loader = loaders.get(ftype)
    if loader is None:
        raise ValueError(f"Unsupported file type: {ftype!r}")
    return loader(content)
