"""Reject binary payloads and damaged extraction before they reach NLP."""
from __future__ import annotations

import re

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f-\x9f]")


def validate_text(text: str) -> str:
    if not text.strip():
        raise ValueError("No readable text was extracted. Use a text-based document or OCR.")
    if text.lstrip("\ufeff \r\n\t").startswith(("%PDF-", "PK\x03\x04", "{\\rtf")):
        raise ValueError("Document binary data was supplied as text. Use the PDF or DOCX loader.")
    if "\ufffd" in text or _CONTROL.search(text):
        raise ValueError("Document text contains corrupt encoding or binary characters. Re-export it as PDF, DOCX, or UTF-8 text.")
    return text
