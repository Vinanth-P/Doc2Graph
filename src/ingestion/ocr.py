"""OCR fallback for scanned PDF pages using pytesseract."""
from __future__ import annotations

from PIL.Image import Image


def ocr_page_image(image: Image) -> str:
    """Run Tesseract OCR on a PIL Image and return the extracted text."""
    try:
        import pytesseract
    except ImportError:
        raise ImportError(
            "pytesseract is required for OCR. "
            "Install it with: pip install pytesseract"
        )
    # PSM 3 = auto page segmentation (default)
    config = "--psm 3 --oem 3"
    return pytesseract.image_to_string(image, lang="eng", config=config)
