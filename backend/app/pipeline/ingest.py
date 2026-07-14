"""Document ingestion: PDF / image -> normalized page representation.

Native PDFs are parsed with pdfplumber (text + word-level bounding boxes).
Scanned PDFs (no embedded text layer) and images are rasterized and pushed
through Tesseract OCR when available. Page images are rendered lazily with
pypdfium2 for the visual extractors (LayoutLMv3 / Donut) and the frontend
highlight overlay.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


@dataclass
class Word:
    text: str
    # normalized coordinates in [0, 1] relative to page size
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class Page:
    number: int  # 1-based
    text: str
    words: List[Word] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    image: Optional[object] = None  # PIL.Image, populated lazily


@dataclass
class IngestedDocument:
    path: str
    pages: List[Page]
    is_scanned: bool = False
    source: str = "pdf-native"  # pdf-native | pdf-ocr | image-ocr | image

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


def _ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401
        from pytesseract import get_tesseract_version

        get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_page(pil_image) -> Page:
    """OCR a PIL image into a Page with word-level boxes."""
    import pytesseract

    data = pytesseract.image_to_data(
        pil_image, output_type=pytesseract.Output.DICT
    )
    w, h = pil_image.size
    words: List[Word] = []
    lines: dict = {}
    for i, token in enumerate(data["text"]):
        token = token.strip()
        if not token:
            continue
        x, y = data["left"][i], data["top"][i]
        bw, bh = data["width"][i], data["height"][i]
        words.append(Word(token, x / w, y / h, (x + bw) / w, (y + bh) / h))
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(token)
    text = "\n".join(" ".join(toks) for toks in lines.values())
    return Page(number=1, text=text, words=words, width=w, height=h, image=pil_image)


def render_pdf_page(path: str, page_number: int, scale: float = 2.0):
    """Render one PDF page (1-based) to a PIL image via pypdfium2."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        page = pdf[page_number - 1]
        bitmap = page.render(scale=scale)
        return bitmap.to_pil()
    finally:
        pdf.close()


def _ingest_pdf(path: str) -> IngestedDocument:
    import pdfplumber

    pages: List[Page] = []
    total_chars = 0
    with pdfplumber.open(path) as pdf:
        for idx, p in enumerate(pdf.pages, start=1):
            text = p.extract_text() or ""
            total_chars += len(text.strip())
            words = [
                Word(
                    w["text"],
                    w["x0"] / p.width,
                    w["top"] / p.height,
                    w["x1"] / p.width,
                    w["bottom"] / p.height,
                )
                for w in (p.extract_words() or [])
            ]
            pages.append(
                Page(number=idx, text=text, words=words, width=p.width, height=p.height)
            )

    # Heuristic: fewer than ~30 chars/page of embedded text -> scanned document
    if pages and total_chars / len(pages) < 30:
        if _ocr_available():
            logger.info("PDF %s has no text layer; running OCR", path)
            ocr_pages: List[Page] = []
            for idx in range(1, len(pages) + 1):
                image = render_pdf_page(path, idx)
                page = _ocr_page(image)
                page.number = idx
                ocr_pages.append(page)
            return IngestedDocument(path, ocr_pages, is_scanned=True, source="pdf-ocr")
        logger.warning("Scanned PDF %s but Tesseract is unavailable", path)
        return IngestedDocument(path, pages, is_scanned=True, source="pdf-native")

    return IngestedDocument(path, pages, is_scanned=False, source="pdf-native")


def _ingest_image(path: str) -> IngestedDocument:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if _ocr_available():
        page = _ocr_page(image)
        return IngestedDocument(path, [page], is_scanned=True, source="image-ocr")
    w, h = image.size
    page = Page(number=1, text="", words=[], width=w, height=h, image=image)
    return IngestedDocument(path, [page], is_scanned=True, source="image")


def ingest(path: str) -> IngestedDocument:
    """Ingest a PDF or image file into the normalized page representation."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _ingest_pdf(path)
    if ext in IMAGE_EXTENSIONS:
        return _ingest_image(path)
    raise ValueError(f"Unsupported file type: {ext}")
