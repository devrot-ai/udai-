"""Shared extractor types and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

from ..ingest import IngestedDocument, Page


@dataclass
class ExtractedField:
    name: str
    value: str
    confidence: float  # [0, 1]
    source: str        # layoutlmv3 | donut | ocr_spacy | rules
    page: Optional[int] = None
    bbox: Optional[List[float]] = None  # normalized [x0, y0, x1, y1]


class Extractor(Protocol):
    name: str

    def available(self) -> bool: ...

    def extract(self, doc: IngestedDocument, doc_type: str) -> List[ExtractedField]: ...


def locate_value(page: Page, value: str) -> Optional[List[float]]:
    """Find the bounding box of a value string on a page by matching its
    token sequence against the page's word boxes. Returns a normalized
    [x0, y0, x1, y1] union box, or None if not found."""
    tokens = value.split()
    if not tokens or not page.words:
        return None
    words = page.words
    n = len(tokens)

    def norm(s: str) -> str:
        return s.strip(".,;:()[]$\"'").lower()

    target = [norm(t) for t in tokens]
    for i in range(len(words) - n + 1):
        window = [norm(w.text) for w in words[i : i + n]]
        if window == target:
            xs0 = min(w.x0 for w in words[i : i + n])
            ys0 = min(w.y0 for w in words[i : i + n])
            xs1 = max(w.x1 for w in words[i : i + n])
            ys1 = max(w.y1 for w in words[i : i + n])
            return [xs0, ys0, xs1, ys1]
    # fall back to first-token match for long values
    first = target[0]
    for w in page.words:
        if norm(w.text) == first and len(first) > 3:
            return [w.x0, w.y0, w.x1, w.y1]
    return None


def attach_location(doc: IngestedDocument, f: ExtractedField) -> ExtractedField:
    """Best-effort: attach page/bbox to a field by searching for its value."""
    if f.bbox is not None:
        return f
    for page in doc.pages:
        box = locate_value(page, f.value)
        if box:
            f.page = page.number
            f.bbox = box
            return f
    return f
