"""Hybrid extraction pipeline — the heart of the system.

Order of operations:

1. Ingest       — PDF/image → text + word boxes (+OCR for scans)
2. Classify     — detect document type (keyword scorer + optional zero-shot)
3. Extract      — run every *available* extractor:
                    LayoutLMv3 (layout-aware entity extraction)
                    Donut      (OCR-free visual QA)
                    Rules      (domain regex + layout heuristics)
                    Tesseract+spaCy (NER fallback)
4. Merge        — per-field vote: highest source-weighted confidence wins,
                  cross-extractor agreement boosts confidence
5. Tables       — structured table extraction (pdfplumber)

Every extracted field carries value, confidence, extractor provenance and,
when locatable, a normalized bounding box for frontend highlighting.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List

from .classifier import classify_detailed
from .extractors.base import ExtractedField
from .extractors.donut import DonutExtractor
from .extractors.layoutlm import LayoutLMExtractor
from .extractors.ocr_fallback import OcrSpacyExtractor
from .extractors.rules import RulesExtractor
from .ingest import IngestedDocument, ingest
from .tables import extract_tables

logger = logging.getLogger(__name__)

# Reliability prior per extractor — multiplies its self-reported confidence
# during the merge so a wildly over-confident weak source can't outvote a
# precise one.
_SOURCE_WEIGHT = {
    "layoutlmv3": 1.0,
    "donut": 0.9,
    "rules": 0.95,
    "ocr_spacy": 0.6,
}

_EXTRACTORS = [
    LayoutLMExtractor(),
    DonutExtractor(),
    RulesExtractor(),
    OcrSpacyExtractor(),
]


def _norm_value(v: str) -> str:
    return " ".join(v.lower().replace(",", "").replace("$", "").split())


def merge_fields(candidates: List[ExtractedField]) -> Dict[str, ExtractedField]:
    """Per-field merge with source weighting and agreement boosting."""
    by_field: Dict[str, List[ExtractedField]] = {}
    for c in candidates:
        by_field.setdefault(c.name, []).append(c)

    merged: Dict[str, ExtractedField] = {}
    for name, cands in by_field.items():
        scored = sorted(
            cands,
            key=lambda f: f.confidence * _SOURCE_WEIGHT.get(f.source, 0.5),
            reverse=True,
        )
        winner = scored[0]
        # agreement across extractors is strong evidence — boost confidence
        agreeing = {
            f.source for f in cands if _norm_value(f.value) == _norm_value(winner.value)
        }
        if len(agreeing) > 1:
            winner.confidence = round(min(0.99, winner.confidence + 0.1 * (len(agreeing) - 1)), 3)
            winner.source = "+".join(sorted(agreeing))
        merged[name] = winner
    return merged


def process_document(path: str) -> dict:
    """Run the full pipeline on one file and return the result dict."""
    t0 = time.time()
    doc: IngestedDocument = ingest(path)
    classification = classify_detailed(doc.full_text)
    doc_type, type_conf = classification["type"], classification["confidence"]

    candidates: List[ExtractedField] = []
    used: List[str] = []
    for extractor in _EXTRACTORS:
        try:
            if not extractor.available():
                continue
            fields = extractor.extract(doc, doc_type)
            if fields:
                used.append(extractor.name)
                candidates.extend(fields)
        except Exception as exc:  # extractor bugs must never kill the job
            logger.exception("extractor %s failed: %s", extractor.name, exc)

    fields = merge_fields(candidates)
    tables = extract_tables(path)
    elapsed = round(time.time() - t0, 2)

    return {
        "document_type": doc_type,
        "document_type_confidence": type_conf,
        "fields": {
            name: {
                "value": f.value,
                "confidence": f.confidence,
                "source": f.source,
                "page": f.page,
                "bbox": f.bbox,
            }
            for name, f in fields.items()
        },
        "tables": tables,
        "meta": {
            "classifier": classification["method"],
            "extractors_used": used,
            "is_scanned": doc.is_scanned,
            "ingest_source": doc.source,
            "num_pages": len(doc.pages),
            "processing_seconds": elapsed,
        },
        # kept out of the API response; used for RAG indexing
        "_pages_text": [p.text for p in doc.pages],
    }
