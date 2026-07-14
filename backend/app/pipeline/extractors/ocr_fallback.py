"""Tesseract + spaCy fallback extractor.

Named-entity based extraction over the OCR (or native) text layer. This is
the weakest layer of the hybrid pipeline — used when the neural extractors
are unavailable or produce nothing — and also serves as the *baseline*
system in the evaluation harness (this is what a classic pre-LLM document
pipeline looked like).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from ...config import settings
from ..ingest import IngestedDocument
from .base import ExtractedField, attach_location

logger = logging.getLogger(__name__)

_NLP_CACHE: dict = {}

# entity label -> candidate field names per doc type (first unfilled wins)
_ENTITY_MAP: Dict[str, Dict[str, List[str]]] = {
    "invoice": {
        "ORG": ["vendor_name", "client_name"],
        "DATE": ["invoice_date", "due_date"],
        "MONEY": ["total_amount", "subtotal", "tax"],
    },
    "contract": {
        "ORG": ["party_1", "party_2"],
        "PERSON": ["party_1", "party_2"],
        "DATE": ["effective_date", "termination_date"],
        "MONEY": ["contract_value"],
        "GPE": ["governing_law"],
    },
    "financial_statement": {
        "ORG": ["company_name"],
        "DATE": ["period"],
        "MONEY": ["total_revenue", "net_income", "total_assets"],
    },
    "court_document": {
        "ORG": ["plaintiff", "defendant"],
        "PERSON": ["plaintiff", "defendant", "judge"],
        "DATE": ["filing_date"],
    },
}
_ENTITY_MAP["receipt"] = _ENTITY_MAP["invoice"]


def _load_nlp():
    if "nlp" in _NLP_CACHE:
        return _NLP_CACHE["nlp"]
    try:
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model en_core_web_sm not downloaded")
            return None
        _NLP_CACHE["nlp"] = nlp
        return nlp
    except Exception:
        return None


class OcrSpacyExtractor:
    name = "ocr_spacy"

    def available(self) -> bool:
        return settings.enable_ocr and _load_nlp() is not None

    def extract(self, doc: IngestedDocument, doc_type: str) -> List[ExtractedField]:
        nlp = _load_nlp()
        if nlp is None:
            return []
        entity_map = _ENTITY_MAP.get(doc_type, _ENTITY_MAP["invoice"])
        text = doc.full_text[:20000]
        if not text.strip():
            return []

        parsed = nlp(text)
        filled: Dict[str, ExtractedField] = {}
        for ent in parsed.ents:
            candidates = entity_map.get(ent.label_)
            if not candidates:
                continue
            value = re.sub(r"\s+", " ", ent.text).strip()
            if len(value) < 2:
                continue
            for field_name in candidates:
                if field_name in filled:
                    continue
                # NER labels alone are weak evidence -> modest confidence
                conf = 0.55 if ent.label_ in ("MONEY", "DATE") else 0.45
                filled[field_name] = ExtractedField(
                    name=field_name, value=value, confidence=conf, source=self.name
                )
                break

        return [attach_location(doc, f) for f in filled.values()]
