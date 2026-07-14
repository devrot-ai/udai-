"""LayoutLMv3 extractor — layout-aware key/value extraction.

Uses a FUNSD-fine-tuned LayoutLMv3 checkpoint for token classification into
QUESTION / ANSWER / HEADER entities, pairs each question with the nearest
answer entity, then maps the question phrasing onto the canonical field
schema. Confidence is the mean softmax probability of the answer tokens.

The model is loaded lazily on first use and cached process-wide. If the
transformers/torch stack is not installed the extractor reports itself
unavailable and the hybrid pipeline falls through to the other extractors.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from ...config import settings
from ..ingest import IngestedDocument, render_pdf_page
from .base import ExtractedField

logger = logging.getLogger(__name__)

# question-phrase → canonical field name, per document type
_KEY_MAP: Dict[str, Dict[str, str]] = {
    "invoice": {
        "invoice number": "invoice_number", "invoice no": "invoice_number",
        "invoice #": "invoice_number", "invoice date": "invoice_date",
        "date": "invoice_date", "due date": "due_date", "bill to": "client_name",
        "billed to": "client_name", "customer": "client_name", "from": "vendor_name",
        "vendor": "vendor_name", "subtotal": "subtotal", "tax": "tax",
        "total": "total_amount", "total due": "total_amount",
        "amount due": "total_amount", "balance due": "total_amount",
        "payment terms": "payment_terms", "terms": "payment_terms",
    },
    "contract": {
        "effective date": "effective_date", "date": "effective_date",
        "party": "party_1", "between": "party_1", "governing law": "governing_law",
        "term": "term", "payment terms": "payment_terms",
    },
    "financial_statement": {
        "total revenue": "total_revenue", "revenue": "total_revenue",
        "net income": "net_income", "total assets": "total_assets",
        "total liabilities": "total_liabilities", "period": "period",
    },
    "court_document": {
        "case no": "case_number", "case number": "case_number",
        "plaintiff": "plaintiff", "defendant": "defendant",
        "filed": "filing_date", "judge": "judge",
    },
}
_KEY_MAP["receipt"] = _KEY_MAP["invoice"]

_MODEL_CACHE: dict = {}


class LayoutLMExtractor:
    name = "layoutlmv3"

    def __init__(self) -> None:
        self._checked: Optional[bool] = None

    def available(self) -> bool:
        if not settings.enable_layoutlm:
            return False
        if self._checked is None:
            try:
                import torch  # noqa: F401
                import transformers  # noqa: F401

                self._checked = True
            except Exception:
                self._checked = False
        return self._checked

    def _load(self):
        if "model" not in _MODEL_CACHE:
            from transformers import (AutoModelForTokenClassification,
                                      AutoProcessor)

            logger.info("Loading LayoutLMv3 checkpoint %s", settings.layoutlm_model)
            _MODEL_CACHE["processor"] = AutoProcessor.from_pretrained(
                settings.layoutlm_model, apply_ocr=False
            )
            _MODEL_CACHE["model"] = AutoModelForTokenClassification.from_pretrained(
                settings.layoutlm_model
            )
        return _MODEL_CACHE["processor"], _MODEL_CACHE["model"]

    def _page_image(self, doc: IngestedDocument, page) -> Optional[object]:
        if page.image is not None:
            return page.image
        if doc.path.lower().endswith(".pdf"):
            try:
                return render_pdf_page(doc.path, page.number, scale=1.5)
            except Exception as exc:  # pragma: no cover
                logger.warning("failed to render page %s: %s", page.number, exc)
        return None

    def extract(self, doc: IngestedDocument, doc_type: str) -> List[ExtractedField]:
        if not self.available():
            return []
        try:
            return self._extract(doc, doc_type)
        except Exception as exc:  # pragma: no cover - inference failure
            logger.exception("LayoutLMv3 extraction failed: %s", exc)
            return []

    def _extract(self, doc: IngestedDocument, doc_type: str) -> List[ExtractedField]:
        import torch

        processor, model = self._load()
        id2label = model.config.id2label
        key_map = _KEY_MAP.get(doc_type, _KEY_MAP["invoice"])
        results: List[ExtractedField] = []

        for page in doc.pages[:3]:  # entity-dense pages come first
            if not page.words:
                continue
            image = self._page_image(doc, page)
            if image is None:
                continue

            words = [w.text for w in page.words][:512]
            boxes = [
                [
                    int(w.x0 * 1000), int(w.y0 * 1000),
                    int(w.x1 * 1000), int(w.y1 * 1000),
                ]
                for w in page.words
            ][:512]

            encoding = processor(
                image, words, boxes=boxes, truncation=True,
                return_tensors="pt", return_offsets_mapping=False,
            )
            with torch.no_grad():
                logits = model(**encoding).logits[0]
            probs = torch.softmax(logits, dim=-1)
            pred_ids = probs.argmax(-1).tolist()
            pred_conf = probs.max(-1).values.tolist()
            word_ids = encoding.word_ids(0)

            # collapse subword predictions to word level (first subtoken wins)
            word_pred: Dict[int, Tuple[str, float]] = {}
            for tok_idx, wid in enumerate(word_ids):
                if wid is None or wid in word_pred:
                    continue
                word_pred[wid] = (id2label[pred_ids[tok_idx]], pred_conf[tok_idx])

            # group consecutive same-label words into entities
            entities: List[dict] = []
            current: Optional[dict] = None
            for wid in sorted(word_pred):
                label, conf = word_pred[wid]
                base = label.split("-")[-1].upper()  # B-QUESTION -> QUESTION
                if base in ("QUESTION", "ANSWER") and current and current["label"] == base \
                        and label.startswith("I-"):
                    current["word_ids"].append(wid)
                    current["confs"].append(conf)
                    continue
                if current:
                    entities.append(current)
                    current = None
                if base in ("QUESTION", "ANSWER"):
                    current = {"label": base, "word_ids": [wid], "confs": [conf]}
            if current:
                entities.append(current)

            def entity_text(e) -> str:
                return " ".join(page.words[i].text for i in e["word_ids"] if i < len(page.words))

            def entity_box(e) -> List[float]:
                ws = [page.words[i] for i in e["word_ids"] if i < len(page.words)]
                return [min(w.x0 for w in ws), min(w.y0 for w in ws),
                        max(w.x1 for w in ws), max(w.y1 for w in ws)]

            # pair each QUESTION with the nearest following ANSWER
            answers = [e for e in entities if e["label"] == "ANSWER"]
            for q in (e for e in entities if e["label"] == "QUESTION"):
                q_text = re.sub(r"[:#]", "", entity_text(q)).strip().lower()
                field_name = None
                for phrase, canonical in key_map.items():
                    if phrase in q_text:
                        field_name = canonical
                        break
                if not field_name:
                    continue
                q_end = q["word_ids"][-1]
                following = [a for a in answers if a["word_ids"][0] > q_end]
                if not following:
                    continue
                ans = min(following, key=lambda a: a["word_ids"][0] - q_end)
                value = entity_text(ans).strip()
                if not value:
                    continue
                conf = float(sum(ans["confs"]) / len(ans["confs"]))
                results.append(
                    ExtractedField(
                        name=field_name, value=value,
                        confidence=round(conf, 3), source=self.name,
                        page=page.number, bbox=entity_box(ans),
                    )
                )
        return results
