"""Document type detection.

Primary strategy is a weighted-keyword scorer that is fast, deterministic and
dependency-free. When the transformers stack is installed, a zero-shot NLI
classifier is blended in for documents where the keyword scorer is uncertain.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

DOC_TYPES = [
    "invoice",
    "contract",
    "financial_statement",
    "court_document",
    "receipt",
    "other",
]

# keyword -> weight, per document type. Phrases are matched case-insensitively.
_KEYWORDS: Dict[str, Dict[str, float]] = {
    "invoice": {
        "invoice": 3.0, "invoice number": 4.0, "invoice no": 4.0, "bill to": 3.0,
        "due date": 2.0, "payment terms": 1.5, "subtotal": 2.0, "amount due": 3.0,
        "remit to": 2.0, "purchase order": 1.0, "net 30": 1.5, "net 15": 1.5,
        "tax id": 0.5, "quantity": 0.5, "unit price": 1.5,
    },
    "contract": {
        "agreement": 3.0, "this agreement": 4.0, "hereinafter": 3.0, "whereas": 3.0,
        "party": 1.0, "parties": 2.0, "termination": 2.0, "governing law": 3.0,
        "indemnif": 2.5, "confidentiality": 2.0, "effective date": 2.0,
        "in witness whereof": 4.0, "obligations": 1.5, "breach": 1.5,
        "services agreement": 3.0, "lease": 1.5, "term of this": 2.0,
    },
    "financial_statement": {
        "balance sheet": 4.0, "income statement": 4.0, "statement of operations": 4.0,
        "cash flow": 3.0, "total assets": 3.5, "total liabilities": 3.5,
        "shareholders' equity": 3.0, "stockholders": 2.5, "net income": 3.0,
        "revenue": 2.0, "fiscal year": 2.0, "retained earnings": 3.0,
        "operating expenses": 2.5, "cost of goods sold": 2.5, "ebitda": 2.0,
    },
    "court_document": {
        "plaintiff": 4.0, "defendant": 4.0, "case no": 4.0, "docket": 3.0,
        "court": 2.0, "united states district court": 4.0, "superior court": 3.5,
        "motion": 2.0, "complaint": 2.0, "hereby ordered": 3.0, "judge": 1.5,
        "civil action": 3.0, "vs.": 1.5, " v. ": 2.0, "summons": 3.0,
    },
    "receipt": {
        "receipt": 3.5, "cash": 1.0, "change due": 3.0, "cashier": 2.5,
        "thank you for your purchase": 3.0, "merchant": 1.5, "card ending": 2.0,
    },
}


def _keyword_scores(text: str) -> Dict[str, float]:
    lower = text.lower()
    scores: Dict[str, float] = {}
    for doc_type, kw in _KEYWORDS.items():
        score = 0.0
        for phrase, weight in kw.items():
            hits = len(re.findall(re.escape(phrase), lower))
            if hits:
                score += weight * min(hits, 3)  # cap repeat contribution
        scores[doc_type] = score
    return scores


def _zero_shot(text: str) -> Tuple[str, float] | None:
    """Optional transformer-based zero-shot classification (lazy, cached)."""
    try:
        from transformers import pipeline as hf_pipeline  # noqa: F401
    except Exception:
        return None
    global _ZS
    try:
        if "_ZS" not in globals() or _ZS is None:
            from transformers import pipeline as hf_pipeline
            _ZS = hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        labels = ["invoice", "legal contract", "financial statement",
                  "court filing", "purchase receipt", "other document"]
        out = _ZS(text[:2000], candidate_labels=labels)
        mapping = {
            "invoice": "invoice", "legal contract": "contract",
            "financial statement": "financial_statement",
            "court filing": "court_document", "purchase receipt": "receipt",
            "other document": "other",
        }
        return mapping[out["labels"][0]], float(out["scores"][0])
    except Exception as exc:  # pragma: no cover
        logger.warning("zero-shot classifier unavailable: %s", exc)
        return None


def classify(text: str) -> Tuple[str, float]:
    """Return (document_type, confidence in [0, 1])."""
    if not text.strip():
        return "other", 0.0

    scores = _keyword_scores(text)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_type, best = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

    if best <= 0:
        return "other", 0.3

    # Confidence from absolute evidence and separation from the runner-up
    margin = (best - runner_up) / best
    confidence = min(0.98, 0.45 + 0.35 * margin + min(best, 20) / 60)

    # If the keyword scorer is unsure, let a zero-shot model break the tie
    if confidence < 0.6:
        zs = _zero_shot(text)
        if zs and zs[1] > confidence:
            return zs
    return best_type, round(confidence, 3)
