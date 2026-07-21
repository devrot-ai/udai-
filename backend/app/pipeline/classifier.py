"""Document type detection — a three-tier ensemble.

Tier 1 (supervised ML): TF-IDF + Logistic Regression trained on a randomized
        synthetic corpus (training/train_classifier.py). Calibrated
        probabilities via predict_proba.
Tier 2 (heuristic):     weighted-keyword scorer — deterministic,
        dependency-free, always available.
Tier 3 (zero-shot):     bart-large-mnli NLI classifier, consulted only when
        the first two tiers are jointly uncertain and transformers is
        installed.

When tiers 1 and 2 agree, confidence is boosted; when they disagree, the more
confident tier wins with a capped confidence. The deciding method is exposed
via classify_detailed() and recorded in extraction metadata.
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


def _keyword_classify(text: str) -> Tuple[str, float]:
    scores = _keyword_scores(text)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_type, best = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

    if best <= 0:
        return "other", 0.3

    # Confidence from absolute evidence and separation from the runner-up
    margin = (best - runner_up) / best
    confidence = min(0.98, 0.45 + 0.35 * margin + min(best, 20) / 60)
    return best_type, round(confidence, 3)


def classify_detailed(text: str) -> dict:
    """Return {"type", "confidence", "method"} from the ensemble."""
    if not text.strip():
        return {"type": "other", "confidence": 0.0, "method": "empty"}

    from ..ml import doc_classifier

    ml = doc_classifier.predict(text)
    kw_type, kw_conf = _keyword_classify(text)

    if ml is not None:
        ml_type, ml_conf = ml
        if ml_type == kw_type:
            # independent agreement is strong evidence
            conf = min(0.99, max(ml_conf, kw_conf) + 0.05)
            return {"type": ml_type, "confidence": round(conf, 3),
                    "method": "ml+keywords"}
        # disagreement: trust the more confident tier, but cap the score
        if ml_conf >= kw_conf:
            return {"type": ml_type, "confidence": round(min(ml_conf, 0.85), 3),
                    "method": "ml"}
        return {"type": kw_type, "confidence": round(min(kw_conf, 0.85), 3),
                "method": "keywords"}

    # no trained model available — keyword scorer, zero-shot as tie-breaker
    if kw_conf < 0.6:
        zs = _zero_shot(text)
        if zs and zs[1] > kw_conf:
            return {"type": zs[0], "confidence": round(zs[1], 3),
                    "method": "zero-shot"}
    return {"type": kw_type, "confidence": kw_conf, "method": "keywords"}


def classify(text: str) -> Tuple[str, float]:
    """Return (document_type, confidence in [0, 1])."""
    d = classify_detailed(text)
    return d["type"], d["confidence"]
