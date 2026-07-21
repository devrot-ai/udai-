"""Supervised document-type classifier (TF-IDF + Logistic Regression).

Wraps the model artifact produced by training/train_classifier.py. The model
is loaded lazily and cached; when the artifact or scikit-learn is missing the
predictor reports unavailable and the classification ensemble falls back to
the keyword scorer (see pipeline/classifier.py).

Retrain with:  python training/train_classifier.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

MODEL_PATH = Path(
    os.environ.get("DOCINTEL_CLASSIFIER_MODEL", "models/doc_classifier.joblib")
)

_CACHE: dict = {}


def available() -> bool:
    if "model" in _CACHE:
        return _CACHE["model"] is not None
    if not MODEL_PATH.exists():
        _CACHE["model"] = None
        return False
    try:
        import joblib

        _CACHE["model"] = joblib.load(MODEL_PATH)
        logger.info("loaded supervised doc classifier from %s", MODEL_PATH)
        return True
    except Exception as exc:
        logger.warning("could not load doc classifier (%s); using fallback", exc)
        _CACHE["model"] = None
        return False


def predict(text: str) -> Optional[Tuple[str, float]]:
    """Return (label, probability) or None when the model is unavailable."""
    if not available():
        return None
    model = _CACHE["model"]
    try:
        probs = model.predict_proba([text[:20000]])[0]
        idx = probs.argmax()
        return str(model.classes_[idx]), float(probs[idx])
    except Exception as exc:  # pragma: no cover
        logger.warning("doc classifier prediction failed: %s", exc)
        return None
