"""Dashboard statistics endpoint."""

from __future__ import annotations

import json
from collections import Counter

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import Document, get_db
from ..schemas import DashboardStats, DocumentOut

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/stats", response_model=DashboardStats)
def stats(db: Session = Depends(get_db)):
    docs = db.query(Document).all()
    by_status = Counter(d.status for d in docs)
    by_type = Counter(d.doc_type for d in docs if d.doc_type)

    processing_times = [d.processing_seconds for d in docs if d.processing_seconds]
    confidences: list[float] = []
    total_fields = 0
    for d in docs:
        if not d.result_json:
            continue
        try:
            fields = json.loads(d.result_json).get("fields", {})
        except json.JSONDecodeError:
            continue
        total_fields += len(fields)
        confidences.extend(f["confidence"] for f in fields.values())

    recent = sorted(docs, key=lambda d: d.created_at or 0, reverse=True)[:10]
    from .documents import _to_out

    return DashboardStats(
        total_documents=len(docs),
        by_status=dict(by_status),
        by_type=dict(by_type),
        avg_processing_seconds=round(sum(processing_times) / len(processing_times), 2)
        if processing_times else None,
        avg_field_confidence=round(sum(confidences) / len(confidences), 3)
        if confidences else None,
        total_fields_extracted=total_fields,
        recent=[_to_out(d) for d in recent],
    )
