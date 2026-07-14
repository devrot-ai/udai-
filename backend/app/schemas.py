"""Pydantic API schemas."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExtractedFieldOut(BaseModel):
    value: str
    confidence: float
    source: str
    page: Optional[int] = None
    bbox: Optional[List[float]] = None  # normalized [x0, y0, x1, y1]


class TableOut(BaseModel):
    page: int
    rows: List[List[Optional[str]]]
    bbox: Optional[List[float]] = None


class ExtractionResultOut(BaseModel):
    document_type: str
    document_type_confidence: float
    fields: Dict[str, ExtractedFieldOut]
    tables: List[TableOut] = []
    meta: Dict[str, Any] = {}


class DocumentOut(BaseModel):
    id: str
    filename: str
    content_type: str
    status: str
    error: Optional[str] = None
    doc_type: Optional[str] = None
    doc_type_confidence: Optional[float] = None
    num_pages: Optional[int] = None
    is_scanned: bool = False
    processing_seconds: Optional[float] = None
    created_at: Optional[dt.datetime] = None

    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    id: str
    filename: str
    status: str
    detail: str = "Document accepted; processing started."


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    document_id: Optional[str] = None
    top_k: int = Field(5, ge=1, le=20)


class QuerySource(BaseModel):
    document_id: str
    filename: Optional[str] = None
    page: Optional[int] = None
    snippet: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    answer_mode: str  # "llm" | "extractive"
    sources: List[QuerySource]


class DashboardStats(BaseModel):
    total_documents: int
    by_status: Dict[str, int]
    by_type: Dict[str, int]
    avg_processing_seconds: Optional[float]
    avg_field_confidence: Optional[float]
    total_fields_extracted: int
    recent: List[DocumentOut]
