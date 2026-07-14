"""Semantic search / RAG query endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import Document, get_db
from ..rag import qa
from ..schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/api/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def query_documents(req: QueryRequest, db: Session = Depends(get_db)):
    if req.document_id is not None and db.get(Document, req.document_id) is None:
        raise HTTPException(404, "Document not found")
    result = qa.answer(req.question, top_k=req.top_k, document_id=req.document_id)
    return QueryResponse(question=req.question, **result)
