"""Asynchronous document processing job.

Runs as a FastAPI BackgroundTask after upload: executes the hybrid pipeline,
indexes text in the vector store, and persists the result. Failures are
recorded on the document row rather than propagated.
"""

from __future__ import annotations

import json
import logging

from ..database import Document, SessionLocal
from ..pipeline.hybrid import process_document
from ..rag import store as rag_store

logger = logging.getLogger(__name__)


def run(document_id: str) -> None:
    db = SessionLocal()
    try:
        doc = db.get(Document, document_id)
        if doc is None:
            logger.error("document %s vanished before processing", document_id)
            return
        doc.status = "processing"
        db.commit()

        try:
            result = process_document(doc.stored_path)
        except Exception as exc:
            logger.exception("processing failed for %s", document_id)
            doc.status = "failed"
            doc.error = str(exc)[:2000]
            db.commit()
            return

        pages_text = result.pop("_pages_text", [])
        try:
            chunks = rag_store.index_document(document_id, doc.filename, pages_text)
            result["meta"]["rag_chunks_indexed"] = chunks
        except Exception as exc:  # RAG failure shouldn't fail the extraction
            logger.exception("RAG indexing failed for %s", document_id)
            result["meta"]["rag_error"] = str(exc)[:500]

        doc.status = "completed"
        doc.doc_type = result["document_type"]
        doc.doc_type_confidence = result["document_type_confidence"]
        doc.num_pages = result["meta"]["num_pages"]
        doc.is_scanned = 1 if result["meta"]["is_scanned"] else 0
        doc.processing_seconds = result["meta"]["processing_seconds"]
        doc.result_json = json.dumps(result)
        db.commit()
        logger.info(
            "processed %s: type=%s fields=%d",
            document_id, doc.doc_type, len(result["fields"]),
        )
    finally:
        db.close()
