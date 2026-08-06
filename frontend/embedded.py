"""In-process backend for standalone Streamlit deployment.

streamlit_app.py normally talks to the FastAPI service over HTTP. On a
platform that only runs a single process (e.g. Streamlit Community Cloud)
there is no separate API server, so this module reimplements the same
request/response shapes by calling straight into backend.app.* — same
pipeline, same database, same RAG store, just no network hop and no
background task queue (processing runs synchronously on upload).

streamlit_app.py falls back to this automatically when the configured API
URL is unreachable, so the same app works against a real backend (Docker /
local dev) or standalone (Streamlit Cloud) with no code change.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os  # noqa: E402

os.environ.setdefault("DOCINTEL_DATA_DIR", str(REPO_ROOT / "data"))
os.environ.setdefault(
    "DOCINTEL_DATABASE_URL", f"sqlite:///{REPO_ROOT / 'data' / 'docintel.db'}"
)

from backend.app.database import Document, SessionLocal, init_db  # noqa: E402
from backend.app.pipeline.hybrid import process_document  # noqa: E402
from backend.app.rag import qa as rag_qa  # noqa: E402
from backend.app.rag import store as rag_store  # noqa: E402

_initialized = False


def _ensure_db() -> None:
    global _initialized
    if not _initialized:
        init_db()
        _initialized = True


class FakeResponse:
    """Minimal stand-in for requests.Response, enough for streamlit_app.py."""

    def __init__(self, status_code: int, payload: Any = None,
                 content: bytes = b"", content_type: str = "application/json"):
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.headers = {"content-type": content_type}

    def json(self) -> Any:
        return self._payload


def _doc_out(d: Document) -> dict:
    return {
        "id": d.id, "filename": d.filename, "content_type": d.content_type,
        "status": d.status, "error": d.error, "doc_type": d.doc_type,
        "doc_type_confidence": d.doc_type_confidence, "num_pages": d.num_pages,
        "is_scanned": bool(d.is_scanned),
        "processing_seconds": d.processing_seconds,
        "created_at": d.created_at.isoformat() if d.created_at else None,
    }


def _upload(files: dict) -> FakeResponse:
    import uuid

    field = files["file"]
    filename, contents = (field[0], field[1]) if isinstance(field, tuple) else (
        getattr(field, "name", "upload"), field,
    )
    if hasattr(contents, "read"):
        contents = contents.read()

    suffix = Path(filename).suffix.lower()
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
    if suffix not in allowed:
        return FakeResponse(415, {"detail": f"Unsupported file type '{suffix}'"})
    if not contents:
        return FakeResponse(400, {"detail": "Empty file"})

    _ensure_db()
    doc_id = uuid.uuid4().hex
    upload_dir = REPO_ROOT / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored = upload_dir / f"{doc_id}{suffix}"
    stored.write_bytes(contents)

    db = SessionLocal()
    try:
        doc = Document(
            id=doc_id, filename=filename,
            content_type="application/octet-stream",
            stored_path=str(stored), status="processing",
        )
        db.add(doc)
        db.commit()

        try:
            result = process_document(str(stored))
        except Exception as exc:
            doc.status = "failed"
            doc.error = str(exc)[:2000]
            db.commit()
            return FakeResponse(202, {"id": doc_id, "filename": filename,
                                       "status": "queued"})

        pages_text = result.pop("_pages_text", [])
        try:
            chunks = rag_store.index_document(doc_id, filename, pages_text)
            result["meta"]["rag_chunks_indexed"] = chunks
        except Exception as exc:
            result["meta"]["rag_error"] = str(exc)[:500]

        doc.status = "completed"
        doc.doc_type = result["document_type"]
        doc.doc_type_confidence = result["document_type_confidence"]
        doc.num_pages = result["meta"]["num_pages"]
        doc.is_scanned = 1 if result["meta"]["is_scanned"] else 0
        doc.processing_seconds = result["meta"]["processing_seconds"]
        doc.result_json = json.dumps(result)
        db.commit()
    finally:
        db.close()

    return FakeResponse(202, {"id": doc_id, "filename": filename, "status": "queued"})


def _list_documents() -> FakeResponse:
    _ensure_db()
    db = SessionLocal()
    try:
        docs = db.query(Document).order_by(Document.created_at.desc()).all()
        return FakeResponse(200, [_doc_out(d) for d in docs])
    finally:
        db.close()


def _get_document(doc_id: str) -> FakeResponse:
    _ensure_db()
    db = SessionLocal()
    try:
        d = db.get(Document, doc_id)
        if d is None:
            return FakeResponse(404, {"detail": "Document not found"})
        return FakeResponse(200, _doc_out(d))
    finally:
        db.close()


def _get_result(doc_id: str) -> FakeResponse:
    _ensure_db()
    db = SessionLocal()
    try:
        d = db.get(Document, doc_id)
        if d is None:
            return FakeResponse(404, {"detail": "Document not found"})
        if d.status in ("queued", "processing"):
            return FakeResponse(409, {"detail": f"Document is still {d.status}"})
        if d.status == "failed":
            return FakeResponse(422, {"detail": f"Processing failed: {d.error}"})
        return FakeResponse(200, json.loads(d.result_json or "{}"))
    finally:
        db.close()


def _get_page(doc_id: str, page_number: int) -> FakeResponse:
    _ensure_db()
    db = SessionLocal()
    try:
        d = db.get(Document, doc_id)
        if d is None:
            return FakeResponse(404, {"detail": "Document not found"})
        path = d.stored_path
    finally:
        db.close()

    try:
        if path.lower().endswith(".pdf"):
            from backend.app.pipeline.ingest import render_pdf_page

            image = render_pdf_page(path, page_number, scale=2.0)
        else:
            from PIL import Image

            image = Image.open(path).convert("RGB")
    except Exception as exc:
        return FakeResponse(422, {"detail": f"Could not render page: {exc}"})

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return FakeResponse(200, content=buf.getvalue(), content_type="image/png")


def _query(payload: dict) -> FakeResponse:
    question = payload.get("question", "")
    document_id = payload.get("document_id")
    top_k = payload.get("top_k", 5)
    result = rag_qa.answer(question, top_k=top_k, document_id=document_id)
    return FakeResponse(200, {"question": question, **result})


def _dashboard_stats() -> FakeResponse:
    _ensure_db()
    from collections import Counter

    db = SessionLocal()
    try:
        docs = db.query(Document).all()
        by_status = Counter(d.status for d in docs)
        by_type = Counter(d.doc_type for d in docs if d.doc_type)

        proc_times = [d.processing_seconds for d in docs if d.processing_seconds]
        confidences: list = []
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
        return FakeResponse(200, {
            "total_documents": len(docs),
            "by_status": dict(by_status),
            "by_type": dict(by_type),
            "avg_processing_seconds": round(sum(proc_times) / len(proc_times), 2)
            if proc_times else None,
            "avg_field_confidence": round(sum(confidences) / len(confidences), 3)
            if confidences else None,
            "total_fields_extracted": total_fields,
            "recent": [_doc_out(d) for d in recent],
        })
    finally:
        db.close()


def _route(method: str, path: str, **kwargs) -> FakeResponse:
    parts = [p for p in path.split("/") if p]  # ["api", "documents", ...]

    if method == "POST" and path == "/api/documents/upload":
        return _upload(kwargs.get("files", {}))
    if method == "GET" and path == "/api/documents":
        return _list_documents()
    if method == "GET" and len(parts) == 3 and parts[:2] == ["api", "documents"]:
        return _get_document(parts[2])
    if method == "GET" and len(parts) == 4 and parts[:2] == ["api", "documents"] \
            and parts[3] == "result":
        return _get_result(parts[2])
    if method == "GET" and len(parts) == 5 and parts[:2] == ["api", "documents"] \
            and parts[3] == "page":
        return _get_page(parts[2], int(parts[4]))
    if method == "POST" and path == "/api/query":
        return _query(kwargs.get("json", {}))
    if method == "GET" and path == "/api/dashboard/stats":
        return _dashboard_stats()

    return FakeResponse(404, {"detail": f"Unhandled embedded route: {method} {path}"})


def dispatch(method: str, path: str, **kwargs) -> FakeResponse:
    """Route an api(method, path, **kwargs) call to a local handler.

    Any unexpected exception (e.g. a filesystem permission issue on a
    hosting platform, a corrupt row) becomes a visible 500 response instead
    of an uncaught exception — a failure should be diagnosable from the UI,
    never a silent blank screen.
    """
    method = method.upper()
    try:
        return _route(method, path, **kwargs)
    except Exception as exc:
        logger.exception("embedded dispatch failed: %s %s", method, path)
        return FakeResponse(500, {"detail": f"{type(exc).__name__}: {exc}"})
