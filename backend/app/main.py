"""FastAPI application entry point.

Run with:  uvicorn backend.app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .api import dashboard, documents, query
from .database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Document Intelligence System",
    description=(
        "Multi-modal AI pipeline for extracting structured information from "
        "legal and financial documents. Hybrid LayoutLMv3 + Donut + "
        "Tesseract/spaCy extraction with ChromaDB-backed semantic search."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(query.router)
app.include_router(dashboard.router)


@app.get("/health", tags=["system"])
def health():
    from .ml import doc_classifier
    from .pipeline.hybrid import _EXTRACTORS
    from .rag.store import get_store

    return {
        "status": "ok",
        "version": __version__,
        "extractors": {e.name: e.available() for e in _EXTRACTORS},
        "ml_classifier": doc_classifier.available(),
        "vector_backend": get_store().backend,
    }
