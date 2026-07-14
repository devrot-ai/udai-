"""Vector store for semantic search over uploaded documents.

Primary backend: ChromaDB (persistent) with sentence-transformer embeddings,
chunked via LangChain's RecursiveCharacterTextSplitter. When those packages
are missing (lite install), a dependency-free TF-IDF cosine store persisted
as JSON keeps semantic search functional so the whole API still works.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from ..config import settings

logger = logging.getLogger(__name__)


# ────────────────────────────── chunking ──────────────────────────────

def _split_text(text: str) -> List[str]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )
        return splitter.split_text(text)
    except ImportError:
        # simple recursive fallback with overlap
        size, overlap = settings.chunk_size, settings.chunk_overlap
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: List[str] = []
        buf = ""
        for para in paragraphs:
            if len(buf) + len(para) + 2 <= size:
                buf = f"{buf}\n\n{para}" if buf else para
            else:
                if buf:
                    chunks.append(buf)
                while len(para) > size:
                    chunks.append(para[:size])
                    para = para[size - overlap:]
                buf = para
        if buf:
            chunks.append(buf)
        return [c.strip() for c in chunks if c.strip()]


# ─────────────────────── TF-IDF fallback backend ───────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class _TfidfStore:
    """Tiny persistent TF-IDF cosine-similarity store (no dependencies)."""

    backend = "tfidf-fallback"

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._docs: List[dict] = []
        if path.exists():
            try:
                self._docs = json.loads(path.read_text())
            except Exception:
                self._docs = []

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._docs))

    def add(self, chunks: List[str], metadatas: List[dict], ids: List[str]) -> None:
        with self._lock:
            for chunk, meta, cid in zip(chunks, metadatas, ids):
                self._docs.append({"id": cid, "text": chunk, "meta": meta})
            self._persist()

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            self._docs = [d for d in self._docs if d["meta"].get("document_id") != document_id]
            self._persist()

    def query(self, question: str, top_k: int, document_id: Optional[str]) -> List[dict]:
        docs = [
            d for d in self._docs
            if document_id is None or d["meta"].get("document_id") == document_id
        ]
        if not docs:
            return []
        n = len(docs)
        df: Counter = Counter()
        tokenized = []
        for d in docs:
            toks = _tokenize(d["text"])
            tokenized.append(toks)
            df.update(set(toks))

        def vec(tokens: List[str]) -> Dict[str, float]:
            tf = Counter(tokens)
            return {
                t: (c / len(tokens)) * math.log((n + 1) / (1 + df.get(t, 0)))
                for t, c in tf.items()
            }

        qv = vec(_tokenize(question))
        results = []
        for d, toks in zip(docs, tokenized):
            dv = vec(toks)
            dot = sum(qv[t] * dv.get(t, 0.0) for t in qv)
            norm = math.sqrt(sum(v * v for v in qv.values())) * math.sqrt(
                sum(v * v for v in dv.values())
            )
            score = dot / norm if norm else 0.0
            results.append({"text": d["text"], "meta": d["meta"], "score": round(score, 4)})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]


# ───────────────────────── ChromaDB backend ─────────────────────────

class _ChromaStore:
    backend = "chromadb"

    def __init__(self, path: Path) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=str(path))
        embed_fn = None
        try:
            from chromadb.utils import embedding_functions

            embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model
            )
        except Exception:
            logger.info("sentence-transformers unavailable; using Chroma default embeddings")
        kwargs = {"embedding_function": embed_fn} if embed_fn else {}
        self._collection = self._client.get_or_create_collection("documents", **kwargs)

    def add(self, chunks: List[str], metadatas: List[dict], ids: List[str]) -> None:
        self._collection.add(documents=chunks, metadatas=metadatas, ids=ids)

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    def query(self, question: str, top_k: int, document_id: Optional[str]) -> List[dict]:
        where = {"document_id": document_id} if document_id else None
        res = self._collection.query(
            query_texts=[question], n_results=top_k, where=where
        )
        out = []
        for text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append({"text": text, "meta": meta, "score": round(1.0 / (1.0 + dist), 4)})
        return out


# ───────────────────────────── public API ─────────────────────────────

_STORE = None
_STORE_LOCK = threading.Lock()


def get_store():
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            chroma_dir = settings.data_dir / "chroma"
            try:
                _STORE = _ChromaStore(chroma_dir)
                logger.info("vector store backend: chromadb")
            except Exception as exc:
                logger.info("chromadb unavailable (%s); using TF-IDF fallback", exc)
                _STORE = _TfidfStore(settings.data_dir / "chroma" / "tfidf_store.json")
        return _STORE


def index_document(document_id: str, filename: str, pages_text: List[str]) -> int:
    """Chunk + embed a document's pages. Returns the number of chunks."""
    store = get_store()
    chunks: List[str] = []
    metas: List[dict] = []
    ids: List[str] = []
    for page_num, page_text in enumerate(pages_text, start=1):
        if not page_text.strip():
            continue
        for j, chunk in enumerate(_split_text(page_text)):
            chunks.append(chunk)
            metas.append({"document_id": document_id, "filename": filename, "page": page_num})
            ids.append(f"{document_id}:{page_num}:{j}")
    if chunks:
        store.add(chunks, metas, ids)
    return len(chunks)


def search(question: str, top_k: int = 5, document_id: Optional[str] = None) -> List[dict]:
    return get_store().query(question, top_k, document_id)


def remove_document(document_id: str) -> None:
    try:
        get_store().delete_document(document_id)
    except Exception as exc:  # pragma: no cover
        logger.warning("failed to remove %s from vector store: %s", document_id, exc)
