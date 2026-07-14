"""RAG query interface: retrieve relevant chunks, then answer.

Answer synthesis is pluggable:
  * "llm"        — when ANTHROPIC_API_KEY is set and the anthropic SDK is
                   installed, retrieved chunks are handed to an LLM that
                   answers strictly from the provided context.
  * "extractive" — otherwise, the most relevant sentences from the top
                   chunks are returned directly. Zero-dependency default.
"""

from __future__ import annotations

import logging
import os
import re
from typing import List, Optional

from ..config import settings
from . import store

logger = logging.getLogger(__name__)


def _extractive_answer(question: str, hits: List[dict]) -> str:
    """Pick the sentences from the retrieved chunks that best overlap the
    question's content words."""
    def stem(t: str) -> str:
        # crude stemmer so "termination" matches "terminated"
        for suffix in ("ations", "ation", "ments", "ment", "ings", "ing",
                       "ities", "ity", "ies", "ed", "es", "s"):
            if t.endswith(suffix) and len(t) - len(suffix) >= 4:
                t = t[: len(t) - len(suffix)]
                break
        return t[:6]  # prefix-truncate so "termin(ation)" == "terminat(ed)"

    q_terms = {
        stem(t) for t in re.findall(r"[a-z0-9]{3,}", question.lower())
        if t not in {"what", "who", "when", "which", "where", "how", "the",
                     "are", "does", "this", "that", "have", "was", "were"}
    }
    scored: List[tuple] = []
    for hit in hits[:3]:
        for sent in re.split(r"(?<=[.!?])\s+|\n", hit["text"]):
            sent = sent.strip()
            if len(sent) < 15:
                continue
            terms = {stem(t) for t in re.findall(r"[a-z0-9]{3,}", sent.lower())}
            overlap = len(q_terms & terms)
            if overlap:
                scored.append((overlap + hit["score"], sent))
    scored.sort(key=lambda t: t[0], reverse=True)
    if not scored:
        return "No relevant passage found in the indexed documents."
    top: List[str] = []
    for _, sent in scored:
        # chunk overlap yields truncated near-duplicates — drop any sentence
        # that substantially overlaps one already selected
        if any(sent[5:] in kept or kept[5:] in sent for kept in top):
            continue
        top.append(sent)
        if len(top) == 3:
            break
    return " ".join(top)


def _llm_answer(question: str, hits: List[dict]) -> Optional[str]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    try:
        context = "\n\n---\n\n".join(
            f"[{h['meta'].get('filename', '?')} p.{h['meta'].get('page', '?')}]\n{h['text']}"
            for h in hits
        )
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=1024,
            system=(
                "You answer questions about legal and financial documents using "
                "ONLY the provided excerpts. Quote exact values (amounts, dates, "
                "clause language) where possible and cite the source file and "
                "page. If the excerpts do not contain the answer, say so."
            ),
            messages=[{
                "role": "user",
                "content": f"Excerpts:\n\n{context}\n\nQuestion: {question}",
            }],
        )
        parts = [b.text for b in response.content if b.type == "text"]
        return "".join(parts).strip() or None
    except Exception as exc:  # pragma: no cover
        logger.warning("LLM answer synthesis failed, falling back: %s", exc)
        return None


def answer(question: str, top_k: int = 5, document_id: Optional[str] = None) -> dict:
    hits = store.search(question, top_k=top_k, document_id=document_id)
    if not hits:
        return {
            "answer": "No documents have been indexed yet, or nothing matched.",
            "answer_mode": "extractive",
            "sources": [],
        }
    llm = _llm_answer(question, hits)
    if llm is not None:
        mode, text = "llm", llm
    else:
        mode, text = "extractive", _extractive_answer(question, hits)
    return {
        "answer": text,
        "answer_mode": mode,
        "sources": [
            {
                "document_id": h["meta"].get("document_id", ""),
                "filename": h["meta"].get("filename"),
                "page": h["meta"].get("page"),
                "snippet": h["text"][:300],
                "score": h["score"],
            }
            for h in hits
        ],
    }
