"""RAG query interface: retrieve relevant chunks, then answer.

Answer synthesis has three tiers, tried in order:
  * field lookup — when the question is scoped to one document and matches
                   a well-known intent ("who are the parties", "total
                   amount due", ...), answer directly from that document's
                   already-extracted structured fields. Keyword-overlap
                   search over free text can match a sentence that merely
                   mentions the right word ("the parties agree as follows")
                   without naming anyone — the structured field is the
                   pipeline's actual, verified answer to that question.
  * "llm"        — when ANTHROPIC_API_KEY is set and the anthropic SDK is
                   installed, retrieved chunks are handed to an LLM that
                   answers strictly from the provided context.
  * "extractive" — otherwise, the most relevant sentence from the top
                   chunks is returned directly. Zero-dependency default.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import List, Optional

from ..config import settings
from . import store

logger = logging.getLogger(__name__)

# question keywords -> candidate field names. Checked in order, most
# specific phrases first — bare words like "total" are avoided as triggers
# since they collide across many fields ("total revenue", "total assets",
# "total amount due" all contain it); multi-word phrases disambiguate.
_FIELD_INTENTS: List[tuple] = [
    (["total revenue", "revenue"], ["total_revenue"]),
    (["net income", "profit"], ["net_income"]),
    (["total assets"], ["total_assets"]),
    (["total liabilities"], ["total_liabilities"]),
    (["amount due", "balance due", "total due", "how much"],
     ["total_amount", "contract_value"]),
    (["payment term", "terms of payment"], ["payment_terms"]),
    (["due date"], ["due_date"]),
    (["effective date", "start date", "commence"], ["effective_date"]),
    (["governing law", "which state", "jurisdiction"], ["governing_law"]),
    (["case number", "case no", "docket"], ["case_number"]),
    (["judge"], ["judge"]),
    (["who", "part", "plaintiff", "defendant"],
     ["party_1", "party_2", "plaintiff", "defendant", "vendor_name", "client_name"]),
]


def _field_shortcut(question: str, document_id: Optional[str]) -> Optional[dict]:
    if not document_id:
        return None
    try:
        from ..database import Document, SessionLocal

        db = SessionLocal()
        try:
            doc = db.get(Document, document_id)
            if doc is None or not doc.result_json:
                return None
            fields = json.loads(doc.result_json).get("fields", {})
            filename = doc.filename
        finally:
            db.close()
    except Exception as exc:  # pragma: no cover
        logger.warning("field shortcut lookup failed: %s", exc)
        return None

    q_lower = question.lower()
    for keywords, field_names in _FIELD_INTENTS:
        if not any(kw in q_lower for kw in keywords):
            continue
        matched = [(name, fields[name]) for name in field_names if name in fields]
        if not matched:
            continue

        def humanize(s: str) -> str:
            return s.replace("_", " ").title()

        if len(matched) == 1:
            name, f = matched[0]
            text = f"{humanize(name)}: {f['value']}"
        else:
            text = "; ".join(f"{humanize(n)}: {f['value']}" for n, f in matched)

        return {
            "answer": text,
            "answer_mode": "extractive",
            "sources": [{
                "document_id": document_id,
                "filename": filename,
                "page": f.get("page"),
                "snippet": f"{humanize(n)}: {f['value']}",
                "score": 1.0,
            } for n, f in matched],
        }
    return None


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
            # section headings ("3. TERM AND TERMINATION") aren't answers on
            # their own — skip fragments with no lowercase letters
            if not re.search(r"[a-z]", sent):
                continue
            terms = {stem(t) for t in re.findall(r"[a-z0-9]{3,}", sent.lower())}
            overlap = len(q_terms & terms)
            if overlap:
                scored.append((overlap + hit["score"], sent, hit["text"]))
    scored.sort(key=lambda t: t[0], reverse=True)
    if not scored:
        return "No relevant passage found in the indexed documents."

    # Concatenating several independently top-scoring sentences tends to
    # glue unrelated facts together (e.g. a total, a due date, and a table
    # header row can all individually overlap a question's keywords). One
    # precise sentence beats three garbled ones; the Sources list below
    # carries the surrounding context for anyone who wants more.
    best_score, best_sent, best_source = scored[0]
    top = [best_sent]

    # A close second sentence that reads as a natural continuation of the
    # first (same source chunk, comparably strong score) is still worth
    # keeping — e.g. a clause split across two sentences.
    if len(scored) > 1:
        second_score, second_sent, second_source = scored[1]
        same_chunk = second_source == best_source
        if same_chunk and second_score >= best_score * 0.85 and second_sent not in best_sent:
            top.append(second_sent)

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
    shortcut = _field_shortcut(question, document_id)
    if shortcut is not None:
        return shortcut

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
