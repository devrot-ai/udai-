"""Rules-based extractor: regex + layout heuristics over the text layer.

This is the deterministic backbone of the hybrid pipeline. It is always
available (no ML dependencies), understands the domain schema per document
type, and assigns pattern-strength-based confidence scores. The neural
extractors (LayoutLMv3 / Donut) can outvote it in the merge step when they
are more confident.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

from ..ingest import IngestedDocument
from .base import ExtractedField, attach_location

DATE = r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})"
MONEY = r"(?:USD\s*)?\$?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?"
COMPANY = r"[A-Z][A-Za-z0-9&.,'\- ]{2,60}?(?:Inc\.?|LLC|L\.L\.C\.|Ltd\.?|LLP|Corp\.?|Corporation|Company|Co\.|GmbH|Group|Partners|Holdings|Bank|Services|Solutions|Technologies)"
NAME_LINE = r"[A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){1,3}"


def _search(patterns: List[str], text: str, group: int = 1,
            flags: int = re.IGNORECASE) -> Optional[re.Match]:
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            return m
    return None


def _clean_money(v: str) -> str:
    v = v.strip()
    v = re.sub(r"^(USD)\s*", "", v, flags=re.IGNORECASE)
    v = v.replace("$ ", "$").strip()
    if not v.startswith("$") and re.match(r"^\d", v):
        v = "$" + v
    return v


def _field(name: str, value: str, conf: float) -> ExtractedField:
    return ExtractedField(name=name, value=value.strip().rstrip(".,;"),
                          confidence=conf, source="rules")


def _strip_party_prefix(v: str) -> str:
    """Drop a leading 'v.'/'vs.'/'versus' that a caption line (e.g. "v. REDWOOD
    RETAIL CORP") can pull into a plaintiff/defendant match when it shares a
    text line with the party name."""
    return re.sub(r"^(?:v\.?|vs\.?|versus)\s+", "", v.strip(), flags=re.IGNORECASE)


# ─────────────────────────── per-type extractors ───────────────────────────

def _extract_invoice(text: str) -> List[ExtractedField]:
    out: List[ExtractedField] = []

    m = _search([r"invoice\s*(?:number|no\.?|#)\s*[:#]?\s*([A-Z0-9][A-Z0-9\-/]{2,20})"], text)
    if m:
        out.append(_field("invoice_number", m.group(1), 0.95))

    m = _search([rf"invoice\s*date\s*[:\-]?\s*({DATE})", rf"date\s*of\s*issue\s*[:\-]?\s*({DATE})",
                 rf"(?<!due )date\s*[:\-]\s*({DATE})"], text)
    if m:
        out.append(_field("invoice_date", m.group(1), 0.9))

    m = _search([rf"due\s*date\s*[:\-]?\s*({DATE})", rf"payment\s*due\s*[:\-]?\s*({DATE})"], text)
    if m:
        out.append(_field("due_date", m.group(1), 0.9))

    m = _search([rf"(?:from|vendor|seller|issued\s+by)\s*[:\-]?\s*({COMPANY})",
                 rf"^\s*({COMPANY})"], text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        out.append(_field("vendor_name", m.group(1), 0.75))

    m = _search([rf"bill(?:ed)?\s*to\s*[:\-]?\s*\n?\s*({COMPANY}|{NAME_LINE})",
                 rf"customer\s*[:\-]?\s*({COMPANY}|{NAME_LINE})"], text)
    if m:
        out.append(_field("client_name", m.group(1), 0.85))

    m = _search([rf"subtotal\s*[:\-]?\s*({MONEY})"], text)
    if m:
        out.append(_field("subtotal", _clean_money(m.group(1)), 0.9))

    m = _search([rf"(?:sales\s*)?tax(?:\s*\(\s*[\d.]+%\s*\))?\s*[:\-]?\s*({MONEY})",
                 rf"vat\s*[:\-]?\s*({MONEY})"], text)
    if m:
        out.append(_field("tax", _clean_money(m.group(1)), 0.85))

    m = _search([rf"(?:total\s*(?:amount\s*)?due|amount\s*due|grand\s*total|balance\s*due)\s*[:\-]?\s*({MONEY})",
                 rf"total\s*[:\-]?\s*({MONEY})"], text)
    if m:
        out.append(_field("total_amount", _clean_money(m.group(1)), 0.92))

    m = _search([r"payment\s*terms?\s*[:\-]?\s*(Net\s*\d{1,3}|Due\s+on\s+receipt|[A-Za-z0-9 ,]{3,60}?)(?:\n|$)",
                 r"terms?\s*[:\-]\s*(Net\s*\d{1,3})"], text)
    if m:
        out.append(_field("payment_terms", m.group(1), 0.8))

    return out


def _extract_contract(text: str) -> List[ExtractedField]:
    out: List[ExtractedField] = []

    m = re.search(
        rf"between\s+({COMPANY}|{NAME_LINE})\s*(?:\([^)]*\))?\s*,?\s*(?:and)\s+({COMPANY}|{NAME_LINE})",
        text, re.IGNORECASE | re.DOTALL)
    if m:
        out.append(_field("party_1", m.group(1), 0.9))
        out.append(_field("party_2", m.group(2), 0.9))

    m = _search([rf"effective\s*(?:as\s*of|date)\s*[:\-]?\s*(?:the\s*)?({DATE})",
                 rf"dated\s+(?:as\s+of\s+)?({DATE})",
                 rf"entered\s+into\s+(?:on|as\s+of)\s+({DATE})"], text)
    if m:
        out.append(_field("effective_date", m.group(1), 0.88))

    m = _search([rf"(?:terminat\w+|expir\w+)\s+(?:on|date[:\-]?)\s*({DATE})",
                 rf"until\s+({DATE})"], text)
    if m:
        out.append(_field("termination_date", m.group(1), 0.8))

    m = _search([r"governed\s+by\s+(?:and\s+construed\s+in\s+accordance\s+with\s+)?the\s+laws?\s+of\s+(?:the\s+)?(?:state\s+of\s+)?([A-Z][A-Za-z ]{2,30}?)(?:[.,\n]|without)",
                 r"governing\s+law\s*[:\-]?\s*([A-Z][A-Za-z ]{2,30})"], text)
    if m:
        out.append(_field("governing_law", m.group(1), 0.9))

    m = _search([rf"(?:total\s+)?(?:contract\s+value|consideration|fee)\s+of\s+({MONEY})",
                 rf"(?:shall|will)\s+pay\s+[^.\n]{{0,40}}?({MONEY})",
                 rf"amount\s+of\s+({MONEY})"], text)
    if m:
        out.append(_field("contract_value", _clean_money(m.group(1)), 0.75))

    m = _search([r"payment\s*terms?\s*[:\-]?\s*([^\n.]{3,80})",
                 r"payable\s+(within\s+[^\n.]{3,60})"], text)
    if m:
        out.append(_field("payment_terms", m.group(1), 0.7))

    m = _search([rf"(?:initial\s+)?term\s+of\s+((?:\w+\s*)?\(?\d+\)?\s*(?:years?|months?))"], text)
    if m:
        out.append(_field("term", m.group(1), 0.75))

    return out


def _extract_financial(text: str) -> List[ExtractedField]:
    out: List[ExtractedField] = []

    m = _search([rf"^\s*({COMPANY})", rf"({COMPANY})\s*\n\s*(?:consolidated\s+)?(?:balance\s+sheet|income\s+statement|statement)"],
                text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        out.append(_field("company_name", m.group(1), 0.8))

    m = _search([rf"(?:for\s+the\s+(?:year|quarter|period)\s+end(?:ed|ing)\s+)({DATE})",
                 rf"as\s+of\s+({DATE})",
                 r"fiscal\s+year\s+(\d{4})"], text)
    if m:
        out.append(_field("period", m.group(1), 0.85))

    def money_after(labels: List[str], name: str, conf: float) -> None:
        m = _search([rf"{lab}\s*[:\-]?\s*\$?\s*({MONEY})" for lab in labels], text)
        if m:
            out.append(_field(name, _clean_money(m.group(1)), conf))

    money_after([r"total\s+revenue[s]?", r"net\s+revenue[s]?", r"total\s+net\s+sales"], "total_revenue", 0.9)
    money_after([r"net\s+income", r"net\s+earnings", r"net\s+profit"], "net_income", 0.9)
    money_after([r"total\s+assets"], "total_assets", 0.92)
    money_after([r"total\s+liabilities(?:\s+and\s+equity)?"], "total_liabilities", 0.88)
    money_after([r"(?:total\s+)?(?:share|stock)holders'?\s+equity"], "shareholders_equity", 0.85)
    money_after([r"operating\s+expenses"], "operating_expenses", 0.82)

    return out


def _extract_court(text: str) -> List[ExtractedField]:
    out: List[ExtractedField] = []

    m = _search([r"case\s*(?:no\.?|number)\s*[:#]?\s*([A-Z0-9][A-Za-z0-9:\-cv]{3,25})",
                 r"civil\s+action\s+no\.?\s*[:#]?\s*([A-Z0-9][A-Za-z0-9:\-cv]{3,25})",
                 r"docket\s*(?:no\.?|number)\s*[:#]?\s*([A-Z0-9][A-Za-z0-9:\-]{3,25})"], text)
    if m:
        out.append(_field("case_number", m.group(1), 0.95))

    m = _search([r"((?:UNITED STATES DISTRICT COURT|SUPERIOR COURT|SUPREME COURT|CIRCUIT COURT|COURT OF APPEALS)"
                 r"[^\n]{0,60}(?:\n\s*DISTRICT OF [^\n]{0,40})?)"],
                text, flags=0)
    if m:
        out.append(_field("court_name", re.sub(r"\s+", " ", m.group(1)), 0.9))

    m = re.search(rf"(?:^|\n)\s*({COMPANY}|{NAME_LINE})\s*,?\s*\n?\s*Plaintiffs?", text, re.IGNORECASE)
    if m:
        out.append(_field("plaintiff", _strip_party_prefix(m.group(1)), 0.88))
    else:
        m = re.search(rf"Plaintiffs?\s*[:,\-]?\s*({COMPANY}|{NAME_LINE})", text, re.IGNORECASE)
        if m:
            out.append(_field("plaintiff", _strip_party_prefix(m.group(1)), 0.8))

    m = re.search(rf"(?:^|\n)\s*({COMPANY}|{NAME_LINE})\s*,?\s*\n?\s*Defendants?", text, re.IGNORECASE)
    if m:
        out.append(_field("defendant", _strip_party_prefix(m.group(1)), 0.88))
    else:
        m = re.search(rf"Defendants?\s*[:,\-]?\s*({COMPANY}|{NAME_LINE})", text, re.IGNORECASE)
        if m:
            out.append(_field("defendant", _strip_party_prefix(m.group(1)), 0.8))

    m = _search([rf"filed\s*(?:on)?\s*[:\-]?\s*({DATE})", rf"date\s*filed\s*[:\-]?\s*({DATE})",
                 rf"dated\s*[:\-]?\s*({DATE})"], text)
    if m:
        out.append(_field("filing_date", m.group(1), 0.85))

    m = _search([rf"(?:honorable|hon\.?|judge)\s+({NAME_LINE})"], text)
    if m:
        out.append(_field("judge", m.group(1), 0.85))

    return out


def _extract_generic(text: str) -> List[ExtractedField]:
    out: List[ExtractedField] = []
    m = _search([rf"({DATE})"], text)
    if m:
        out.append(_field("date", m.group(1), 0.5))
    m = _search([rf"total\s*[:\-]?\s*({MONEY})"], text)
    if m:
        out.append(_field("total_amount", _clean_money(m.group(1)), 0.5))
    return out


_DISPATCH: Dict[str, Callable[[str], List[ExtractedField]]] = {
    "invoice": _extract_invoice,
    "receipt": _extract_invoice,
    "contract": _extract_contract,
    "financial_statement": _extract_financial,
    "court_document": _extract_court,
    "other": _extract_generic,
}


class RulesExtractor:
    name = "rules"

    def available(self) -> bool:
        return True

    def extract(self, doc: IngestedDocument, doc_type: str) -> List[ExtractedField]:
        fn = _DISPATCH.get(doc_type, _extract_generic)
        fields = fn(doc.full_text)
        # scanned/OCR text is noisier -> discount confidence
        if doc.is_scanned:
            for f in fields:
                f.confidence = round(f.confidence * 0.85, 3)
        return [attach_location(doc, f) for f in fields]
