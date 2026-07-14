"""Evaluation harness: hybrid pipeline vs. classic baseline.

Metrics
-------
Field-level accuracy against ground truth, per document type and overall:

    accuracy = correctly extracted fields / total ground-truth fields

A field counts as correct when the normalized predicted value matches the
normalized ground truth (exact, containment, or numeric-equal for amounts).

Systems compared
----------------
* hybrid    — the full pipeline (backend.app.pipeline.hybrid): document-type
              detection + LayoutLMv3/Donut (when installed) + rules layer +
              merge. This is the production path.
* baseline  — the classic pre-LLM approach: OCR/text layer + spaCy NER
              mapping (backend/app/pipeline/extractors/ocr_fallback.py).
              When spaCy is not installed, a naive first-match heuristic
              stands in (documented in the report).

Usage:
    python evaluation/run_eval.py [--dataset evaluation/dataset]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.pipeline import hybrid  # noqa: E402
from backend.app.pipeline.classifier import classify  # noqa: E402
from backend.app.pipeline.extractors.ocr_fallback import OcrSpacyExtractor  # noqa: E402
from backend.app.pipeline.ingest import ingest  # noqa: E402


# ───────────────────────── value comparison ─────────────────────────

def _norm(v: str) -> str:
    v = v.lower().strip()
    v = re.sub(r"[$,]", "", v)
    v = re.sub(r"[^\w. ]", " ", v)
    return re.sub(r"\s+", " ", v).strip()


def _as_number(v: str):
    m = re.search(r"\d[\d,]*(?:\.\d+)?", v.replace(",", ""))
    return float(m.group()) if m else None


def values_match(predicted: str, truth: str) -> bool:
    p, t = _norm(predicted), _norm(truth)
    if not p or not t:
        return False
    if p == t or t in p or p in t:
        return True
    pn, tn = _as_number(predicted), _as_number(truth)
    if pn is not None and tn is not None and "$" in predicted + truth:
        return abs(pn - tn) < 0.01
    return False


# ───────────────────────────── baseline ─────────────────────────────

_DATE_RE = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
_MONEY_RE = r"\$\s?\d[\d,]*(?:\.\d{2})?"

_DATE_FIELDS = {"invoice_date", "due_date", "effective_date", "termination_date",
                "period", "filing_date", "date"}
_MONEY_FIELDS = {"subtotal", "tax", "total_amount", "contract_value", "total_revenue",
                 "net_income", "total_assets", "total_liabilities",
                 "operating_expenses", "shareholders_equity"}


def naive_baseline(text: str, truth_fields: dict) -> dict:
    """First-match heuristics — what a hand-rolled script without domain
    schemas would do. Used only when spaCy is unavailable."""
    dates = re.findall(_DATE_RE, text)
    monies = re.findall(_MONEY_RE, text)
    caps_lines = [
        line.strip() for line in text.splitlines()
        if re.match(r"^[A-Z][A-Za-z&.,' ]{4,60}$", line.strip())
    ]
    out = {}
    di = mi = ci = 0
    for name in truth_fields:
        if name in _DATE_FIELDS and di < len(dates):
            out[name] = dates[di]; di += 1
        elif name in _MONEY_FIELDS and mi < len(monies):
            out[name] = monies[mi]; mi += 1
        elif ci < len(caps_lines):
            out[name] = caps_lines[ci]; ci += 1
    return out


def run_baseline(path: str, truth: dict) -> dict:
    doc = ingest(path)
    extractor = OcrSpacyExtractor()
    if extractor.available():
        doc_type, _ = classify(doc.full_text)
        fields = extractor.extract(doc, doc_type)
        return {f.name: f.value for f in fields}, "tesseract+spacy"
    return naive_baseline(doc.full_text, truth["fields"]), "naive-regex (spaCy not installed)"


# ─────────────────────────────── main ───────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation/dataset")
    parser.add_argument("--results", default="evaluation/results")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    manifest = json.loads((dataset / "manifest.json").read_text())
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    per_doc = []
    scores = {"hybrid": defaultdict(lambda: [0, 0]), "baseline": defaultdict(lambda: [0, 0])}
    type_correct = 0
    baseline_name = None
    t0 = time.time()

    for entry in manifest:
        path = str(dataset / entry["file"])
        truth = {"doc_type": entry["doc_type"], "fields": entry["fields"]}

        hybrid_result = hybrid.process_document(path)
        hybrid_fields = {k: v["value"] for k, v in hybrid_result["fields"].items()}
        if hybrid_result["document_type"] == truth["doc_type"]:
            type_correct += 1

        baseline_fields, baseline_name = run_baseline(path, truth)

        row = {"id": entry["id"], "doc_type": truth["doc_type"],
               "detected_type": hybrid_result["document_type"], "fields": {}}
        for name, expected in truth["fields"].items():
            h_val = hybrid_fields.get(name)
            b_val = baseline_fields.get(name)
            h_ok = bool(h_val and values_match(h_val, expected))
            b_ok = bool(b_val and values_match(b_val, expected))
            scores["hybrid"][truth["doc_type"]][0] += int(h_ok)
            scores["hybrid"][truth["doc_type"]][1] += 1
            scores["baseline"][truth["doc_type"]][0] += int(b_ok)
            scores["baseline"][truth["doc_type"]][1] += 1
            row["fields"][name] = {
                "expected": expected,
                "hybrid": h_val, "hybrid_correct": h_ok,
                "baseline": b_val, "baseline_correct": b_ok,
            }
        per_doc.append(row)
        print(f"  {entry['id']:<24} type={'OK ' if row['detected_type'] == truth['doc_type'] else 'MISS'} "
              f"hybrid {sum(f['hybrid_correct'] for f in row['fields'].values())}/{len(row['fields'])}  "
              f"baseline {sum(f['baseline_correct'] for f in row['fields'].values())}/{len(row['fields'])}")

    def overall(system: str):
        c = sum(v[0] for v in scores[system].values())
        n = sum(v[1] for v in scores[system].values())
        return c, n, (c / n if n else 0.0)

    hc, hn, hacc = overall("hybrid")
    bc, bn, bacc = overall("baseline")
    elapsed = time.time() - t0

    summary = {
        "documents": len(manifest),
        "doc_type_detection_accuracy": round(type_correct / len(manifest), 4),
        "hybrid": {
            "correct": hc, "total": hn, "accuracy": round(hacc, 4),
            "per_type": {t: {"correct": v[0], "total": v[1],
                             "accuracy": round(v[0] / v[1], 4)}
                         for t, v in scores["hybrid"].items()},
        },
        "baseline": {
            "system": baseline_name,
            "correct": bc, "total": bn, "accuracy": round(bacc, 4),
            "per_type": {t: {"correct": v[0], "total": v[1],
                             "accuracy": round(v[0] / v[1], 4)}
                         for t, v in scores["baseline"].items()},
        },
        "improvement_pp": round((hacc - bacc) * 100, 1),
        "elapsed_seconds": round(elapsed, 1),
    }

    (results_dir / "results.json").write_text(
        json.dumps({"summary": summary, "per_document": per_doc}, indent=2)
    )

    lines = [
        "# Evaluation Report",
        "",
        f"- Documents evaluated: **{summary['documents']}**",
        f"- Document-type detection accuracy: **{summary['doc_type_detection_accuracy']:.1%}**",
        f"- Baseline system: {baseline_name}",
        "",
        "| System | Fields correct | Accuracy |",
        "|---|---|---|",
        f"| **Hybrid pipeline** | {hc}/{hn} | **{hacc:.1%}** |",
        f"| Baseline | {bc}/{bn} | {bacc:.1%} |",
        "",
        f"**Improvement: +{summary['improvement_pp']} percentage points**",
        "",
        "## Per document type (hybrid)",
        "",
        "| Type | Hybrid | Baseline |",
        "|---|---|---|",
    ]
    for t in scores["hybrid"]:
        h = scores["hybrid"][t]
        b = scores["baseline"][t]
        lines.append(f"| {t} | {h[0]}/{h[1]} ({h[0]/h[1]:.1%}) | {b[0]}/{b[1]} ({b[0]/b[1]:.1%}) |")
    (results_dir / "report.md").write_text("\n".join(lines) + "\n")

    print("\n=== SUMMARY ===")
    print(f"Doc-type detection: {summary['doc_type_detection_accuracy']:.1%}")
    print(f"Hybrid pipeline : {hc}/{hn} = {hacc:.1%}")
    print(f"Baseline ({baseline_name}): {bc}/{bn} = {bacc:.1%}")
    print(f"Improvement: +{summary['improvement_pp']} pp")
    print(f"Report written to {results_dir / 'report.md'}")


if __name__ == "__main__":
    main()
