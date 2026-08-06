"""Generate the sample documents bundled with the app for first-time visitors
who don't have a contract or invoice handy. Reuses the same generators as the
evaluation dataset, at fixed index 0 for clean, deterministic output.

Output: frontend/samples/*.pdf — committed to the repo so they ship with
every deployment, including the standalone Streamlit Cloud demo.

Usage: python scripts/make_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

from generate_dataset import (make_contract, make_court, make_financial,  # noqa: E402
                              make_invoice)

OUT = ROOT / "frontend" / "samples"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    ("sample_invoice.pdf", make_invoice,
     "Invoice", "A vendor invoice with line items, tax, and a total due."),
    ("sample_contract.pdf", make_contract,
     "Contract", "A services agreement with parties, term, and governing law."),
    ("sample_financial_statement.pdf", make_financial,
     "Financial statement", "An income statement and balance sheet with GAAP figures."),
    ("sample_court_filing.pdf", make_court,
     "Court filing", "A civil complaint with case number, plaintiff, and defendant."),
]

manifest = []
for filename, generator, label, description in SAMPLES:
    path = OUT / filename
    generator(path, 0)
    manifest.append({"file": filename, "label": label, "description": description})
    print(f"wrote {path}")

import json  # noqa: E402
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"wrote {OUT / 'manifest.json'}")
