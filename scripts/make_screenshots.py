"""Produce demo screenshots: extraction highlight overlays for the docs.

Usage: python scripts/make_screenshots.py
Writes PNGs into docs/screenshots/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import ImageDraw  # noqa: E402

from backend.app.pipeline.hybrid import process_document  # noqa: E402
from backend.app.pipeline.ingest import render_pdf_page  # noqa: E402

OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = ["invoice_00", "contract_00", "court_00", "financial_00"]

for sample in SAMPLES:
    pdf = ROOT / "evaluation" / "dataset" / f"{sample}.pdf"
    if not pdf.exists():
        print(f"skip {sample} (run evaluation/generate_dataset.py first)")
        continue
    result = process_document(str(pdf))
    image = render_pdf_page(str(pdf), 1, scale=2.0)
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for name, f in result["fields"].items():
        if f.get("page") == 1 and f.get("bbox"):
            x0, y0, x1, y1 = f["bbox"]
            box = [x0 * w - 5, y0 * h - 5, x1 * w + 5, y1 * h + 5]
            draw.rectangle(box, outline=(220, 60, 60), width=3)
            draw.text((box[0], max(box[1] - 18, 0)), name, fill=(220, 60, 60))
    dest = OUT / f"{sample}_highlights.png"
    image.save(dest)
    print(f"wrote {dest}  ({len(result['fields'])} fields, "
          f"type={result['document_type']})")
