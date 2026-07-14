"""Table extraction from native PDFs.

Two strategies:
  1. pdfplumber's line-based detection for ruled tables.
  2. A word-geometry detector for borderless tables: words are grouped into
     lines, split into cells wherever a large horizontal gap occurs, and
     consecutive multi-cell lines with aligned columns are emitted as a table.
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

_GAP_FRACTION = 0.035   # min horizontal gap between cells, as fraction of page width
_LINE_TOLERANCE = 0.6   # same-line tolerance, as fraction of median word height
_MIN_RUN = 3            # min consecutive multi-cell lines to call it a table


def _group_lines(words: List[dict], page_height: float) -> List[List[dict]]:
    if not words:
        return []
    heights = sorted(w["bottom"] - w["top"] for w in words)
    median_h = heights[len(heights) // 2] or 8.0
    tol = median_h * _LINE_TOLERANCE

    lines: List[List[dict]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        center = (w["top"] + w["bottom"]) / 2
        if lines:
            last_center = sum(
                (x["top"] + x["bottom"]) / 2 for x in lines[-1]
            ) / len(lines[-1])
            if abs(center - last_center) <= tol:
                lines[-1].append(w)
                continue
        lines.append([w])
    for line in lines:
        line.sort(key=lambda w: w["x0"])
    return lines


def _split_cells(line: List[dict], page_width: float) -> List[dict]:
    """Split a line of words into cells at large horizontal gaps."""
    gap = page_width * _GAP_FRACTION
    cells: List[dict] = []
    current = [line[0]]
    for prev, word in zip(line, line[1:]):
        if word["x0"] - prev["x1"] > gap:
            cells.append(current)
            current = [word]
        else:
            current.append(word)
    cells.append(current)
    return [
        {
            "text": " ".join(w["text"] for w in c),
            "x0": min(w["x0"] for w in c),
            "x1": max(w["x1"] for w in c),
            "top": min(w["top"] for w in c),
            "bottom": max(w["bottom"] for w in c),
        }
        for c in cells
    ]


def _borderless_tables(page, page_number: int) -> List[dict]:
    words = page.extract_words() or []
    lines = _group_lines(words, page.height)
    cell_lines = [_split_cells(line, page.width) for line in lines]

    tables: List[dict] = []
    run: List[List[dict]] = []

    def flush() -> None:
        nonlocal run
        if len(run) >= _MIN_RUN:
            tables.append(_build_table(run, page, page_number))
        run = []

    for cells in cell_lines:
        if len(cells) >= 2:
            run.append(cells)
        else:
            flush()
    flush()
    return [t for t in tables if t is not None]


def _build_table(run: List[List[dict]], page, page_number: int) -> Optional[dict]:
    # cluster cell x0 positions into columns
    xs = sorted(c["x0"] for row in run for c in row)
    tol = page.width * 0.02
    columns: List[float] = []
    for x in xs:
        if not columns or x - columns[-1] > tol:
            columns.append(x)
    if len(columns) < 2:
        return None

    def col_index(x: float) -> int:
        return min(range(len(columns)), key=lambda i: abs(columns[i] - x))

    rows: List[List[Optional[str]]] = []
    for line in run:
        row: List[Optional[str]] = [None] * len(columns)
        for cell in line:
            idx = col_index(cell["x0"])
            row[idx] = f"{row[idx]} {cell['text']}" if row[idx] else cell["text"]
        rows.append(row)

    all_cells = [c for row in run for c in row]
    bbox = [
        min(c["x0"] for c in all_cells) / page.width,
        min(c["top"] for c in all_cells) / page.height,
        max(c["x1"] for c in all_cells) / page.width,
        max(c["bottom"] for c in all_cells) / page.height,
    ]
    return {"page": page_number, "rows": rows, "bbox": bbox}


def extract_tables(path: str) -> List[dict]:
    """Return [{page, rows, bbox}] for every detected table in a PDF.
    bbox is normalized [x0, y0, x1, y1]."""
    if not path.lower().endswith(".pdf"):
        return []
    import pdfplumber

    out: List[dict] = []
    try:
        with pdfplumber.open(path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                ruled = page.find_tables()
                if ruled:
                    for table in ruled:
                        rows = table.extract()
                        if not rows or (len(rows) == 1 and len(rows[0]) <= 1):
                            continue
                        cleaned = [
                            [c.strip() if isinstance(c, str) else c for c in row]
                            for row in rows
                        ]
                        bbox = None
                        if table.bbox:
                            x0, top, x1, bottom = table.bbox
                            bbox = [x0 / page.width, top / page.height,
                                    x1 / page.width, bottom / page.height]
                        out.append({"page": idx, "rows": cleaned, "bbox": bbox})
                else:
                    out.extend(_borderless_tables(page, idx))
    except Exception as exc:  # pragma: no cover
        logger.warning("table extraction failed for %s: %s", path, exc)
    return out
