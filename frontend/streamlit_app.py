"""Document Intelligence — Streamlit frontend.

Talks to the FastAPI backend over HTTP when one is reachable (local dev,
Docker). If it isn't — e.g. running standalone on Streamlit Community
Cloud — it transparently falls back to embedded.py, which runs the same
pipeline in-process. Either way:
    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import io
import json
import os
import time

import requests
import streamlit as st

import embedded

API = os.environ.get("DOCINTEL_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Document Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────── styling ───────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    #MainMenu, footer { visibility: hidden; }

    .app-header { padding: 0.25rem 0 0.75rem 0; border-bottom: 1px solid #E4E2DA; margin-bottom: 1.25rem; }
    .app-header h1 { font-size: 1.9rem; font-weight: 700; margin: 0; color: #1C1C1A; letter-spacing: -0.01em; }
    .app-header p { font-size: 0.98rem; color: #5B5A53; margin: 0.35rem 0 0 0; max-width: 46rem; }

    .step-strip { display: flex; gap: 0.9rem; margin: 1rem 0 1.4rem 0; flex-wrap: wrap; }
    .step-item { flex: 1; min-width: 190px; background: #FFFFFF; border: 1px solid #E4E2DA;
                 border-radius: 10px; padding: 0.85rem 1rem; }
    .step-item .step-num { display: inline-block; width: 22px; height: 22px; border-radius: 50%;
                            background: #2F5D5D; color: #fff; font-size: 0.78rem; font-weight: 700;
                            text-align: center; line-height: 22px; margin-right: 0.5rem; }
    .step-item .step-title { font-weight: 600; font-size: 0.92rem; color: #1C1C1A; }
    .step-item .step-desc { font-size: 0.82rem; color: #6B6A62; margin-top: 0.3rem; line-height: 1.35; }

    .tab-intro { font-size: 0.9rem; color: #6B6A62; margin: 0.6rem 0 1.1rem 0; }

    .meta-strip { font-size: 0.88rem; color: #4A4940; background: #F0EFEA; border-radius: 8px;
                  padding: 0.6rem 0.85rem; margin-bottom: 0.9rem; }
    .meta-strip b { color: #1C1C1A; }

    table.field-table { width: 100%; border-collapse: collapse; font-size: 0.89rem; }
    table.field-table th { text-align: left; font-weight: 600; color: #6B6A62; font-size: 0.76rem;
                            text-transform: uppercase; letter-spacing: 0.03em; padding: 0.4rem 0.6rem;
                            border-bottom: 1px solid #E4E2DA; }
    table.field-table td { padding: 0.55rem 0.6rem; border-bottom: 1px solid #EEEDE7; vertical-align: top; }
    table.field-table tr:last-child td { border-bottom: none; }
    table.field-table .fname { font-weight: 600; color: #1C1C1A; }
    table.field-table .fsource { color: #8B8A81; font-size: 0.8rem; }

    .pill { display: inline-block; padding: 0.16rem 0.55rem; border-radius: 999px; font-size: 0.78rem;
            font-weight: 600; white-space: nowrap; }
    .pill-high { background: #E4F0E9; color: #226644; }
    .pill-med  { background: #FBF0DA; color: #8A5A10; }
    .pill-low  { background: #F8E4E1; color: #A13B2E; }

    .answer-card { background: #FFFFFF; border: 1px solid #E4E2DA; border-left: 4px solid #2F5D5D;
                   border-radius: 8px; padding: 1rem 1.2rem; margin: 0.9rem 0; }
    .answer-card .answer-label { font-size: 0.76rem; font-weight: 700; text-transform: uppercase;
                                  letter-spacing: 0.04em; color: #2F5D5D; margin-bottom: 0.4rem; }
    .answer-card .answer-text { font-size: 1.02rem; color: #1C1C1A; line-height: 1.5; }

    .source-card { background: #FAFAF7; border: 1px solid #E4E2DA; border-radius: 8px;
                   padding: 0.7rem 0.9rem; margin-bottom: 0.55rem; }
    .source-card .source-head { font-size: 0.82rem; font-weight: 600; color: #1C1C1A; }
    .source-card .source-meta { font-size: 0.76rem; color: #8B8A81; margin-left: 0.4rem; }
    .source-card .source-snippet { font-size: 0.85rem; color: #4A4940; margin-top: 0.3rem;
                                    font-style: italic; line-height: 1.4; }

    div[data-testid="stButton"] button[kind="primary"] { background-color: #2F5D5D; border-color: #2F5D5D; }
    div[data-testid="stButton"] button[kind="primary"]:hover { background-color: #254A4A; border-color: #254A4A; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
      <h1>Document Intelligence</h1>
      <p>Upload a contract, invoice, financial statement, or court filing.
      A hybrid pipeline (a trained classifier, layout-aware extraction, and a
      rules engine) pulls out the fields that matter, scores its own
      confidence, and shows you exactly where each value came from on the
      page — so you can verify it instead of just trusting it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "embedded_mode" not in st.session_state:
    try:
        requests.get(f"{API}/health", timeout=2)
        st.session_state["embedded_mode"] = False
    except requests.exceptions.RequestException:
        st.session_state["embedded_mode"] = True

st.markdown(
    """
    <div class="step-strip">
      <div class="step-item"><span class="step-num">1</span><span class="step-title">Upload</span>
        <div class="step-desc">Drop in a PDF or scanned image. The document type is detected automatically.</div></div>
      <div class="step-item"><span class="step-num">2</span><span class="step-title">Review</span>
        <div class="step-desc">Every extracted field comes with a confidence score and a highlighted box on the page.</div></div>
      <div class="step-item"><span class="step-num">3</span><span class="step-title">Ask</span>
        <div class="step-desc">Ask questions in plain language across everything you've uploaded, with citations.</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state["embedded_mode"]:
    st.caption(
        "Running standalone for this demo — extraction happens in-process on "
        "upload rather than as a background job. Stored documents reset "
        "when this instance restarts."
    )


def api(method: str, path: str, **kwargs):
    if st.session_state["embedded_mode"]:
        return embedded.dispatch(method, path, **kwargs)
    try:
        r = requests.request(method, f"{API}{path}", timeout=120, **kwargs)
        return r
    except requests.exceptions.RequestException:
        st.session_state["embedded_mode"] = True
        return embedded.dispatch(method, path, **kwargs)


def humanize(s: str) -> str:
    return (s or "").replace("_", " ").replace("+", " + ").strip().title()


def confidence_pill(conf: float) -> str:
    cls = "pill-high" if conf >= 0.8 else ("pill-med" if conf >= 0.55 else "pill-low")
    return f'<span class="pill {cls}">{conf:.0%}</span>'


def render_result(doc_id: str, meta: dict) -> None:
    """Shared extraction view — used right after upload and from Review."""
    result = api("GET", f"/api/documents/{doc_id}/result").json()

    doc_type = humanize(result["document_type"])
    extractors = humanize(", ".join(result["meta"]["extractors_used"]))
    scan_note = "Scanned document (OCR)" if result["meta"]["is_scanned"] else "Native PDF text"
    st.markdown(
        f"""<div class="meta-strip">
        Detected as <b>{doc_type}</b> ({result['document_type_confidence']:.0%} confidence)
        &nbsp;·&nbsp; Extracted with <b>{extractors}</b>
        &nbsp;·&nbsp; {scan_note}
        &nbsp;·&nbsp; {result['meta']['num_pages']} page(s)
        </div>""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("**Extracted fields**")
        rows_html = "".join(
            f"""<tr>
                <td class="fname">{humanize(name)}</td>
                <td>{f['value']}</td>
                <td>{confidence_pill(f['confidence'])}</td>
                <td class="fsource">{humanize(f['source'])}</td>
                <td class="fsource">{f.get('page') or '—'}</td>
            </tr>"""
            for name, f in sorted(result["fields"].items())
        )
        st.markdown(
            f"""<table class="field-table">
            <tr><th>Field</th><th>Value</th><th>Confidence</th><th>Source</th><th>Page</th></tr>
            {rows_html}
            </table>""",
            unsafe_allow_html=True,
        )

        if result.get("tables"):
            st.markdown("**Tables**")
            for i, table in enumerate(result["tables"], 1):
                st.caption(f"Table {i} · page {table['page']}")
                st.table(table["rows"])

        st.download_button(
            "Download extraction as JSON",
            data=json.dumps(result, indent=2),
            file_name=f"{meta['filename']}.extraction.json",
            mime="application/json",
        )

    with right:
        st.markdown("**Where it came from**")
        pages_with_fields = sorted(
            {f.get("page") for f in result["fields"].values() if f.get("page")}
        ) or [1]
        page_num = (
            st.selectbox("Page", pages_with_fields, key=f"page_{doc_id}")
            if len(pages_with_fields) > 1 else pages_with_fields[0]
        )
        img = api("GET", f"/api/documents/{doc_id}/page/{page_num}")
        if img.status_code == 200:
            from PIL import Image, ImageDraw

            image = Image.open(io.BytesIO(img.content)).convert("RGB")
            draw = ImageDraw.Draw(image)
            w, h = image.size
            accent = (47, 93, 93)

            fields_on_page = [
                (name, f) for name, f in result["fields"].items()
                if f.get("page") == page_num and f.get("bbox")
            ]
            # process top-to-bottom so label collision-avoidance is predictable
            fields_on_page.sort(key=lambda nf: nf[1]["bbox"][1])

            placed_labels: list = []  # [x0, y0, x1, y1] of chips already drawn

            def overlaps(a, b) -> bool:
                return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

            for name, f in fields_on_page:
                x0, y0, x1, y1 = f["bbox"]
                box = [x0 * w - 4, y0 * h - 4, x1 * w + 4, y1 * h + 4]
                draw.rectangle(box, outline=accent, width=3)

                label = humanize(name)
                tb = draw.textbbox((0, 0), label)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                chip_w, chip_h = tw + 10, th + 6

                # prefer beside the box (to the right, vertically centered) —
                # avoids colliding with the line above on tightly packed
                # tables/statements. Fall back below the box if no room.
                if box[2] + 6 + chip_w <= w:
                    chip = [box[2] + 6, box[1] - 2, box[2] + 6 + chip_w, box[1] - 2 + chip_h]
                else:
                    chip = [box[0], box[3] + 4, box[0] + chip_w, box[3] + 4 + chip_h]

                attempts = 0
                while any(overlaps(chip, p) for p in placed_labels) and attempts < 6:
                    chip = [chip[0], chip[1] + chip_h + 3, chip[2], chip[3] + chip_h + 3]
                    attempts += 1

                draw.rectangle(chip, fill=accent)
                draw.text((chip[0] + 5, chip[1] + 3), label, fill=(255, 255, 255))
                placed_labels.append(chip)

            st.image(image, width="stretch")
        else:
            st.info("Page preview unavailable.")


tab_upload, tab_docs, tab_query, tab_dash = st.tabs(
    ["Upload", "Review", "Ask", "Insights"]
)

# ────────────────────────────── Upload ──────────────────────────────
with tab_upload:
    st.markdown(
        '<div class="tab-intro">Add a document below. Extraction runs '
        "automatically — no configuration needed — and results appear on "
        "this page as soon as it's done.</div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload a PDF or scanned image",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
        label_visibility="collapsed",
    )
    if uploaded and st.button("Process document", type="primary"):
        r = api("POST", "/api/documents/upload",
                files={"file": (uploaded.name, uploaded.getvalue())})
        if r.status_code != 202:
            st.error(r.json().get("detail", r.text))
        else:
            doc_id = r.json()["id"]
            with st.spinner("Reading and extracting…"):
                for _ in range(240):
                    status = api("GET", f"/api/documents/{doc_id}").json()
                    if status["status"] in ("completed", "failed"):
                        break
                    time.sleep(1)
            if status["status"] == "completed":
                st.success(
                    f"Detected **{humanize(status['doc_type'])}** "
                    f"in {status['processing_seconds']}s."
                )
                st.session_state["selected_doc"] = doc_id
                render_result(doc_id, status)
            else:
                st.error(f"Processing failed: {status.get('error')}")

# ───────────────────────────── Review ─────────────────────────────
with tab_docs:
    st.markdown(
        '<div class="tab-intro">Browse anything you\'ve uploaded — fields, '
        "confidence scores, tables, and the exact region of the page each "
        "value was pulled from.</div>",
        unsafe_allow_html=True,
    )
    docs = api("GET", "/api/documents").json()
    if not docs:
        st.info("Nothing here yet. Upload a document in the first tab to see it appear here.")
    else:
        labels = {
            d["id"]: f"{d['filename']} — {humanize(d.get('doc_type')) or d['status']}"
            for d in docs
        }
        default = st.session_state.get("selected_doc", docs[0]["id"])
        ids = list(labels)
        doc_id = st.selectbox(
            "Choose a document", ids,
            index=ids.index(default) if default in ids else 0,
            format_func=lambda i: labels[i],
        )
        meta = next(d for d in docs if d["id"] == doc_id)

        if meta["status"] != "completed":
            st.warning(f"Status: {meta['status']} {meta.get('error') or ''}")
        else:
            render_result(doc_id, meta)

# ──────────────────────────────── Ask ────────────────────────────────
with tab_query:
    st.markdown(
        '<div class="tab-intro">Ask a question in plain language across '
        "everything you've uploaded, or narrow it to one document. Answers "
        "are backed by the actual text, with the source page shown "
        "underneath.</div>",
        unsafe_allow_html=True,
    )
    docs = api("GET", "/api/documents").json()
    completed_docs = [d for d in docs if d["status"] == "completed"]

    examples = [
        "What are the termination clauses?",
        "What is the total amount due?",
        "Who are the parties involved?",
        "What are the payment terms?",
    ]
    st.caption("Try one of these, or write your own:")
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, key=f"ex_{ex}"):
            st.session_state["ask_question_input"] = ex

    scope = st.selectbox(
        "Scope", ["All documents"] + [d["filename"] for d in completed_docs],
    )
    question = st.text_input(
        "Question", placeholder="e.g. What are the termination clauses?",
        key="ask_question_input",
    )
    if question and st.button("Search", type="primary"):
        payload = {"question": question, "top_k": 5}
        if scope != "All documents":
            payload["document_id"] = next(
                d["id"] for d in completed_docs if d["filename"] == scope
            )
        r = api("POST", "/api/query", json=payload)
        body = r.json()
        mode_note = "Answered by Claude from indexed excerpts" if body["answer_mode"] == "llm" \
            else "Answered directly from the indexed text"
        st.markdown(
            f"""<div class="answer-card">
            <div class="answer-label">Answer</div>
            <div class="answer-text">{body['answer']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.caption(mode_note)
        # Raw cosine-similarity scores from a small TF-IDF corpus are tiny
        # and not human-legible (0.03, 0.00...). Show relative match
        # strength against the best hit instead, and drop true-zero noise.
        sources = [s for s in body["sources"] if s["score"] > 0]
        if sources:
            best = max(s["score"] for s in sources)
            st.markdown("**Sources**")
            for s in sources[:3]:
                rel = s["score"] / best if best else 0
                cls = "pill-high" if rel >= 0.7 else ("pill-med" if rel >= 0.35 else "pill-low")
                st.markdown(
                    f"""<div class="source-card">
                    <span class="source-head">{s['filename']}</span>
                    <span class="source-meta">page {s['page']}</span>
                    <span class="pill {cls}">{rel:.0%} match</span>
                    <div class="source-snippet">"{s['snippet']}"</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

# ───────────────────────────── Insights ─────────────────────────────
with tab_dash:
    st.markdown(
        '<div class="tab-intro">Aggregate view across everything this '
        "instance has processed.</div>",
        unsafe_allow_html=True,
    )
    stats = api("GET", "/api/dashboard/stats").json()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", stats["total_documents"])
    c2.metric("Fields extracted", stats["total_fields_extracted"])
    c3.metric(
        "Avg. confidence",
        f"{stats['avg_field_confidence']:.0%}" if stats["avg_field_confidence"] else "—",
    )
    c4.metric(
        "Avg. processing time",
        f"{stats['avg_processing_seconds']}s" if stats["avg_processing_seconds"] else "—",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**By document type**")
        if stats["by_type"]:
            st.bar_chart({humanize(k): v for k, v in stats["by_type"].items()})
        else:
            st.info("No completed documents yet.")
    with col_b:
        st.markdown("**By status**")
        st.bar_chart({humanize(k): v for k, v in stats["by_status"].items()})

    st.markdown("**Recent documents**")
    st.dataframe(
        [
            {"File": d["filename"], "Type": humanize(d.get("doc_type")) or "—",
             "Status": humanize(d["status"]), "Pages": d.get("num_pages") or "—",
             "Seconds": d.get("processing_seconds") or "—"}
            for d in stats["recent"]
        ],
        width="stretch", hide_index=True,
    )
