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
from pathlib import Path

import requests
import streamlit as st

import embedded

API = os.environ.get("DOCINTEL_API_URL", "http://localhost:8000")
SAMPLES_DIR = Path(__file__).parent / "samples"

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
    .app-header p { font-size: 0.98rem; color: #5B5A53; margin: 0.35rem 0 0 0; max-width: 46rem; line-height: 1.5; }

    .step-strip { display: flex; gap: 0.9rem; margin: 1rem 0 1.1rem 0; flex-wrap: wrap; }
    .step-item { flex: 1; min-width: 190px; background: #FFFFFF; border: 1px solid #E4E2DA;
                 border-radius: 10px; padding: 0.85rem 1rem; }
    .step-item .step-num { display: inline-block; width: 22px; height: 22px; border-radius: 50%;
                            background: #2F5D5D; color: #fff; font-size: 0.78rem; font-weight: 700;
                            text-align: center; line-height: 22px; margin-right: 0.5rem; }
    .step-item .step-title { font-weight: 600; font-size: 0.92rem; color: #1C1C1A; }
    .step-item .step-desc { font-size: 0.82rem; color: #6B6A62; margin-top: 0.3rem; line-height: 1.35; }

    .nav-row { display: flex; gap: 0.4rem; margin-bottom: 1.1rem; border-bottom: 1px solid #E4E2DA;
               padding-bottom: 1rem; }

    .tab-intro { font-size: 0.92rem; color: #6B6A62; margin: 0 0 1.1rem 0; line-height: 1.5; }

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

    .sample-grid { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 0.4rem; }
    .sample-card { flex: 1; min-width: 220px; background: #FFFFFF; border: 1px solid #E4E2DA;
                   border-radius: 10px; padding: 0.9rem 1rem 0.7rem 1rem; }
    .sample-card .sample-title { font-weight: 600; font-size: 0.92rem; color: #1C1C1A; }
    .sample-card .sample-desc { font-size: 0.8rem; color: #6B6A62; margin: 0.3rem 0 0.7rem 0;
                                line-height: 1.4; min-height: 2.6rem; }

    .cta-row { margin-top: 1rem; padding-top: 0.9rem; border-top: 1px solid #E4E2DA; }
    .cta-row .cta-label { font-size: 0.85rem; color: #6B6A62; margin-bottom: 0.5rem; }

    .pipeline-step { display: flex; gap: 0.7rem; padding: 0.5rem 0; border-bottom: 1px solid #EEEDE7; }
    .pipeline-step:last-child { border-bottom: none; }
    .pipeline-step .pstep-num { flex-shrink: 0; width: 20px; height: 20px; border-radius: 50%;
                                 background: #F0EFEA; color: #4A4940; font-size: 0.72rem; font-weight: 700;
                                 text-align: center; line-height: 20px; }
    .pipeline-step .pstep-title { font-weight: 600; font-size: 0.86rem; color: #1C1C1A; }
    .pipeline-step .pstep-desc { font-size: 0.82rem; color: #6B6A62; line-height: 1.4; margin-top: 0.15rem; }

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
      <p>Drop in a contract, invoice, financial statement, or court filing and
      a hybrid pipeline reads it the way a careful paralegal would: it figures
      out what kind of document this is, pulls out the values that matter,
      and — instead of asking you to take its word for it — shows you a
      confidence score and the exact spot on the page each value came from.
      No account, no setup. If you don't have a document on hand, there are
      sample files below to try it with.</p>
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
        <div class="step-desc">Use your own file or one of the samples. The document type is detected automatically.</div></div>
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
        "This demo instance runs standalone — extraction happens in-process "
        "the moment you upload rather than as a background job, and stored "
        "documents reset whenever this instance restarts. Everything else "
        "is the real pipeline."
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


def suggested_question(doc_type: str) -> str:
    return {
        "invoice": "What is the total amount due?",
        "receipt": "What is the total amount due?",
        "contract": "What are the termination clauses?",
        "financial_statement": "What is the total revenue?",
        "court_document": "Who are the parties involved?",
    }.get(doc_type, "What are the key details in this document?")


def goto(page: str, **prefill) -> None:
    """Programmatic navigation — used by 'Ask about this document' style
    call-to-actions to jump pages with context pre-filled."""
    st.session_state["page"] = page
    for key, value in prefill.items():
        st.session_state[key] = value
    st.rerun()


# ────────────────────────────── processing ──────────────────────────────

_PIPELINE_STAGES = [
    "Reading the document and locating text on the page…",
    "Detecting the document type…",
    "Running layout-aware extraction and the rules engine…",
    "Cross-checking extractors and scoring confidence…",
]


def process_and_store(filename: str, file_bytes: bytes) -> dict | None:
    """Upload + poll to completion, narrating the real pipeline stages while
    it runs. Returns the document status dict, or None on failure."""
    r = api("POST", "/api/documents/upload", files={"file": (filename, file_bytes)})
    if r.status_code != 202:
        st.error(r.json().get("detail", r.text))
        return None

    doc_id = r.json()["id"]
    with st.status("Processing…", expanded=True) as status:
        for stage in _PIPELINE_STAGES:
            status.update(label=stage)
            time.sleep(0.2)
        for _ in range(240):
            result = api("GET", f"/api/documents/{doc_id}").json()
            if result["status"] in ("completed", "failed"):
                break
            time.sleep(1)
        if result["status"] == "completed":
            status.update(
                label=f"Detected {humanize(result['doc_type'])} "
                      f"in {result['processing_seconds']}s.",
                state="complete",
            )
        else:
            status.update(label="Processing failed.", state="error")

    if result["status"] != "completed":
        st.error(f"Processing failed: {result.get('error')}")
        return None
    st.session_state["selected_doc"] = doc_id
    return result


def render_result(doc_id: str, meta: dict, show_ask_cta: bool = True) -> None:
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
            key=f"dl_{doc_id}",
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
            fields_on_page.sort(key=lambda nf: nf[1]["bbox"][1])
            placed_labels: list = []

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

    if show_ask_cta:
        st.markdown('<div class="cta-row"><div class="cta-label">Have a question about '
                     'this document?</div></div>', unsafe_allow_html=True)
        suggestion = suggested_question(result["document_type"])
        if st.button(f'Ask: "{suggestion}"', key=f"ask_cta_{doc_id}"):
            goto("Ask", ask_scope_select=meta["filename"], ask_question_input=suggestion)


# ─────────────────────────────── navigation ───────────────────────────────

PAGES = ["Upload", "Review", "Ask", "Insights"]
if "page" not in st.session_state:
    st.session_state["page"] = "Upload"

nav_cols = st.columns([1, 1, 1, 1, 4])
for col, item in zip(nav_cols, PAGES):
    is_active = st.session_state["page"] == item
    if col.button(item, key=f"nav_{item}", type="primary" if is_active else "secondary"):
        st.session_state["page"] = item
        st.rerun()
st.markdown('<div style="margin-bottom: 0.8rem;"></div>', unsafe_allow_html=True)

page = st.session_state["page"]

# ────────────────────────────── Upload ──────────────────────────────
if page == "Upload":
    st.markdown(
        '<div class="tab-intro">Add a document below. Extraction runs '
        "automatically, and results appear on this page as soon as it's "
        "done — no other tab to go find them in.</div>",
        unsafe_allow_html=True,
    )

    with st.expander("What actually happens when you upload a document?"):
        st.markdown(
            """
            <div class="pipeline-step">
              <div class="pstep-num">1</div>
              <div><div class="pstep-title">Read the page</div>
              <div class="pstep-desc">Native PDFs are parsed directly for text and word positions.
              Scans and images with no text layer are routed through OCR (Tesseract) instead —
              either way, the result is the same: text plus where it sits on the page.</div></div>
            </div>
            <div class="pipeline-step">
              <div class="pstep-num">2</div>
              <div><div class="pstep-title">Classify the document</div>
              <div class="pstep-desc">A classifier trained specifically for this task (not a
              general-purpose language model) decides whether it's an invoice, contract, financial
              statement, or court filing, cross-checked against a keyword scorer.</div></div>
            </div>
            <div class="pipeline-step">
              <div class="pstep-num">3</div>
              <div><div class="pstep-title">Extract the fields</div>
              <div class="pstep-desc">Several independent methods run in parallel — a layout-aware
              model that reads position on the page, a rules engine built for the vocabulary of
              legal and financial documents, and an OCR/NLP fallback. Nobody's word is taken alone.</div></div>
            </div>
            <div class="pipeline-step">
              <div class="pstep-num">4</div>
              <div><div class="pstep-title">Merge and score confidence</div>
              <div class="pstep-desc">When two methods independently agree on a value, confidence
              goes up. When they disagree, the more reliable source wins — but always with a score
              attached, never presented as certain when it isn't.</div></div>
            </div>
            <div class="pipeline-step">
              <div class="pstep-num">5</div>
              <div><div class="pstep-title">Index for search</div>
              <div class="pstep-desc">The document's text is chunked and embedded so the Ask tab
              can answer questions about it later, with the source page cited.</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    uploaded = st.file_uploader(
        "Upload a PDF or scanned image",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
        label_visibility="collapsed",
    )
    if uploaded and st.button("Process document", type="primary"):
        result = process_and_store(uploaded.name, uploaded.getvalue())
        if result:
            st.session_state["just_processed"] = (result["id"], uploaded.name)

    if not uploaded:
        st.markdown("**Don't have a document handy? Try a sample:**")
        manifest_path = SAMPLES_DIR / "manifest.json"
        if manifest_path.exists():
            samples = json.loads(manifest_path.read_text())
            cols = st.columns(len(samples))
            for col, sample in zip(cols, samples):
                with col:
                    st.markdown(
                        f"""<div class="sample-card">
                        <div class="sample-title">{sample['label']}</div>
                        <div class="sample-desc">{sample['description']}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    if st.button(f"Use this {sample['label'].lower()}", key=f"sample_{sample['file']}"):
                        file_bytes = (SAMPLES_DIR / sample["file"]).read_bytes()
                        result = process_and_store(sample["file"], file_bytes)
                        if result:
                            st.session_state["just_processed"] = (result["id"], sample["file"])

    just_processed = st.session_state.get("just_processed")
    if just_processed:
        doc_id, filename = just_processed
        status = api("GET", f"/api/documents/{doc_id}").json()
        if status["status"] == "completed":
            st.success(
                f"Detected **{humanize(status['doc_type'])}** "
                f"({status['doc_type_confidence']:.0%} confidence) "
                f"in {status['processing_seconds']}s."
            )
            suggestion = suggested_question(status["doc_type"])
            b1, b2 = st.columns(2)
            if b1.button("View the extraction →", type="primary", key=f"view_{doc_id}"):
                goto("Review")
            if b2.button(f'Ask: "{suggestion}"', key=f"ask_upload_cta_{doc_id}"):
                goto("Ask", ask_scope_select=filename, ask_question_input=suggestion)

# ───────────────────────────── Review ─────────────────────────────
elif page == "Review":
    st.markdown(
        '<div class="tab-intro">Browse anything you\'ve uploaded — fields, '
        "confidence scores, tables, and the exact region of the page each "
        "value was pulled from.</div>",
        unsafe_allow_html=True,
    )
    docs = api("GET", "/api/documents").json()
    if not docs:
        st.info("Nothing here yet. Upload a document (or try a sample) in the "
                "first tab to see it appear here.")
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
elif page == "Ask":
    st.markdown(
        '<div class="tab-intro">Ask a question in plain language across '
        "everything you've uploaded, or narrow it to one document. Answers "
        "are backed by the actual text — for well-known questions (parties, "
        "totals, dates) the answer comes straight from the fields already "
        "extracted, so it's exact rather than reconstructed.</div>",
        unsafe_allow_html=True,
    )
    docs = api("GET", "/api/documents").json()
    completed_docs = [d for d in docs if d["status"] == "completed"]

    if not completed_docs:
        st.info("Upload a document first — there's nothing to ask about yet.")
    else:
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

        scope_options = ["All documents"] + [d["filename"] for d in completed_docs]
        scope = st.selectbox("Scope", scope_options, key="ask_scope_select")
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
elif page == "Insights":
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
