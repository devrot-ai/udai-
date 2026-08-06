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

st.set_page_config(page_title="Document Intelligence", layout="wide")
st.title("Document Intelligence")
st.caption(
    "Hybrid LayoutLMv3 + Donut + rules extraction over legal and financial "
    "documents, with semantic search."
)

if "embedded_mode" not in st.session_state:
    try:
        requests.get(f"{API}/health", timeout=2)
        st.session_state["embedded_mode"] = False
    except requests.exceptions.RequestException:
        st.session_state["embedded_mode"] = True

if st.session_state["embedded_mode"]:
    st.caption(
        "Running standalone (no separate API process) — extraction runs "
        "in-process on upload instead of as a background job."
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


def confidence_badge(conf: float) -> str:
    label = "high" if conf >= 0.8 else ("medium" if conf >= 0.55 else "low")
    return f"{conf:.0%} ({label})"


tab_upload, tab_docs, tab_query, tab_dash = st.tabs(
    ["Upload", "Documents", "Ask", "Dashboard"]
)

# ────────────────────────────── Upload ──────────────────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Upload a PDF or scanned image",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
    )
    if uploaded and st.button("Process document", type="primary"):
        r = api("POST", "/api/documents/upload",
                files={"file": (uploaded.name, uploaded.getvalue())})
        if r.status_code != 202:
            st.error(r.json().get("detail", r.text))
        else:
            doc_id = r.json()["id"]
            with st.spinner("Processing…"):
                for _ in range(240):
                    status = api("GET", f"/api/documents/{doc_id}").json()
                    if status["status"] in ("completed", "failed"):
                        break
                    time.sleep(1)
            if status["status"] == "completed":
                st.success(
                    f"Done — detected **{status['doc_type']}** "
                    f"({status['doc_type_confidence']:.0%} confidence) "
                    f"in {status['processing_seconds']}s. "
                    "See the Documents tab for the extraction."
                )
                st.session_state["selected_doc"] = doc_id
            else:
                st.error(f"Processing failed: {status.get('error')}")

# ───────────────────────────── Documents ─────────────────────────────
with tab_docs:
    docs = api("GET", "/api/documents").json()
    if not docs:
        st.info("No documents yet — upload one in the first tab.")
    else:
        labels = {
            d["id"]: f"{d['filename']}  ·  {d.get('doc_type') or d['status']}"
            for d in docs
        }
        default = st.session_state.get("selected_doc", docs[0]["id"])
        ids = list(labels)
        doc_id = st.selectbox(
            "Document", ids,
            index=ids.index(default) if default in ids else 0,
            format_func=lambda i: labels[i],
        )
        meta = next(d for d in docs if d["id"] == doc_id)

        if meta["status"] != "completed":
            st.warning(f"Status: {meta['status']} {meta.get('error') or ''}")
        else:
            result = api("GET", f"/api/documents/{doc_id}/result").json()
            left, right = st.columns([1.1, 1])

            with left:
                st.subheader("Extracted fields")
                st.markdown(
                    f"**Type:** `{result['document_type']}` "
                    f"({result['document_type_confidence']:.0%}) · "
                    f"**Extractors:** {', '.join(result['meta']['extractors_used'])} · "
                    f"**Scanned:** {result['meta']['is_scanned']}"
                )
                rows = [
                    {
                        "Field": name,
                        "Value": f["value"],
                        "Confidence": confidence_badge(f["confidence"]),
                        "Source": f["source"],
                        "Page": f.get("page") or "—",
                    }
                    for name, f in sorted(result["fields"].items())
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)

                if result.get("tables"):
                    st.subheader("Tables")
                    for i, table in enumerate(result["tables"], 1):
                        st.caption(f"Table {i} · page {table['page']}")
                        st.table(table["rows"])

                st.download_button(
                    "Download JSON",
                    data=json.dumps(result, indent=2),
                    file_name=f"{meta['filename']}.extraction.json",
                    mime="application/json",
                )

            with right:
                st.subheader("Highlighted regions")
                pages_with_fields = sorted(
                    {f.get("page") for f in result["fields"].values() if f.get("page")}
                ) or [1]
                page_num = st.selectbox("Page", pages_with_fields)
                img = api("GET", f"/api/documents/{doc_id}/page/{page_num}")
                if img.status_code == 200:
                    from PIL import Image, ImageDraw

                    image = Image.open(io.BytesIO(img.content)).convert("RGB")
                    draw = ImageDraw.Draw(image)
                    w, h = image.size
                    for name, f in result["fields"].items():
                        if f.get("page") == page_num and f.get("bbox"):
                            x0, y0, x1, y1 = f["bbox"]
                            box = [x0 * w - 4, y0 * h - 4, x1 * w + 4, y1 * h + 4]
                            draw.rectangle(box, outline=(220, 60, 60), width=3)
                            draw.text((box[0], max(box[1] - 16, 0)), name,
                                      fill=(220, 60, 60))
                    st.image(image, use_container_width=True)
                else:
                    st.info("Page preview unavailable.")

# ──────────────────────────────── Ask ────────────────────────────────
with tab_query:
    st.subheader("Semantic search over your documents")
    docs = api("GET", "/api/documents").json()
    scope = st.selectbox(
        "Scope", ["All documents"] + [d["filename"] for d in docs if d["status"] == "completed"],
    )
    question = st.text_input(
        "Question", placeholder="e.g. What are the termination clauses?"
    )
    if question and st.button("Search", type="primary"):
        payload = {"question": question, "top_k": 5}
        if scope != "All documents":
            payload["document_id"] = next(
                d["id"] for d in docs if d["filename"] == scope
            )
        r = api("POST", "/api/query", json=payload)
        body = r.json()
        st.markdown(f"### Answer  \n{body['answer']}")
        st.caption(f"mode: {body['answer_mode']}")
        with st.expander(f"Sources ({len(body['sources'])})"):
            for s in body["sources"]:
                st.markdown(
                    f"**{s['filename']}** · page {s['page']} · score {s['score']}\n\n"
                    f"> {s['snippet']}"
                )

# ───────────────────────────── Dashboard ─────────────────────────────
with tab_dash:
    stats = api("GET", "/api/dashboard/stats").json()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", stats["total_documents"])
    c2.metric("Fields extracted", stats["total_fields_extracted"])
    c3.metric(
        "Avg confidence",
        f"{stats['avg_field_confidence']:.0%}" if stats["avg_field_confidence"] else "—",
    )
    c4.metric(
        "Avg processing",
        f"{stats['avg_processing_seconds']}s" if stats["avg_processing_seconds"] else "—",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("By type")
        if stats["by_type"]:
            st.bar_chart(stats["by_type"])
        else:
            st.info("No completed documents yet.")
    with col_b:
        st.subheader("By status")
        st.bar_chart(stats["by_status"])
    st.subheader("Recent documents")
    st.dataframe(
        [
            {"File": d["filename"], "Type": d.get("doc_type") or "—",
             "Status": d["status"], "Pages": d.get("num_pages") or "—",
             "Seconds": d.get("processing_seconds") or "—"}
            for d in stats["recent"]
        ],
        use_container_width=True, hide_index=True,
    )
