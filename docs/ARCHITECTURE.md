# Architecture

## System overview

```
                        ┌──────────────────────────────────────────────────┐
                        │                 FastAPI backend                  │
   Streamlit UI  ───►   │                                                  │
   (upload /            │  POST /api/documents/upload ──► BackgroundTask   │
    results /           │  GET  /api/documents/{id}/result                 │
    ask / stats)        │  POST /api/query          GET /api/dashboard     │
                        └───────────────┬──────────────────────────────────┘
                                        │
                        ┌───────────────▼──────────────────────────────────┐
                        │            Hybrid extraction pipeline            │
                        │                                                  │
                        │  1. Ingest  (pdfplumber / pypdfium2 / Tesseract) │
                        │  2. Classify (keyword scorer + zero-shot NLI)    │
                        │  3. Extract  ┌ LayoutLMv3 (layout-aware KV)      │
                        │              ├ Donut (OCR-free visual QA)        │
                        │              ├ Rules (domain regex + layout)     │
                        │              └ Tesseract + spaCy (NER fallback)  │
                        │  4. Merge   (source-weighted confidence vote)    │
                        │  5. Tables  (ruled + borderless word-geometry)   │
                        └───────┬──────────────────────────┬───────────────┘
                                │                          │
                     ┌──────────▼─────────┐     ┌──────────▼──────────┐
                     │  SQLite (metadata, │     │  ChromaDB vector    │
                     │  results JSON)     │     │  store (RAG chunks) │
                     └────────────────────┘     └─────────────────────┘
```

## Ingestion (`backend/app/pipeline/ingest.py`)

Every input is normalized to a list of `Page` objects containing plain text
plus **word-level bounding boxes in normalized [0, 1] coordinates**. That
single representation feeds every downstream component:

| Input | Path |
|---|---|
| Native PDF | pdfplumber text + `extract_words()` boxes |
| Scanned PDF (< ~30 chars/page of embedded text) | pypdfium2 rasterization → Tesseract `image_to_data` word boxes |
| Image (PNG/JPEG/TIFF…) | Tesseract directly |

Normalized coordinates mean the same bbox works for LayoutLMv3's 0–1000
coordinate space, the merge step, and the frontend highlight overlay at any
render scale.

## Document type detection (`pipeline/classifier.py`)

A weighted-keyword scorer covers five domain types (invoice, contract,
financial statement, court document, receipt). Confidence combines absolute
evidence with the margin over the runner-up type. When the scorer is
uncertain (< 0.6) *and* transformers is installed, a `bart-large-mnli`
zero-shot classifier breaks the tie. The scorer alone achieves 100% on the
evaluation set; the zero-shot path is a safety net for out-of-domain inputs.

## The hybrid extractor (`pipeline/hybrid.py`)

Each extractor implements a two-method protocol (`available()`, `extract()`),
so the pipeline composes whatever the runtime environment supports:

| Extractor | What it does | Confidence signal |
|---|---|---|
| **LayoutLMv3** (`extractors/layoutlm.py`) | FUNSD-fine-tuned token classification into QUESTION/ANSWER entities over words+boxes+page image; questions paired with the nearest following answer and mapped onto the canonical schema | mean softmax prob. of answer tokens |
| **Donut** (`extractors/donut.py`) | OCR-free DocVQA: one generated answer per schema field, straight from pixels | exp(mean log-prob) of generated tokens |
| **Rules** (`extractors/rules.py`) | Domain regex + layout heuristics per document type; the deterministic backbone | pattern-strength priors, discounted 15% on OCR text |
| **Tesseract + spaCy** (`extractors/ocr_fallback.py`) | Classic NER (ORG/PERSON/DATE/MONEY) mapped to fields — also the evaluation baseline | fixed low priors (0.45–0.55) |

### Merge strategy

Candidates for the same field are ranked by
`confidence × source_weight` (LayoutLMv3 1.0, rules 0.95, Donut 0.9,
spaCy 0.6). When two or more extractors independently agree on the same
normalized value, the winner's confidence gets a +0.1 boost per agreeing
source and provenance is recorded as e.g. `"layoutlmv3+rules"`. This gives
calibrated, explainable per-field scores rather than a single opaque number.

### Graceful degradation

Model loading is lazy and cached; any `ImportError`/inference failure marks
the extractor unavailable for the request rather than failing the job.
`GET /health` reports exactly which extractors are live. The lite install
(no torch) still runs the full API with the rules layer at 100% eval
accuracy on native PDFs.

## Table extraction (`pipeline/tables.py`)

Ruled tables use pdfplumber's line-based detection. Borderless tables (the
common case in invoices and financial statements) use a custom word-geometry
detector: words → lines (y-clustering) → cells (split at horizontal gaps
> 3.5% of page width) → runs of ≥3 consecutive multi-cell lines → columns by
x0 clustering. Output is row-major JSON plus a normalized bbox.

## RAG component (`backend/app/rag/`)

* **Chunking** — LangChain `RecursiveCharacterTextSplitter` (900 chars,
  150 overlap), falling back to a paragraph-aware splitter.
* **Store** — ChromaDB persistent collection with sentence-transformer
  embeddings (`all-MiniLM-L6-v2`). Lite installs fall back to a
  dependency-free persistent TF-IDF cosine store so `/api/query` always
  works. The active backend is reported by `/health`.
* **Answering** — retrieved chunks go to an LLM (when `ANTHROPIC_API_KEY`
  is set) that answers strictly from the provided excerpts with page-level
  citations; otherwise a stemmed-overlap extractive answerer returns the
  most relevant sentences. The response always includes ranked sources.

## Async processing

Uploads return `202 Accepted` immediately; extraction runs in a FastAPI
`BackgroundTask` and status transitions `queued → processing →
completed|failed` are persisted on the document row. Extraction failures are
recorded, RAG-indexing failures are noted in the result metadata without
failing the extraction. For multi-instance deployments the worker call in
`api/documents.py` is a one-line swap for a Celery/RQ enqueue.

## Data model

One `documents` table (SQLite by default, any SQLAlchemy URL via
`DOCINTEL_DATABASE_URL`): file metadata, processing status, detected type,
timing, and the full extraction result as JSON. Vector chunks live in
ChromaDB keyed by `document_id` for scoped queries and deletion.

## Evaluation (`evaluation/`)

`generate_dataset.py` builds 24 synthetic documents (6 each of invoice /
contract / financial statement / court filing) with pixel-rendered PDFs and
field-level ground truth; `--scanned` adds rasterized variants for the OCR
path. `run_eval.py` scores the hybrid pipeline against the classic baseline
(Tesseract + spaCy NER; naive first-match regex when spaCy is absent) using
normalized exact/containment/numeric matching. Results land in
`evaluation/results/{results.json, report.md}`.
