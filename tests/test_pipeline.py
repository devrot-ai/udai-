"""Unit tests for the extraction pipeline (no heavy models required)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation" / "dataset"


@pytest.fixture(scope="session", autouse=True)
def dataset():
    if not (DATASET / "manifest.json").exists():
        subprocess.run(
            [sys.executable, str(ROOT / "evaluation" / "generate_dataset.py")],
            check=True, cwd=ROOT,
        )
    return DATASET


def _sample(prefix: str) -> Path:
    return DATASET / f"{prefix}_00.pdf"


class TestClassifier:
    def test_invoice(self):
        from backend.app.pipeline.classifier import classify
        from backend.app.pipeline.ingest import ingest

        doc = ingest(str(_sample("invoice")))
        doc_type, conf = classify(doc.full_text)
        assert doc_type == "invoice"
        assert conf > 0.5

    def test_all_types(self):
        from backend.app.pipeline.classifier import classify
        from backend.app.pipeline.ingest import ingest

        expected = {"invoice": "invoice", "contract": "contract",
                    "financial": "financial_statement", "court": "court_document"}
        for prefix, doc_type in expected.items():
            doc = ingest(str(_sample(prefix)))
            detected, _ = classify(doc.full_text)
            assert detected == doc_type, f"{prefix}: got {detected}"

    def test_empty_text(self):
        from backend.app.pipeline.classifier import classify

        assert classify("")[0] == "other"


class TestMLClassifier:
    """Supervised TF-IDF + LogisticRegression tier (train_classifier.py)."""

    @pytest.fixture(scope="class", autouse=True)
    def model(self):
        from backend.app.ml import doc_classifier

        if not doc_classifier.MODEL_PATH.exists():
            subprocess.run(
                [sys.executable, str(ROOT / "training" / "train_classifier.py"),
                 "--samples-per-class", "80"],
                check=True, cwd=ROOT,
            )
        doc_classifier._CACHE.clear()
        assert doc_classifier.available()

    def test_predicts_all_types_on_pdfs(self):
        from backend.app.ml import doc_classifier
        from backend.app.pipeline.ingest import ingest

        expected = {"invoice": "invoice", "contract": "contract",
                    "financial": "financial_statement", "court": "court_document"}
        for prefix, doc_type in expected.items():
            doc = ingest(str(_sample(prefix)))
            label, prob = doc_classifier.predict(doc.full_text)
            assert label == doc_type, f"{prefix}: got {label}"
            assert 0 < prob <= 1

    def test_other_class_negative(self):
        from backend.app.ml import doc_classifier

        label, _ = doc_classifier.predict(
            "INTERNAL MEMORANDUM\nTo: All Staff\nSubject: Office relocation\n"
            "Please note the office will be closed on Friday for maintenance."
        )
        assert label == "other"

    def test_ensemble_agreement_boosts_and_reports_method(self):
        from backend.app.pipeline.classifier import classify_detailed
        from backend.app.pipeline.ingest import ingest

        doc = ingest(str(_sample("invoice")))
        d = classify_detailed(doc.full_text)
        assert d["type"] == "invoice"
        assert d["method"] == "ml+keywords"
        assert d["confidence"] > 0.8

    def test_ensemble_falls_back_without_model(self, monkeypatch):
        from backend.app.ml import doc_classifier
        from backend.app.pipeline.classifier import classify_detailed

        monkeypatch.setattr(doc_classifier, "predict", lambda text: None)
        d = classify_detailed("Invoice Number: INV-1 Total Due: $10.00 Bill To: X")
        assert d["type"] == "invoice"
        assert d["method"] in ("keywords", "zero-shot")


class TestIngest:
    def test_native_pdf_has_words(self):
        from backend.app.pipeline.ingest import ingest

        doc = ingest(str(_sample("invoice")))
        assert not doc.is_scanned
        assert doc.pages[0].words
        w = doc.pages[0].words[0]
        assert 0 <= w.x0 <= w.x1 <= 1
        assert 0 <= w.y0 <= w.y1 <= 1

    def test_unsupported_extension(self):
        from backend.app.pipeline.ingest import ingest

        with pytest.raises(ValueError):
            ingest("file.docx")


class TestRulesExtractor:
    def test_invoice_fields(self):
        from backend.app.pipeline.extractors.rules import RulesExtractor
        from backend.app.pipeline.ingest import ingest

        doc = ingest(str(_sample("invoice")))
        fields = {f.name: f for f in RulesExtractor().extract(doc, "invoice")}
        truth = json.loads(
            (DATASET / "ground_truth" / "invoice_00.json").read_text()
        )["fields"]
        assert fields["invoice_number"].value == truth["invoice_number"]
        assert truth["total_amount"].replace(",", "") in fields["total_amount"].value.replace(",", "")
        assert all(0 <= f.confidence <= 1 for f in fields.values())

    def test_bboxes_attached(self):
        from backend.app.pipeline.extractors.rules import RulesExtractor
        from backend.app.pipeline.ingest import ingest

        doc = ingest(str(_sample("invoice")))
        fields = RulesExtractor().extract(doc, "invoice")
        located = [f for f in fields if f.bbox]
        assert located, "expected at least one field with a bounding box"
        for f in located:
            x0, y0, x1, y1 = f.bbox
            assert 0 <= x0 <= x1 <= 1 and 0 <= y0 <= y1 <= 1


class TestHybrid:
    def test_full_pipeline_invoice(self):
        from backend.app.pipeline.hybrid import process_document

        result = process_document(str(_sample("invoice")))
        assert result["document_type"] == "invoice"
        assert "total_amount" in result["fields"]
        assert result["fields"]["total_amount"]["confidence"] > 0.5
        assert result["meta"]["extractors_used"]
        assert result["tables"], "invoice line items should be detected as a table"

    def test_merge_prefers_confident_source(self):
        from backend.app.pipeline.extractors.base import ExtractedField
        from backend.app.pipeline.hybrid import merge_fields

        merged = merge_fields([
            ExtractedField("total_amount", "$100.00", 0.9, "rules"),
            ExtractedField("total_amount", "$999.00", 0.4, "ocr_spacy"),
        ])
        assert merged["total_amount"].value == "$100.00"

    def test_merge_agreement_boosts_confidence(self):
        from backend.app.pipeline.extractors.base import ExtractedField
        from backend.app.pipeline.hybrid import merge_fields

        merged = merge_fields([
            ExtractedField("total_amount", "$100.00", 0.8, "rules"),
            ExtractedField("total_amount", "$100.00", 0.7, "donut"),
        ])
        assert merged["total_amount"].confidence > 0.8
        assert "rules" in merged["total_amount"].source


class TestFieldShortcut:
    """The QA layer should answer well-known intents (parties, totals,
    dates...) from the document's already-extracted structured fields
    rather than free-text search, which can match a sentence that merely
    mentions the right word without actually answering the question."""

    @pytest.fixture()
    def scoped_doc(self, tmp_path, monkeypatch):
        import json
        import sys

        monkeypatch.setenv("DOCINTEL_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("DOCINTEL_DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
        for mod in list(sys.modules):
            if mod.startswith("backend"):
                del sys.modules[mod]

        from backend.app.database import Document, SessionLocal, init_db

        init_db()
        db = SessionLocal()
        try:
            doc = Document(
                id="doc1", filename="contract.pdf", content_type="application/pdf",
                stored_path="/tmp/x.pdf", status="completed",
                result_json=json.dumps({"fields": {
                    "party_1": {"value": "Meridian Software Corp", "confidence": 0.9,
                               "source": "rules", "page": 1},
                    "party_2": {"value": "Pinnacle Healthcare Group", "confidence": 0.9,
                               "source": "rules", "page": 1},
                }}),
            )
            db.add(doc)
            db.commit()
        finally:
            db.close()
        return "doc1"

    def test_answers_from_structured_field_not_free_text(self, scoped_doc, monkeypatch):
        from backend.app.rag import qa

        # even if free-text search would return a misleading match, the
        # field shortcut should win for a party question scoped to one doc
        monkeypatch.setattr(
            qa.store, "search",
            lambda *a, **k: [{"text": "NOW, THEREFORE, the parties agree as follows:",
                              "score": 0.9, "meta": {}}],
        )
        result = qa.answer("Who are the parties involved?", document_id=scoped_doc)
        assert result["answer"] == \
            "Party 1: Meridian Software Corp; Party 2: Pinnacle Healthcare Group"

    def test_no_shortcut_without_document_scope(self, scoped_doc):
        from backend.app.rag.qa import _field_shortcut

        assert _field_shortcut("Who are the parties involved?", None) is None

    def test_total_revenue_does_not_match_total_assets(self, tmp_path, monkeypatch):
        import json
        import sys

        monkeypatch.setenv("DOCINTEL_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("DOCINTEL_DATABASE_URL", f"sqlite:///{tmp_path}/test2.db")
        for mod in list(sys.modules):
            if mod.startswith("backend"):
                del sys.modules[mod]

        from backend.app.database import Document, SessionLocal, init_db
        from backend.app.rag import qa

        init_db()
        db = SessionLocal()
        try:
            db.add(Document(
                id="doc2", filename="financial.pdf", content_type="application/pdf",
                stored_path="/tmp/x.pdf", status="completed",
                result_json=json.dumps({"fields": {
                    "total_revenue": {"value": "$4,500,000.00", "confidence": 0.9,
                                      "source": "rules", "page": 1},
                    "total_assets": {"value": "$13,500,000.00", "confidence": 0.9,
                                     "source": "rules", "page": 1},
                }}),
            ))
            db.commit()
        finally:
            db.close()

        result = qa.answer("What is the total revenue?", document_id="doc2")
        assert "4,500,000" in result["answer"]
        assert "13,500,000" not in result["answer"]


class TestRagFallbackStore:
    def test_index_and_search(self, tmp_path, monkeypatch):
        from backend.app.rag import store as rag_store

        monkeypatch.setattr(rag_store, "_STORE", None)
        tfidf = rag_store._TfidfStore(tmp_path / "store.json")
        monkeypatch.setattr(rag_store, "get_store", lambda: tfidf)

        n = rag_store.index_document(
            "doc1", "contract.pdf",
            ["This agreement may be terminated upon sixty days written notice.",
             "Payment shall be made within thirty days of invoice."],
        )
        assert n >= 2
        hits = rag_store.search("What are the termination clauses?", top_k=2)
        assert hits
        assert "terminated" in hits[0]["text"]

    def test_extractive_answer(self):
        from backend.app.rag.qa import _extractive_answer

        hits = [{"text": "This Agreement may be terminated upon sixty (60) days "
                         "written notice by either party.", "score": 0.8}]
        answer = _extractive_answer("What are the termination clauses?", hits)
        assert "terminated" in answer

    def test_extractive_answer_does_not_glue_unrelated_facts(self):
        """Regression: independently top-scoring but unrelated sentences from
        the same tabular chunk (a total, a due date, a table header) used to
        be concatenated into one garbled answer. Only the single strongest
        sentence should come back."""
        from backend.app.rag.qa import _extractive_answer

        hits = [{
            "text": "Total Due: $3,456.00\nDue Date: February 3, 2024\n"
                    "Description Qty Unit Price Amount",
            "score": 0.5,
        }]
        answer = _extractive_answer("What is the total amount due?", hits)
        assert answer == "Total Due: $3,456.00"

    def test_extractive_answer_skips_all_caps_headings(self):
        from backend.app.rag.qa import _extractive_answer

        hits = [{
            "text": "3. TERM AND TERMINATION\nThis Agreement shall commence on "
                    "the Effective Date. Either party may terminate upon sixty "
                    "(60) days written notice.",
            "score": 0.5,
        }]
        answer = _extractive_answer("What are the termination clauses?", hits)
        assert "TERM AND TERMINATION" not in answer
        assert "Either party may terminate" in answer
