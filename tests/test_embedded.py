"""Smoke test for the standalone-deployment code path (frontend/embedded.py).

Exercises the same dispatch() calls streamlit_app.py makes when no FastAPI
backend is reachable — this is what runs on Streamlit Community Cloud.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation" / "dataset"


@pytest.fixture(scope="module")
def embedded(tmp_path_factory):
    import os

    tmp = tmp_path_factory.mktemp("embedded-data")
    os.environ["DOCINTEL_DATA_DIR"] = str(tmp)
    os.environ["DOCINTEL_DATABASE_URL"] = f"sqlite:///{tmp}/test.db"

    for mod in list(sys.modules):
        if mod.startswith("backend") or mod == "embedded":
            del sys.modules[mod]

    if not (DATASET / "manifest.json").exists():
        subprocess.run(
            [sys.executable, str(ROOT / "evaluation" / "generate_dataset.py")],
            check=True, cwd=ROOT,
        )

    sys.path.insert(0, str(ROOT / "frontend"))
    import embedded as embedded_module

    return embedded_module


def test_upload_result_page_flow(embedded):
    data = (DATASET / "invoice_00.pdf").read_bytes()
    r = embedded.dispatch("POST", "/api/documents/upload",
                          files={"file": ("invoice_00.pdf", data)})
    assert r.status_code == 202
    doc_id = r.json()["id"]

    r = embedded.dispatch("GET", f"/api/documents/{doc_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "completed"
    assert r.json()["doc_type"] == "invoice"

    r = embedded.dispatch("GET", f"/api/documents/{doc_id}/result")
    assert r.status_code == 200
    assert "total_amount" in r.json()["fields"]

    r = embedded.dispatch("GET", f"/api/documents/{doc_id}/page/1")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 0


def test_rejects_unsupported_type(embedded):
    r = embedded.dispatch("POST", "/api/documents/upload",
                          files={"file": ("notes.txt", b"hello")})
    assert r.status_code == 415


def test_query_and_dashboard(embedded):
    data = (DATASET / "contract_00.pdf").read_bytes()
    embedded.dispatch("POST", "/api/documents/upload",
                      files={"file": ("contract_00.pdf", data)})

    r = embedded.dispatch("POST", "/api/query",
                          json={"question": "What are the termination clauses?"})
    assert r.status_code == 200
    assert r.json()["answer"]
    assert r.json()["sources"]

    r = embedded.dispatch("GET", "/api/dashboard/stats")
    assert r.status_code == 200
    assert r.json()["total_documents"] >= 2


def test_unknown_route_404s(embedded):
    r = embedded.dispatch("GET", "/api/nonsense")
    assert r.status_code == 404
