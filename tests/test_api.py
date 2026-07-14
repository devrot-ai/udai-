"""API integration tests using the FastAPI TestClient.

Background processing runs synchronously inside the TestClient, so the
document is fully processed by the time upload returns.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation" / "dataset"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os

    tmp = tmp_path_factory.mktemp("api-data")
    os.environ["DOCINTEL_DATA_DIR"] = str(tmp)
    os.environ["DOCINTEL_DATABASE_URL"] = f"sqlite:///{tmp}/test.db"

    # reset cached modules so config picks up the env vars
    for mod in list(sys.modules):
        if mod.startswith("backend"):
            del sys.modules[mod]

    if not (DATASET / "manifest.json").exists():
        subprocess.run(
            [sys.executable, str(ROOT / "evaluation" / "generate_dataset.py")],
            check=True, cwd=ROOT,
        )

    from fastapi.testclient import TestClient

    from backend.app.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "rules" in body["extractors"]


def test_upload_and_result_flow(client):
    pdf = (DATASET / "invoice_00.pdf").read_bytes()
    r = client.post(
        "/api/documents/upload",
        files={"file": ("invoice_00.pdf", pdf, "application/pdf")},
    )
    assert r.status_code == 202
    doc_id = r.json()["id"]

    r = client.get(f"/api/documents/{doc_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "completed"
    assert r.json()["doc_type"] == "invoice"

    r = client.get(f"/api/documents/{doc_id}/result")
    assert r.status_code == 200
    result = r.json()
    assert result["document_type"] == "invoice"
    assert "total_amount" in result["fields"]
    field = result["fields"]["total_amount"]
    assert 0 <= field["confidence"] <= 1
    assert field["source"]

    r = client.get(f"/api/documents/{doc_id}/page/1")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"


def test_upload_rejects_unknown_type(client):
    r = client.post(
        "/api/documents/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 415


def test_query_endpoint(client):
    pdf = (DATASET / "contract_00.pdf").read_bytes()
    client.post(
        "/api/documents/upload",
        files={"file": ("contract_00.pdf", pdf, "application/pdf")},
    )
    r = client.post(
        "/api/query",
        json={"question": "What are the termination clauses?", "top_k": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert body["sources"]
    assert body["answer_mode"] in ("llm", "extractive")


def test_dashboard_stats(client):
    r = client.get("/api/dashboard/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total_documents"] >= 2
    assert body["by_status"].get("completed", 0) >= 2
    assert body["total_fields_extracted"] > 0


def test_document_not_found(client):
    assert client.get("/api/documents/nope").status_code == 404
    assert client.get("/api/documents/nope/result").status_code == 404
