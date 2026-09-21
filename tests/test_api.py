from fastapi.testclient import TestClient

from mmrag.api.app import create_app
from mmrag.config import Settings
from mmrag.runtime import build_services


def test_health_and_home() -> None:
    client = TestClient(create_app(build_services(Settings(_env_file=None))))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    home = client.get("/")
    assert home.status_code == 200
    assert "MM-RAG" in home.text


def test_upload_search_and_query_flow(tmp_path) -> None:  # type: ignore[no-untyped-def]
    del tmp_path
    client = TestClient(create_app(build_services(Settings(_env_file=None))))
    files = {
        "file": (
            "report.md",
            b"# Finance\nRevenue increased in Q3 after expansion.\n",
            "text/markdown",
        ),
    }
    ingest = client.post("/api/v1/ingest/upload", files=files)
    assert ingest.status_code == 200
    assert ingest.json()["chunks"] >= 1
    search = client.post("/api/v1/search", json={"text": "Q3 revenue", "top_k": 3})
    assert search.status_code == 200
    assert search.json()["results"]
    query = client.post("/api/v1/query", json={"text": "How did revenue change?", "top_k": 3})
    payload = query.json()
    assert query.status_code == 200
    assert payload["answer"]["text"]
    stats = client.get("/api/v1/stats")
    assert stats.json()["chunks"] >= 1


def test_rejects_unsupported_upload() -> None:
    client = TestClient(create_app(build_services(Settings(_env_file=None))))
    files = {"file": ("notes.exe", b"binary", "application/octet-stream")}
    response = client.post("/api/v1/ingest/upload", files=files)
    assert response.status_code == 400
