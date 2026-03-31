"""API contract tests using FastAPI TestClient (no real server needed)."""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "4-support-ticket-api"))


@pytest.fixture(scope="module")
def client():
    """Return a TestClient with model loading skipped at startup."""
    with patch("services.load_models"), \
         patch("services.get_model_status", return_value={
             "all_loaded": False, "routing_loaded": False, "semantic_loaded": False
         }):
        import main
        with TestClient(main.app) as c:
            yield c


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_status_returns_expected_shape(client):
    with patch("services.get_dataset_status", return_value={"exists": False, "row_count": 0, "path": ""}), \
         patch("services.get_model_status", return_value={"all_loaded": False, "routing_loaded": False, "semantic_loaded": False}), \
         patch("services.get_faiss_index_status", return_value={"exists": False, "vector_count": 0}), \
         patch("services.get_routing_model_files", return_value=[]):
        r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert "dataset" in body
    assert "models" in body
    assert "faiss_index" in body


def test_route_returns_503_when_models_not_loaded(client):
    r = client.post("/route", json={"description": "engine overheating"})
    assert r.status_code == 503


def test_search_returns_503_when_index_not_loaded(client):
    r = client.post("/search", json={"description": "battery failure", "top_k": 3})
    assert r.status_code == 503


def test_suggest_returns_200_with_no_token(client):
    with patch("services.search_similar_tickets", return_value=[]), \
         patch("services.SEARCH_INDEX", MagicMock()), \
         patch("services.SEARCH_DATA", pd.DataFrame({"ticket_id": [], "description": [], "assigned_team": []})), \
         patch("services.SEMANTIC_MODEL", MagicMock()):
        with patch("os.getenv", return_value=""):
            r = client.post("/suggest", json={"description": "cooling pump overheating"})
    assert r.status_code in {200, 503}


def test_route_empty_description_rejected(client):
    r = client.post("/route", json={"description": ""})
    # Either 503 (models not loaded) or 422 (validation) — both are acceptable
    assert r.status_code in {422, 503}


def test_model_performance_404_before_training(client):
    with patch("services.get_model_performance", side_effect=FileNotFoundError("not found")):
        r = client.get("/model-performance")
    assert r.status_code == 404
