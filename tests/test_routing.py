"""Tests for the routing model training and prediction pipeline."""
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "4-support-ticket-api"))

import services


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> Path:
    """Write a minimal CSV dataset for fast training tests."""
    rows = []
    teams = ["Team Alpha", "Team Beta", "Team Gamma"]
    descriptions = {
        "Team Alpha": ["fuel injector leak detected", "fuel pump failure critical issue"],
        "Team Beta": ["gearbox noise unusual sound", "transmission slipping badly"],
        "Team Gamma": ["battery short circuit", "wiring harness electrical fault"],
    }
    for team, descs in descriptions.items():
        for desc in descs * 5:  # 10 rows per team = 30 total
            rows.append({"ticket_id": "t1", "description": desc, "assigned_team": team})
    df = pd.DataFrame(rows)
    path = tmp_path / "sample_dataset.csv"
    df.to_csv(path, index=False)
    return path


def test_train_routing_models_creates_artifacts(tiny_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(services, "API_LOCAL_MODEL_DIR", tmp_path / "models")
    result_dir = services.train_routing_models(dataset_path=tiny_dataset)
    assert (result_dir / "vectorizer.pkl").exists()
    assert (result_dir / "lr_model.pkl").exists()
    assert (result_dir / "label_encoder.pkl").exists()
    assert (result_dir / "model_performance.json").exists()


def test_train_routing_models_saves_performance_json(tiny_dataset, tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(services, "API_LOCAL_MODEL_DIR", tmp_path / "models")
    services.train_routing_models(dataset_path=tiny_dataset)
    perf = json.loads((tmp_path / "models" / "model_performance.json").read_text())
    assert "accuracy" in perf
    assert 0.0 <= perf["accuracy"] <= 1.0
    assert "class_names" in perf
    assert "confusion_matrix" in perf
    assert "feature_importance" in perf
    assert len(perf["feature_importance"]) > 0


def test_predict_route_after_training(tiny_dataset, tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    monkeypatch.setattr(services, "API_LOCAL_MODEL_DIR", model_dir)
    monkeypatch.setattr(services, "ROUTING_MODEL_DIRS", [model_dir])
    services.train_routing_models(dataset_path=tiny_dataset)
    services.load_routing_resources()

    team, confidence, all_scores = services.predict_route("fuel injector failure urgent")
    assert isinstance(team, str)
    assert 0.0 <= confidence <= 1.0
    assert isinstance(all_scores, dict)
    assert len(all_scores) == 3


def test_predict_route_returns_known_team(tiny_dataset, tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    monkeypatch.setattr(services, "API_LOCAL_MODEL_DIR", model_dir)
    monkeypatch.setattr(services, "ROUTING_MODEL_DIRS", [model_dir])
    services.train_routing_models(dataset_path=tiny_dataset)
    services.load_routing_resources()

    team, _, _ = services.predict_route("battery short circuit wiring fault")
    assert team in {"Team Alpha", "Team Beta", "Team Gamma"}


def test_get_model_performance_after_training(tiny_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(services, "API_LOCAL_MODEL_DIR", tmp_path / "models")
    services.train_routing_models(dataset_path=tiny_dataset)
    perf = services.get_model_performance()
    assert perf["accuracy"] >= 0.0
    assert len(perf["class_names"]) == 3


def test_get_model_performance_raises_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "API_LOCAL_MODEL_DIR", tmp_path / "empty_models")
    with pytest.raises(FileNotFoundError):
        services.get_model_performance()
