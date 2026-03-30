from __future__ import annotations

import json
import pickle
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import faiss
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ROUTING_VECTORIZER: Any = None
ROUTING_MODEL: Any = None
LABEL_ENCODER: Any = None
SEMANTIC_MODEL: Any = None
SEARCH_INDEX: Any = None
SEARCH_DATA: pd.DataFrame | None = None
MODELS_LOADED = False

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

ROUTING_MODEL_DIRS = [
    ROOT_DIR / "2-support-ticket-routing-ml" / "models",
    ROOT_DIR / "support-ticket-routing-ml" / "models",
]
SEMANTIC_SEARCH_DIRS = [
    ROOT_DIR / "3-support-ticket-semantic-search",
    ROOT_DIR / "support-ticket-semantic-search",
]
SEARCH_INDEX_PATHS = [
    ROOT_DIR / "3-support-ticket-semantic-search" / "models" / "search_index.pkl",
    ROOT_DIR / "support-ticket-semantic-search" / "models" / "search_index.pkl",
]
DATASET_PATHS = [
    ROOT_DIR / "1-support-ticket-dataset" / "data" / "sample_dataset.csv",
    ROOT_DIR / "support-ticket-dataset" / "data" / "sample_dataset.csv",
    ROOT_DIR / "3-support-ticket-semantic-search" / "data" / "sample_dataset.csv",
    ROOT_DIR / "support-ticket-semantic-search" / "data" / "sample_dataset.csv",
]

DEFAULT_PUBLIC_TO_INTERNAL_COLUMNS: dict[str, str] = {
    "ticket_uuid": "ticket_id",
    "product_family": "product",
    "component_name": "component",
    "fault_mode": "failure_mode",
    "severity_level": "severity",
    "current_severity": "severity_cur",
    "geo_region": "region",
    "customer_segment": "customer_type",
    "source_channel": "ticket_channel",
    "issue_description": "description",
    "route_team": "assigned_team",
    "ticket_status": "status",
    "ticket_substatus": "sub_status",
    "status_state": "status_line",
    "created_timestamp": "creation_datetime",
    "created_date": "creation_date",
    "closed_timestamp": "close_datetime",
    "closed_date": "close_date",
    "reporter_country": "creator_country",
    "reporter_department": "creator_department",
    "handler_country": "owner_country",
    "handler_department": "owner_department",
    "dealer_code": "dealer_id",
    "dealer_label": "dealer_name",
    "dealer_country_code": "dealer_country",
    "dealer_state_code": "dealer_state",
    "dealer_city_name": "dealer_city",
    "resolution_seconds": "time_to_close_seconds",
    "first_queue_seconds": "first_queued_seconds",
    "vehicle_vin": "vin_number",
    "vehicle_chassis": "chassis_number",
    "odometer_reading": "mileage",
    "odometer_unit": "mileage_unit",
    "fault_code": "error_code",
    "service_request_type": "sr_type",
    "service_request_area": "sr_area",
    "product_code": "product_id",
    "chassis_series": "chassis_serie",
}

DEFAULT_REQUIRED_INTERNAL_COLUMNS: list[str] = ["ticket_id", "description", "assigned_team"]


def _alias_config_path() -> Path:
    return ROOT_DIR / "column_aliases.json"


def load_alias_config() -> tuple[dict[str, str], list[str]]:
    config_path = _alias_config_path()
    if not config_path.exists():
        return DEFAULT_PUBLIC_TO_INTERNAL_COLUMNS, DEFAULT_REQUIRED_INTERNAL_COLUMNS
    try:
        with open(config_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except Exception:
        return DEFAULT_PUBLIC_TO_INTERNAL_COLUMNS, DEFAULT_REQUIRED_INTERNAL_COLUMNS

    internal_to_public = payload.get("internal_to_public", {})
    required_internal_columns = payload.get("required_internal_columns", DEFAULT_REQUIRED_INTERNAL_COLUMNS)

    if not isinstance(internal_to_public, dict):
        return DEFAULT_PUBLIC_TO_INTERNAL_COLUMNS, DEFAULT_REQUIRED_INTERNAL_COLUMNS

    public_to_internal = {
        str(public_name): str(internal_name)
        for internal_name, public_name in internal_to_public.items()
        if isinstance(internal_name, str)
        and isinstance(public_name, str)
        and internal_name.strip()
        and public_name.strip()
    }
    if not public_to_internal:
        public_to_internal = DEFAULT_PUBLIC_TO_INTERNAL_COLUMNS

    if not isinstance(required_internal_columns, list):
        required_internal_columns = DEFAULT_REQUIRED_INTERNAL_COLUMNS
    required = [str(col) for col in required_internal_columns if isinstance(col, str) and col.strip()]
    if not required:
        required = DEFAULT_REQUIRED_INTERNAL_COLUMNS

    return public_to_internal, required


PUBLIC_TO_INTERNAL_COLUMNS, REQUIRED_INTERNAL_COLUMNS = load_alias_config()


def _find_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No valid path found in candidates: {[str(p) for p in candidates]}")


def get_dataset_path() -> Path:
    try:
        return _find_existing_path(DATASET_PATHS)
    except FileNotFoundError:
        return ROOT_DIR / "1-support-ticket-dataset" / "data" / "sample_dataset.csv"


def get_dataset_status() -> dict[str, Any]:
    dataset_path = get_dataset_path()
    exists = dataset_path.exists()
    row_count = 0
    if exists:
        try:
            row_count = int(pd.read_csv(dataset_path).shape[0])
        except Exception:
            row_count = 0
    return {
        "exists": exists,
        "row_count": row_count,
        "path": str(dataset_path),
    }


def _normalize_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        col: PUBLIC_TO_INTERNAL_COLUMNS[col]
        for col in df.columns
        if col in PUBLIC_TO_INTERNAL_COLUMNS
    }
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def get_routing_model_files() -> list[str]:
    for candidate in ROUTING_MODEL_DIRS:
        if candidate.exists():
            return [path.name for path in sorted(candidate.glob("*.pkl"))]
    return []


def get_faiss_index_status() -> dict[str, Any]:
    if SEARCH_INDEX is not None:
        return {"exists": True, "vector_count": int(SEARCH_INDEX.ntotal)}
    for index_path in SEARCH_INDEX_PATHS:
        if index_path.exists():
            try:
                with open(index_path, "rb") as fp:
                    saved = pickle.load(fp)
                    index = saved.get("index")
                    if index is not None:
                        return {"exists": True, "vector_count": int(index.ntotal)}
            except Exception:
                continue
    return {"exists": False, "vector_count": 0}


def save_dataset_file(file_bytes: bytes) -> tuple[Path, int, list[str]]:
    dataset_path = get_dataset_path()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(BytesIO(file_bytes))
    df = _normalize_dataset_columns(df)
    df.to_csv(dataset_path, index=False)
    return dataset_path, int(df.shape[0]), list(df.columns)


def generate_synthetic_dataset(size: int = 50000, include_columns: list[str] | None = None) -> tuple[Path, int]:
    script_path = ROOT_DIR / "1-support-ticket-dataset" / "generator" / "generate_dataset.py"
    output_path = ROOT_DIR / "1-support-ticket-dataset" / "data" / "sample_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(script_path), "--size", str(size), "--output", str(output_path)]
    if include_columns:
        selected = [str(col).strip() for col in include_columns if str(col).strip()]
        if selected:
            command.extend(["--include-columns", ",".join(selected)])
    process = subprocess.run(
        command,
        cwd=script_path.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Dataset generator failed: {process.returncode}\nstdout:{process.stdout}\nstderr:{process.stderr}"
        )
    df = pd.read_csv(output_path)
    return output_path, int(df.shape[0])


def _create_search_index_from_dataframe(df: pd.DataFrame) -> int:
    global SEMANTIC_MODEL, SEARCH_INDEX, SEARCH_DATA
    df = _normalize_dataset_columns(df)
    if "description" not in df.columns:
        raise ValueError("Semantic dataset must contain a 'description' column.")
    SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    descriptions = df["description"].fillna("").tolist()
    embeddings = SEMANTIC_MODEL.encode(descriptions, convert_to_numpy=True, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    SEARCH_INDEX = faiss.IndexFlatIP(embeddings.shape[1])
    SEARCH_INDEX.add(embeddings)
    SEARCH_DATA = df
    return int(SEARCH_INDEX.ntotal)


def build_faiss_index() -> int:
    dataset_path = get_dataset_path()
    if not dataset_path.exists():
        raise FileNotFoundError("Dataset file not found for FAISS index build.")
    df = pd.read_csv(dataset_path)
    return _create_search_index_from_dataframe(df)


def _resolve_routing_model_dir() -> Path:
    for candidate in ROUTING_MODEL_DIRS:
        if (candidate / "vectorizer.pkl").exists() and (candidate / "lr_model.pkl").exists() and (candidate / "label_encoder.pkl").exists():
            return candidate
    raise FileNotFoundError(
        "Routing model directory not found. Expected ../2-support-ticket-routing-ml/models or ../support-ticket-routing-ml/models"
    )


def _resolve_semantic_base_dir() -> Path:
    for candidate in SEMANTIC_SEARCH_DIRS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Semantic search directory not found. Expected ../3-support-ticket-semantic-search or ../support-ticket-semantic-search"
    )


def load_routing_resources() -> None:
    global ROUTING_VECTORIZER, ROUTING_MODEL, LABEL_ENCODER

    model_dir = _resolve_routing_model_dir()
    ROUTING_VECTORIZER = joblib.load(model_dir / "vectorizer.pkl")
    ROUTING_MODEL = joblib.load(model_dir / "lr_model.pkl")
    LABEL_ENCODER = joblib.load(model_dir / "label_encoder.pkl")


def load_semantic_search_resources() -> None:
    global SEMANTIC_MODEL, SEARCH_INDEX, SEARCH_DATA

    try:
        index_path = _find_existing_path(SEARCH_INDEX_PATHS)
    except FileNotFoundError:
        index_path = None

    if index_path is not None:
        with open(index_path, "rb") as fp:
            saved = pickle.load(fp)
            SEARCH_INDEX = saved["index"]
            SEARCH_DATA = saved["data"]
            SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return

    data_path = _find_existing_path(DATASET_PATHS)
    SEARCH_DATA = pd.read_csv(data_path)
    _create_search_index_from_dataframe(SEARCH_DATA)


def load_models() -> None:
    global MODELS_LOADED
    try:
        load_routing_resources()
        load_semantic_search_resources()
        MODELS_LOADED = True
    except Exception as exc:
        MODELS_LOADED = False
        raise RuntimeError(f"Failed to load models: {exc}") from exc


def predict_route(description: str) -> tuple[str, float, dict[str, float]]:
    if ROUTING_VECTORIZER is None or ROUTING_MODEL is None or LABEL_ENCODER is None:
        raise RuntimeError("Routing models are not loaded")

    text = description or ""
    input_vector = ROUTING_VECTORIZER.transform([text])
    probabilities = ROUTING_MODEL.predict_proba(input_vector)[0]
    classes = ROUTING_MODEL.classes_
    labels = LABEL_ENCODER.inverse_transform(classes.astype(int))
    score_map = {str(label): float(prob) for label, prob in zip(labels, probabilities)}
    best_index = int(np.argmax(probabilities))
    return str(labels[best_index]), float(probabilities[best_index]), score_map


def search_similar_tickets(description: str, top_k: int = 5) -> list[dict[str, Any]]:
    if SEARCH_INDEX is None or SEARCH_DATA is None or SEMANTIC_MODEL is None:
        raise RuntimeError("Semantic search resources are not loaded")

    query = (description or "").strip()
    if query == "":
        raise ValueError("Search description must not be empty")

    query_embedding = SEMANTIC_MODEL.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding.astype(np.float32)
    faiss.normalize_L2(query_embedding)

    top_k = min(top_k, len(SEARCH_DATA))
    scores, indices = SEARCH_INDEX.search(query_embedding, top_k)

    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(SEARCH_DATA):
            continue
        ticket = SEARCH_DATA.iloc[idx]
        results.append(
            {
                "ticket_id": str(ticket.get("ticket_id", idx)),
                "description": str(ticket.get("description", "")),
                "assigned_team": str(ticket.get("assigned_team", "")),
                "similarity_score": float(score),
            }
        )

    return results
