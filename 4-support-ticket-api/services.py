from __future__ import annotations

import json
import os
import pickle
import random
import hashlib
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import faiss
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ROUTING_VECTORIZER: Any = None
ROUTING_MODEL: Any = None
LABEL_ENCODER: Any = None
SEMANTIC_MODEL: Any = None
SEARCH_INDEX: Any = None
SEARCH_DATA: pd.DataFrame | None = None
MODELS_LOADED = False
ROUTING_MODELS_LOADED = False
SEMANTIC_SEARCH_LOADED = False

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
API_LOCAL_DATASET_PATH = BASE_DIR / "data" / "sample_dataset.csv"
API_LOCAL_MODEL_DIR = BASE_DIR / "models"
# Mirror synthetic datasets here so they appear alongside the bundled samples
DATASET_PREVIEW_DIR = ROOT_DIR / "1-support-ticket-dataset" / "data"
API_LOCAL_SEARCH_INDEX_PATH = API_LOCAL_MODEL_DIR / "search_index.pkl"

ROUTING_MODEL_DIRS = [
    API_LOCAL_MODEL_DIR,
    ROOT_DIR / "2-support-ticket-routing-ml" / "models",
    ROOT_DIR / "support-ticket-routing-ml" / "models",
]
SEMANTIC_SEARCH_DIRS = [
    BASE_DIR,
    ROOT_DIR / "3-support-ticket-semantic-search",
    ROOT_DIR / "support-ticket-semantic-search",
]
SEARCH_INDEX_PATHS = [
    API_LOCAL_SEARCH_INDEX_PATH,
    ROOT_DIR / "3-support-ticket-semantic-search" / "models" / "search_index.pkl",
    ROOT_DIR / "support-ticket-semantic-search" / "models" / "search_index.pkl",
]
DATASET_PATHS = [
    ROOT_DIR / "1-support-ticket-dataset" / "data" / "sample_dataset.csv",
    ROOT_DIR / "support-ticket-dataset" / "data" / "sample_dataset.csv",
    ROOT_DIR / "3-support-ticket-semantic-search" / "data" / "sample_dataset.csv",
    ROOT_DIR / "support-ticket-semantic-search" / "data" / "sample_dataset.csv",
    ROOT_DIR / "data" / "sample_dataset.csv",
    API_LOCAL_DATASET_PATH,
]
DATASET_GENERATOR_SCRIPT_PATHS = [
    ROOT_DIR / "1-support-ticket-dataset" / "generator" / "generate_dataset.py",
    ROOT_DIR / "support-ticket-dataset" / "generator" / "generate_dataset.py",
    BASE_DIR / "generator" / "generate_dataset.py",
]

GENERATOR_PRODUCTS = {
    "Engine Control Module": ["Fuel Injector", "Turbocharger", "Cooling Pump"],
    "Transmission System": ["Gearbox", "Clutch", "Hydraulic Pump"],
    "Electrical System": ["Battery", "Wiring Harness", "Sensors"],
    "Brake System": ["Brake Pads", "Hydraulic Line", "ABS Sensor"],
}
GENERATOR_FAILURE_MODES = ["Leak", "Overheating", "Failure", "Noise", "Short Circuit", "Wear"]
GENERATOR_SEVERITY_LEVELS = ["Low", "Medium", "High", "Critical"]
GENERATOR_REGIONS = ["EU", "North America", "Asia", "South America"]
GENERATOR_CUSTOMER_TYPES = ["Fleet Operator", "Independent Owner", "Dealer", "Service Partner"]
GENERATOR_CHANNELS = ["Dealer Portal", "Email", "Phone", "Monitoring System"]
GENERATOR_STATUS_OPTIONS = ["New", "Assigned", "In Progress", "Awaiting Parts", "Escalated", "Resolved", "Closed"]
GENERATOR_SUB_STATUS_OPTIONS = ["Investigation", "Waiting on Customer", "Parts Ordered", "Action Required", "Completed", "Reopened"]
GENERATOR_COUNTRIES = ["USA", "Germany", "China", "Brazil", "France", "Canada"]
GENERATOR_DEPARTMENTS = ["Field Service", "Customer Support", "Engineering", "Quality", "Sales", "Operations"]
GENERATOR_DEALERS = [
    {"dealer_id": "D-1001", "dealer_name": "Northway Dealer", "dealer_country": "USA", "dealer_state": "CA", "dealer_city": "San Jose"},
    {"dealer_id": "D-1002", "dealer_name": "EuroAuto Dealer", "dealer_country": "Germany", "dealer_state": "Bavaria", "dealer_city": "Munich"},
    {"dealer_id": "D-1003", "dealer_name": "AsiaMotor Dealer", "dealer_country": "China", "dealer_state": "Guangdong", "dealer_city": "Shenzhen"},
    {"dealer_id": "D-1004", "dealer_name": "AmeCar Dealer", "dealer_country": "Brazil", "dealer_state": "SP", "dealer_city": "Sao Paulo"},
    {"dealer_id": "D-1005", "dealer_name": "CanService Dealer", "dealer_country": "Canada", "dealer_state": "ON", "dealer_city": "Toronto"},
]
GENERATOR_SR_TYPES = ["Technical Issue", "Warranty Claim", "Maintenance Request", "Safety Concern", "Performance Issue", "Recall Related"]
GENERATOR_SR_AREAS = ["Engine", "Transmission", "Electrical", "Brakes", "Suspension", "Body/Chassis"]
GENERATOR_ERROR_CODES = ["E-1001", "E-1002", "E-1003", "E-1004", "E-1005", "W-2001", "W-2002", "W-2003", "F-3001", "F-3002"]
GENERATOR_MILEAGE_UNITS = ["km", "miles"]
GENERATOR_ROUTING_RULES = {
    "Fuel Injector": "Powertrain Diagnostics",
    "Turbocharger": "Turbo Systems Team",
    "Cooling Pump": "Cooling Systems",
    "Gearbox": "Transmission Engineering",
    "Clutch": "Transmission Engineering",
    "Hydraulic Pump": "Hydraulics Team",
    "Battery": "Electrical Systems",
    "Wiring Harness": "Electrical Systems",
    "Sensors": "Electronics Diagnostics",
    "Brake Pads": "Brake Systems",
    "Hydraulic Line": "Brake Systems",
    "ABS Sensor": "Electronics Diagnostics",
}

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
    "creation_date": "creation_date",
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

DEFAULT_REQUIRED_INTERNAL_COLUMNS: list[str] = ["ticket_id", "description"]
SUGGEST_PROMPT_PATH = BASE_DIR / "prompts" / "suggest_response.txt"


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
    # Keep API runtime deterministic by always using the local canonical dataset path.
    return API_LOCAL_DATASET_PATH


def _dataset_metadata_path() -> Path:
    return API_LOCAL_DATASET_PATH.parent / "dataset_metadata.json"


def _sanitize_dataset_name(dataset_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", dataset_name.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or "synthetic-dataset"


def _write_named_dataset_snapshot(dataset_path: Path, dataset_name: str | None) -> str | None:
    if not dataset_name:
        return None
    safe_name = _sanitize_dataset_name(dataset_name)
    snapshots_dir = API_LOCAL_DATASET_PATH.parent / "named_datasets"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshots_dir / f"{safe_name}.csv"
    snapshot_path.write_bytes(dataset_path.read_bytes())
    return str(snapshot_path)


def _save_dataset_metadata(
    dataset_path: Path,
    row_count: int,
    source: str,
    dataset_name: str | None = None,
    snapshot_path: str | None = None,
) -> None:
    payload = {
        "dataset_name": (dataset_name or dataset_path.stem),
        "dataset_source": source,
        "dataset_path": str(dataset_path.resolve()),
        "row_count": int(row_count),
        "saved_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "snapshot_path": snapshot_path,
    }
    _dataset_metadata_path().parent.mkdir(parents=True, exist_ok=True)
    _dataset_metadata_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_dataset_metadata() -> dict[str, Any] | None:
    metadata_path = _dataset_metadata_path()
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def list_named_snapshots() -> list[dict[str, Any]]:
    """Return metadata for every named snapshot saved under data/named_datasets/."""
    snapshots_dir = API_LOCAL_DATASET_PATH.parent / "named_datasets"
    if not snapshots_dir.is_dir():
        return []
    results: list[dict[str, Any]] = []
    for csv_file in sorted(snapshots_dir.glob("*.csv")):
        try:
            row_count = sum(1 for _ in csv_file.open(encoding="utf-8")) - 1  # excluding header
        except Exception:
            row_count = -1
        results.append({"name": csv_file.stem, "path": str(csv_file), "row_count": max(row_count, 0)})
    return results


def load_named_snapshot(name: str) -> int:
    """Activate a named snapshot as the current working dataset.

    Copies the snapshot CSV over to the canonical dataset path, writes SQLite,
    and updates dataset metadata. Returns the row count.
    """
    import shutil

    safe_name = _sanitize_dataset_name(name)
    snapshots_dir = API_LOCAL_DATASET_PATH.parent / "named_datasets"
    snapshot_path = snapshots_dir / f"{safe_name}.csv"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"No snapshot found for name '{safe_name}'")

    dest = get_dataset_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snapshot_path, dest)

    df = pd.read_csv(dest)
    _write_dataset_to_sqlite(df)
    _save_dataset_metadata(
        dest,
        row_count=int(df.shape[0]),
        source="synthetic",
        dataset_name=safe_name,
        snapshot_path=str(snapshot_path),
    )
    return int(df.shape[0])


def get_dataset_status() -> dict[str, Any]:
    dataset_path = get_dataset_path()
    exists = dataset_path.exists()
    row_count = 0
    routing_capable = False
    metadata = get_dataset_metadata() or {}
    if exists:
        try:
            df = pd.read_csv(dataset_path)
            row_count = int(df.shape[0])
            # Normalize public-alias column names before checking so datasets
            # saved with e.g. "route_team" are correctly detected as routing-capable.
            df_norm = _normalize_dataset_columns(df)
            routing_capable = "assigned_team" in df_norm.columns
        except Exception:
            row_count = 0
    return {
        "exists": exists,
        "row_count": row_count,
        "path": str(dataset_path),
        "routing_capable": routing_capable,
        "dataset_name": metadata.get("dataset_name"),
        "dataset_source": metadata.get("dataset_source"),
    }


def get_dataset_rows(limit: int = 5000) -> list[dict[str, Any]]:
    dataset_path = get_dataset_path()
    if not dataset_path.exists():
        raise FileNotFoundError("Dataset file not found.")

    capped_limit = max(1, min(int(limit), 5000))
    df = pd.read_csv(dataset_path, nrows=capped_limit)
    if "creation_date" not in df.columns and "created_date" in df.columns:
        df["creation_date"] = df["created_date"]
    return df.to_dict(orient="records")


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


def _sqlite_db_path() -> Path:
    return API_LOCAL_DATASET_PATH.parent / "tickets.db"


def _write_dataset_to_sqlite(df: pd.DataFrame) -> None:
    import sqlite3
    db_path = _sqlite_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        df.to_sql("tickets", conn, if_exists="replace", index=False)
    finally:
        conn.close()


def execute_sql_query(sql: str, limit: int = 500) -> dict[str, Any]:
    import sqlite3
    db_path = _sqlite_db_path()
    if not db_path.exists():
        raise FileNotFoundError("SQL database not found. Generate or upload a dataset first.")
    clean = sql.strip()
    if not clean.upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")
    if "LIMIT" not in clean.upper():
        clean = f"{clean} LIMIT {limit}"
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(clean)
        columns = [d[0] for d in cursor.description]
        rows = [list(r) for r in cursor.fetchall()]
        return {"columns": columns, "rows": rows, "row_count": len(rows)}
    finally:
        conn.close()


def save_dataset_file(file_bytes: bytes, dataset_name: str | None = None) -> tuple[Path, int, list[str]]:
    dataset_path = get_dataset_path()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(BytesIO(file_bytes))
    df = _normalize_dataset_columns(df)
    df.to_csv(dataset_path, index=False)
    _write_dataset_to_sqlite(df)
    snapshot_path = _write_named_dataset_snapshot(dataset_path, dataset_name)
    _save_dataset_metadata(
        dataset_path,
        row_count=int(df.shape[0]),
        source="upload",
        dataset_name=dataset_name,
        snapshot_path=snapshot_path,
    )
    return dataset_path, int(df.shape[0]), list(df.columns)


def _find_dataset_generator_script() -> Path | None:
    for candidate in DATASET_GENERATOR_SCRIPT_PATHS:
        if candidate.is_file():
            return candidate
    return None


def _get_internal_to_public_columns() -> dict[str, str]:
    return {internal_name: public_name for public_name, internal_name in PUBLIC_TO_INTERNAL_COLUMNS.items()}


def _generate_vin() -> str:
    chars = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
    return "".join(random.choice(chars) for _ in range(17))


def _generate_chassis_number() -> str:
    return f"CH{random.randint(100000, 999999)}"


def _rand_datetime_within(days_back: int = 730) -> datetime:
    now = datetime.now(tz=timezone.utc)
    # Bias toward recent activity by sampling age with a power curve.
    age_fraction = random.random() ** 1.8
    age_days = int(age_fraction * days_back)
    date_point = now - timedelta(days=age_days)
    random_seconds = random.randint(0, 86399)
    return date_point.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(seconds=random_seconds)


def _typo_word(word: str) -> str:
    if len(word) < 4:
        return word
    idx = random.randint(0, len(word) - 2)
    return word[:idx] + word[idx + 1] + word[idx] + word[idx + 2 :]


def _add_noise(text: str) -> str:
    rules = [
        lambda value: value.replace("please", "pls"),
        lambda value: value.replace("vehicle", "veh"),
        lambda value: value.replace("diagnostic", "diag"),
        lambda value: value.replace("information", "info"),
        lambda value: value.replace("issue", "iss"),
        lambda value: value.replace("service", "svc"),
    ]
    if random.random() < 0.35:
        for func in random.sample(rules, k=random.randint(1, 2)):
            text = func(text)
    if random.random() < 0.3:
        words = text.split()
        candidate_indexes = [i for i, word in enumerate(words) if len(word) > 3]
        if candidate_indexes:
            selected_index = random.choice(candidate_indexes)
            words[selected_index] = _typo_word(words[selected_index])
            text = " ".join(words)
    if random.random() < 0.25:
        text = text.replace(".", "")
    if random.random() < 0.2:
        suffixes = [" pls", " asap", "Need help.", "Urgent", "FYI", "WIP"]
        text = text + random.choice(suffixes)
    return text


def _failure_phrase(failure_mode: str) -> str:
    variants = {
        "Leak": ["leak", "fluid loss", "drip", "seepage", "leaking"],
        "Overheating": ["overheating", "running hot", "high temp", "thermal issue"],
        "Failure": ["failure", "stopped working", "not responding", "fault"],
        "Noise": ["noise", "rattle", "knocking", "clunking", "unusual sound"],
        "Short Circuit": ["short circuit", "electrical fault", "spark", "power surge"],
        "Wear": ["wear", "worn", "degraded", "abrasion", "wear out"],
    }
    return random.choice(variants.get(failure_mode, [failure_mode.lower()]))


def _random_fragment() -> str:
    fragments = [
        "repro after warm-up.",
        "happens intermittently.",
        "driver says it started after service.",
        "occurs on cold start.",
        "no obvious warning light.",
        "happens during idling.",
        "customer says it is getting worse.",
        "fault appears after long run.",
        "needs inspection.",
        "check asap.",
    ]
    return random.choice(fragments)


def _generate_description(product: str, component: str, failure_mode: str) -> str:
    failure_phrase = _failure_phrase(failure_mode)
    templates = [
        f"Customer reports {failure_phrase} at the {component.lower()} of the {product.lower()}.",
        f"{product} {component} has {failure_phrase}, unclear if related to recent service.",
        f"{component} seems to have {failure_phrase}; driver says it started after startup.",
        f"Urgent: {failure_phrase} observed in {component}, maybe the {product.lower()}.",
        f"{component} has been {failure_phrase} during operation. {random.choice(['Please check', 'Need review', 'Urgent review'])}.",
        f"Rpt: {product} {component} showing {failure_phrase} after long run.",
        f"There is {failure_phrase} on the {component}. {random.choice(['Still open', 'Needs parts', 'High priority'])}.",
        f"{component} appears {failure_phrase}. {random.choice(['No error light', 'Some hesitation', 'Intermittent fault'])}.",
        f"Review {product} {component}: possible {failure_phrase} and reduced performance.",
        f"Found {failure_phrase} from {component}. {random.choice(['No clear root cause', 'Possible sensor issue', 'Looks like wear'])}.",
        f"{failure_phrase.capitalize()} in {component} of the {product}. {random.choice(['Need fix asap', 'Pls advise', 'Customer waiting'])}.",
        f"{product} {component} has {failure_phrase}; note the unusual behavior.",
        f"{failure_phrase.capitalize()} affecting {component} on {product}. {random.choice(['Driver report', 'Customer complaint', 'Tech notes'])}.",
    ]
    description = random.choice(templates)
    if random.random() < 0.4:
        description = description + " " + _random_fragment()
    return _add_noise(description).strip()


def _generate_ticket(days_back: int = 730) -> dict[str, Any]:
    product = random.choice(list(GENERATOR_PRODUCTS.keys()))
    component = random.choice(GENERATOR_PRODUCTS[product])
    failure_mode = random.choice(GENERATOR_FAILURE_MODES)
    status = random.choice(GENERATOR_STATUS_OPTIONS)
    sub_status = random.choice(GENERATOR_SUB_STATUS_OPTIONS)
    created_at = _rand_datetime_within(days_back=days_back)
    close_at = None
    if status in {"Resolved", "Closed"}:
        close_at = created_at + timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
    dealer = random.choice(GENERATOR_DEALERS)

    return {
        "ticket_id": str(uuid.uuid4()),
        "product": product,
        "component": component,
        "failure_mode": failure_mode,
        "severity": random.choice(GENERATOR_SEVERITY_LEVELS),
        "severity_cur": random.choice(GENERATOR_SEVERITY_LEVELS),
        "region": random.choice(GENERATOR_REGIONS),
        "customer_type": random.choice(GENERATOR_CUSTOMER_TYPES),
        "ticket_channel": random.choice(GENERATOR_CHANNELS),
        "description": _generate_description(product, component, failure_mode),
        "assigned_team": GENERATOR_ROUTING_RULES[component],
        "status": status,
        "sub_status": sub_status,
        "status_line": "Closed" if status in {"Resolved", "Closed"} else "Open",
        "creation_datetime": created_at.isoformat(sep=" ", timespec="seconds"),
        "creation_date": created_at.date().isoformat(),
        "close_datetime": close_at.isoformat(sep=" ", timespec="seconds") if close_at else "",
        "close_date": close_at.date().isoformat() if close_at else "",
        "creator_country": random.choice(GENERATOR_COUNTRIES),
        "creator_department": random.choice(GENERATOR_DEPARTMENTS),
        "owner_country": random.choice(GENERATOR_COUNTRIES),
        "owner_department": random.choice(GENERATOR_DEPARTMENTS),
        "dealer_id": dealer["dealer_id"],
        "dealer_name": dealer["dealer_name"],
        "dealer_country": dealer["dealer_country"],
        "dealer_state": dealer["dealer_state"],
        "dealer_city": dealer["dealer_city"],
        "time_to_close_seconds": int((close_at - created_at).total_seconds()) if close_at else "",
        "first_queued_seconds": random.randint(0, 7200),
        "vin_number": _generate_vin(),
        "chassis_number": _generate_chassis_number(),
        "mileage": random.randint(0, 500000),
        "mileage_unit": random.choice(GENERATOR_MILEAGE_UNITS),
        "error_code": random.choice(GENERATOR_ERROR_CODES),
        "sr_type": random.choice(GENERATOR_SR_TYPES),
        "sr_area": random.choice(GENERATOR_SR_AREAS),
        "product_id": f"P-{random.randint(1000, 9999)}",
        "chassis_serie": f"S-{random.randint(100, 999)}",
    }


def _generate_dataset_frame(size: int = 50000, days_back: int = 730) -> pd.DataFrame:
    tickets = [_generate_ticket(days_back=days_back) for _ in range(size)]
    return pd.DataFrame(tickets)


def _resolve_output_columns(columns: list[str], internal_to_public: dict[str, str]) -> list[str]:
    output_columns: list[str] = []
    seen_outputs: set[str] = set()
    for column in columns:
        normalized = str(column).strip()
        if not normalized:
            continue
        resolved: str | None = None
        if normalized in internal_to_public.values():
            resolved = normalized
        elif normalized in internal_to_public:
            resolved = internal_to_public[normalized]
        else:
            raise ValueError(f"Unknown column for selection: {normalized}")
        if resolved in seen_outputs:
            continue
        output_columns.append(resolved)
        seen_outputs.add(resolved)
    if not output_columns:
        raise ValueError("No valid columns selected for output")
    return output_columns


def _resolve_optional_output_column(
    column: str | None,
    internal_name: str,
    internal_to_public: dict[str, str],
) -> str | None:
    if column is None:
        return internal_to_public.get(internal_name)

    normalized = str(column).strip()
    if not normalized:
        return None
    return _resolve_output_columns([normalized], internal_to_public)[0]


def _prepare_generated_dataset_for_training(
    dataset: pd.DataFrame,
    internal_to_public: dict[str, str],
    include_columns: list[str] | None = None,
    description_column: str | None = None,
    assigned_team_column: str | None = None,
    ticket_id_column: str | None = None,
) -> pd.DataFrame:
    resolved_description_column = _resolve_optional_output_column(
        description_column,
        "description",
        internal_to_public,
    )
    if resolved_description_column is None:
        raise ValueError("Synthetic dataset generation requires a mapped description column.")

    resolved_assigned_team_column = _resolve_optional_output_column(
        assigned_team_column,
        "assigned_team",
        internal_to_public,
    )
    resolved_ticket_id_column = _resolve_optional_output_column(
        ticket_id_column,
        "ticket_id",
        internal_to_public,
    )

    if include_columns:
        selected_columns = _resolve_output_columns(include_columns, internal_to_public)
        required_selected = [resolved_description_column]
        if resolved_assigned_team_column is not None:
            required_selected.append(resolved_assigned_team_column)
        if resolved_ticket_id_column is not None:
            required_selected.append(resolved_ticket_id_column)

        missing_selected = [column for column in required_selected if column not in set(selected_columns)]
        if missing_selected:
            raise ValueError(
                "Selected columns must include mapped columns: " + ", ".join(missing_selected)
            )

        missing_columns = [column for column in selected_columns if column not in dataset.columns]
        if missing_columns:
            raise ValueError(f"Configured output columns are not present in generated dataset: {missing_columns}")
        dataset = dataset[selected_columns].copy()
    else:
        dataset = dataset.copy()

    if resolved_description_column not in dataset.columns:
        raise ValueError(
            f"Mapped description column '{resolved_description_column}' is not present in generated dataset."
        )
    if resolved_assigned_team_column is not None and resolved_assigned_team_column not in dataset.columns:
        raise ValueError(
            f"Mapped assigned_team column '{resolved_assigned_team_column}' is not present in generated dataset."
        )
    if resolved_ticket_id_column is not None and resolved_ticket_id_column not in dataset.columns:
        raise ValueError(
            f"Mapped ticket_id column '{resolved_ticket_id_column}' is not present in generated dataset."
        )

    rename_map: dict[str, str] = {
        resolved_description_column: "description",
    }
    if resolved_assigned_team_column is not None:
        rename_map[resolved_assigned_team_column] = "assigned_team"
    if resolved_ticket_id_column is not None:
        rename_map[resolved_ticket_id_column] = "ticket_id"

    for source_col, target_col in rename_map.items():
        if source_col != target_col and target_col in dataset.columns:
            raise ValueError(
                f"Cannot map '{source_col}' to '{target_col}' because '{target_col}' already exists in selected columns. "
                f"Deselect '{target_col}' or choose another mapping."
            )

    dataset = dataset.rename(columns=rename_map)

    if "ticket_id" not in dataset.columns:
        dataset.insert(0, "ticket_id", [f"synthetic-{i}" for i in range(len(dataset))])

    return dataset


def _build_raw_synthetic_dataset(size: int = 50000) -> pd.DataFrame:
    internal_to_public = _get_internal_to_public_columns()
    dataset = _generate_dataset_frame(size=size)
    dataset = dataset.rename(columns=internal_to_public)

    if "created_date" in dataset.columns and "creation_date" not in dataset.columns:
        dataset["creation_date"] = dataset["created_date"]

    return dataset


def _generate_synthetic_dataset_in_process(
    output_path: Path,
    size: int,
    include_columns: list[str] | None,
    description_column: str | None,
    assigned_team_column: str | None,
    ticket_id_column: str | None,
) -> int:
    internal_to_public = _get_internal_to_public_columns()
    dataset = _build_raw_synthetic_dataset(size=size)
    dataset = _prepare_generated_dataset_for_training(
        dataset,
        internal_to_public,
        include_columns=include_columns,
        description_column=description_column,
        assigned_team_column=assigned_team_column,
        ticket_id_column=ticket_id_column,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return int(dataset.shape[0])


def _copy_to_dataset_preview_dir(source_path: Path, dataset_name: str | None) -> None:
    """Copy a generated dataset CSV into the shared dataset preview folder."""
    import shutil

    try:
        DATASET_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        stem = _sanitize_dataset_name(dataset_name) if dataset_name else "synthetic_dataset"
        dest = DATASET_PREVIEW_DIR / f"{stem}.csv"
        shutil.copy2(source_path, dest)
    except Exception:
        pass  # Never block training because of an optional mirror copy


def generate_synthetic_dataset(
    size: int = 50000,
    include_columns: list[str] | None = None,
    description_column: str | None = None,
    assigned_team_column: str | None = None,
    ticket_id_column: str | None = None,
    dataset_name: str | None = None,
) -> tuple[Path, int]:
    script_path = _find_dataset_generator_script()
    output_path = get_dataset_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    internal_to_public = _get_internal_to_public_columns()
    if script_path is not None:
        command = [sys.executable, str(script_path), "--size", str(size), "--output", str(output_path)]
        if include_columns:
            selected = [str(col).strip() for col in include_columns if str(col).strip()]
            if selected:
                command.extend(["--include-columns", ",".join(selected)])
        try:
            process = subprocess.run(
                command,
                cwd=str(script_path.parent) if script_path.parent.exists() else None,
                capture_output=True,
                text=True,
                check=False,
            )
            if process.returncode != 0:
                raise RuntimeError(
                    f"Dataset generator failed: {process.returncode}\nstdout:{process.stdout}\nstderr:{process.stderr}"
                )
            df = pd.read_csv(output_path)
            df = _prepare_generated_dataset_for_training(
                df,
                internal_to_public,
                include_columns=include_columns,
                description_column=description_column,
                assigned_team_column=assigned_team_column,
                ticket_id_column=ticket_id_column,
            )
            df.to_csv(output_path, index=False)
            _write_dataset_to_sqlite(df)
            snapshot_path = _write_named_dataset_snapshot(output_path, dataset_name)
            _save_dataset_metadata(
                output_path,
                row_count=int(df.shape[0]),
                source="synthetic",
                dataset_name=dataset_name,
                snapshot_path=snapshot_path,
            )
            _copy_to_dataset_preview_dir(output_path, dataset_name)
            return output_path, int(df.shape[0])
        except FileNotFoundError:
            # API-only deployments may not include the external dataset generator path.
            pass

    row_count = _generate_synthetic_dataset_in_process(
        output_path,
        size=size,
        include_columns=include_columns,
        description_column=description_column,
        assigned_team_column=assigned_team_column,
        ticket_id_column=ticket_id_column,
    )
    df = pd.read_csv(output_path)
    _write_dataset_to_sqlite(df)
    snapshot_path = _write_named_dataset_snapshot(output_path, dataset_name)
    _save_dataset_metadata(
        output_path,
        row_count=int(df.shape[0]),
        source="synthetic",
        dataset_name=dataset_name,
        snapshot_path=snapshot_path,
    )
    _copy_to_dataset_preview_dir(output_path, dataset_name)
    return output_path, row_count


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
    global SEMANTIC_SEARCH_LOADED, MODELS_LOADED

    dataset_path = get_dataset_path()
    if not dataset_path.exists():
        raise FileNotFoundError("Dataset file not found for FAISS index build.")
    df = pd.read_csv(dataset_path)
    vector_count = _create_search_index_from_dataframe(df)
    API_LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(API_LOCAL_SEARCH_INDEX_PATH, "wb") as fp:
        pickle.dump({"index": SEARCH_INDEX, "data": SEARCH_DATA}, fp)
    SEMANTIC_SEARCH_LOADED = True
    MODELS_LOADED = ROUTING_MODELS_LOADED and SEMANTIC_SEARCH_LOADED
    return vector_count


def routing_artifacts_available() -> bool:
    try:
        _resolve_routing_model_dir()
        return True
    except FileNotFoundError:
        return False


def train_routing_models(dataset_path: Path | None = None) -> Path:
    if dataset_path is None:
        dataset_path = get_dataset_path()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found for routing model training: {dataset_path}")

    df = _normalize_dataset_columns(pd.read_csv(dataset_path))
    required = {"description", "assigned_team"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Dataset is missing required routing columns: {missing}. "
            "Routing model training is only possible when an 'assigned_team' column is present. "
            "Upload a dataset that includes team assignments, or use the synthetic dataset generator."
        )

    text_values = df["description"].fillna("").astype(str)
    label_values = df["assigned_team"].fillna("Unassigned").astype(str)
    if text_values.empty or label_values.nunique() < 2:
        raise ValueError("Dataset must contain routing training samples for at least two teams")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(label_values)

    vectorizer = TfidfVectorizer(max_features=5000)
    x_features = vectorizer.fit_transform(text_values)

    # Hold out 20% for evaluation metrics saved alongside the model.
    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    eval_model = LogisticRegression(max_iter=1000)
    eval_model.fit(x_train, y_train)
    y_pred = eval_model.predict(x_test)

    class_names = [str(c) for c in label_encoder.classes_]
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names)))).tolist()
    acc = float(accuracy_score(y_test, y_pred))

    feature_names = vectorizer.get_feature_names_out()
    importance = np.abs(eval_model.coef_).sum(axis=0)
    top_idx = np.argsort(importance)[-30:][::-1]
    top_features = [
        {"word": str(feature_names[i]), "importance": float(importance[i])}
        for i in top_idx
    ]

    # Train production model on full dataset.
    model = LogisticRegression(max_iter=1000)
    model.fit(x_features, y_encoded)

    API_LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, API_LOCAL_MODEL_DIR / "vectorizer.pkl")
    joblib.dump(model, API_LOCAL_MODEL_DIR / "lr_model.pkl")
    joblib.dump(label_encoder, API_LOCAL_MODEL_DIR / "label_encoder.pkl")

    import json as _json
    perf = {
        "accuracy": acc,
        "class_names": class_names,
        "confusion_matrix": cm,
        "feature_importance": top_features,
    }
    (API_LOCAL_MODEL_DIR / "model_performance.json").write_text(
        _json.dumps(perf, indent=2), encoding="utf-8"
    )

    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    dataset_meta = get_dataset_metadata() or {}
    metadata = {
        "dataset_path": str(dataset_path.resolve()),
        "dataset_name": dataset_meta.get("dataset_name"),
        "dataset_source": dataset_meta.get("dataset_source"),
        "snapshot_path": dataset_meta.get("snapshot_path"),
        "row_count": int(df.shape[0]),
        "dataset_sha256": dataset_sha256,
        "trained_at_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    (API_LOCAL_MODEL_DIR / "routing_training_metadata.json").write_text(
        _json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return API_LOCAL_MODEL_DIR


def get_routing_training_metadata() -> dict[str, Any] | None:
    metadata_path = API_LOCAL_MODEL_DIR / "routing_training_metadata.json"
    if not metadata_path.exists():
        return None

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    return payload


def get_model_performance() -> dict[str, Any]:
    perf_path = API_LOCAL_MODEL_DIR / "model_performance.json"
    if not perf_path.exists():
        raise FileNotFoundError("Model performance data not found. Train the routing models first.")
    import json as _json
    return _json.loads(perf_path.read_text(encoding="utf-8"))


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
    global ROUTING_VECTORIZER, ROUTING_MODEL, LABEL_ENCODER, ROUTING_MODELS_LOADED

    selected_model_dir: Path | None = None
    for candidate in ROUTING_MODEL_DIRS:
        vectorizer_path = candidate / "vectorizer.pkl"
        model_path = candidate / "lr_model.pkl"
        encoder_path = candidate / "label_encoder.pkl"
        vectorizer_exists = vectorizer_path.exists()
        model_exists = model_path.exists()
        encoder_exists = encoder_path.exists()
        print(
            "[routing-load] "
            f"candidate={candidate.resolve()} "
            f"vectorizer={vectorizer_exists} model={model_exists} label_encoder={encoder_exists}"
        )
        if vectorizer_exists and model_exists and encoder_exists:
            selected_model_dir = candidate
            break

    if selected_model_dir is None:
        raise FileNotFoundError(
            "Routing model directory not found. "
            f"Checked candidates: {[str(path.resolve()) for path in ROUTING_MODEL_DIRS]}"
        )

    model_dir = selected_model_dir
    ROUTING_VECTORIZER = joblib.load(model_dir / "vectorizer.pkl")
    ROUTING_MODEL = joblib.load(model_dir / "lr_model.pkl")
    LABEL_ENCODER = joblib.load(model_dir / "label_encoder.pkl")
    ROUTING_MODELS_LOADED = True


def load_semantic_search_resources() -> None:
    global SEMANTIC_MODEL, SEARCH_INDEX, SEARCH_DATA, SEMANTIC_SEARCH_LOADED

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
            SEMANTIC_SEARCH_LOADED = True
        return

    data_path = get_dataset_path()
    SEARCH_DATA = pd.read_csv(data_path)
    _create_search_index_from_dataframe(SEARCH_DATA)
    SEMANTIC_SEARCH_LOADED = True


def load_models() -> None:
    global MODELS_LOADED, ROUTING_MODELS_LOADED, SEMANTIC_SEARCH_LOADED
    ROUTING_MODELS_LOADED = False
    SEMANTIC_SEARCH_LOADED = False
    try:
        load_routing_resources()
        load_semantic_search_resources()
        MODELS_LOADED = True
    except Exception as exc:
        MODELS_LOADED = False
        raise RuntimeError(f"Failed to load models: {exc}") from exc


def sync_model_load_state() -> bool:
    global MODELS_LOADED
    MODELS_LOADED = ROUTING_MODELS_LOADED and SEMANTIC_SEARCH_LOADED
    return MODELS_LOADED


def ensure_routing_models_loaded() -> bool:
    """Best-effort reload of routing artifacts from disk if not already loaded."""
    global ROUTING_MODELS_LOADED
    if ROUTING_MODELS_LOADED:
        return True
    try:
        load_routing_resources()
        return ROUTING_MODELS_LOADED
    except Exception:
        ROUTING_MODELS_LOADED = False
        return False


def get_model_status(auto_recover: bool = False) -> dict[str, bool]:
    if auto_recover:
        ensure_routing_models_loaded()
    sync_model_load_state()
    return {
        "routing_loaded": ROUTING_MODELS_LOADED,
        "semantic_loaded": SEMANTIC_SEARCH_LOADED,
        "all_loaded": MODELS_LOADED,
    }


def clear_all_state() -> dict[str, Any]:
    """Clear local dataset/model/index artifacts and reset loaded runtime state."""
    global ROUTING_VECTORIZER, ROUTING_MODEL, LABEL_ENCODER
    global SEMANTIC_MODEL, SEARCH_INDEX, SEARCH_DATA
    global MODELS_LOADED, ROUTING_MODELS_LOADED, SEMANTIC_SEARCH_LOADED

    removed: list[str] = []

    # Remove API-local dataset and metadata artifacts.
    candidate_files = [
        API_LOCAL_DATASET_PATH,
        _dataset_metadata_path(),
        _sqlite_db_path(),
        API_LOCAL_SEARCH_INDEX_PATH,
        API_LOCAL_MODEL_DIR / "vectorizer.pkl",
        API_LOCAL_MODEL_DIR / "lr_model.pkl",
        API_LOCAL_MODEL_DIR / "label_encoder.pkl",
        API_LOCAL_MODEL_DIR / "model_performance.json",
        API_LOCAL_MODEL_DIR / "routing_training_metadata.json",
    ]

    for file_path in candidate_files:
        try:
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                removed.append(str(file_path))
        except Exception:
            pass

    # Remove named snapshots and mirrored preview copies.
    snapshots_dir = API_LOCAL_DATASET_PATH.parent / "named_datasets"
    snapshot_stems: list[str] = []
    try:
        if snapshots_dir.exists() and snapshots_dir.is_dir():
            snapshot_stems = [p.stem for p in snapshots_dir.glob("*.csv")]
            shutil.rmtree(snapshots_dir)
            removed.append(str(snapshots_dir))
    except Exception:
        pass

    for stem in snapshot_stems:
        try:
            preview_file = DATASET_PREVIEW_DIR / f"{stem}.csv"
            if preview_file.exists() and preview_file.is_file():
                preview_file.unlink()
                removed.append(str(preview_file))
        except Exception:
            pass

    # Reset in-memory runtime state.
    ROUTING_VECTORIZER = None
    ROUTING_MODEL = None
    LABEL_ENCODER = None
    SEMANTIC_MODEL = None
    SEARCH_INDEX = None
    SEARCH_DATA = None
    ROUTING_MODELS_LOADED = False
    SEMANTIC_SEARCH_LOADED = False
    MODELS_LOADED = False

    return {
        "success": True,
        "removed_count": len(removed),
        "removed_paths": removed,
    }


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


def _load_suggest_prompt_template() -> str:
    if SUGGEST_PROMPT_PATH.exists():
        return SUGGEST_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "You are a senior support specialist. Use the similar historical tickets as context to draft a concise, "
        "actionable response for the support agent.\n\n"
        "Current ticket:\n{ticket_description}\n\n"
        "Similar historical tickets:\n{context_tickets}\n\n"
        "Write a suggested response with:\n"
        "1) Diagnosis summary\n"
        "2) Immediate steps\n"
        "3) Follow-up recommendation\n"
    )


def _format_context_tickets(tickets: list[dict[str, Any]]) -> str:
    if not tickets:
        return "No similar historical tickets were found."

    lines: list[str] = []
    for idx, ticket in enumerate(tickets, start=1):
        lines.append(
            f"{idx}. ticket_id={ticket.get('ticket_id', '-')}; "
            f"assigned_team={ticket.get('assigned_team', '-')}; "
            f"similarity_score={float(ticket.get('similarity_score', 0.0)):.4f}; "
            f"description={ticket.get('description', '')}"
        )
    return "\n".join(lines)


class _HFInferenceClientWrapper:
    """Thin wrapper around huggingface_hub.InferenceClient with an invoke() interface."""

    def __init__(self, client: Any, repo_id: str) -> None:
        self._client = client
        self._repo_id = repo_id

    def invoke(self, prompt: str) -> str:
        content = prompt.replace("[INST]", "").replace("[/INST]", "").strip()
        chat_error: str | None = None

        # Prefer chat-completions when the model supports it.
        try:
            response = self._client.chat_completion(
                messages=[{"role": "user", "content": content}],
                model=self._repo_id,
                max_tokens=256,
                temperature=0.2,
            )
            return str(response.choices[0].message.content).strip()
        except Exception as exc:
            chat_error = str(exc)

        # Fall back to text generation for instruct models that are not exposed as chat models.
        try:
            generated = self._client.text_generation(
                prompt=content,
                model=self._repo_id,
                max_new_tokens=256,
                temperature=0.2,
                return_full_text=False,
            )
            return str(generated).strip()
        except Exception as exc:
            generation_error = str(exc)
            raise RuntimeError(
                "HuggingFace inference failed for both chat and text-generation paths. "
                f"chat_error={chat_error}; generation_error={generation_error}"
            ) from exc


def _create_llm_client(hf_token: str | None = None) -> tuple[Any | None, bool, str | None]:
    token = (hf_token or "").strip() or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
    if not token:
        return None, False, "HUGGINGFACEHUB_API_TOKEN is not configured"

    provider = os.getenv("HUGGINGFACE_PROVIDER", "auto").strip() or "auto"
    repo_id = os.getenv("HUGGINGFACE_REPO_ID", "Qwen/Qwen2.5-7B-Instruct").strip() or "Qwen/Qwen2.5-7B-Instruct"

    # Use a single explicit inference path so provider/model behavior is deterministic.
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(provider=provider, api_key=token)
        return _HFInferenceClientWrapper(client, repo_id), True, None
    except Exception as exc:
        return None, False, f"Failed to initialize HuggingFace InferenceClient for provider '{provider}': {exc}"


def _build_local_response_draft(ticket_description: str, context_tickets: list[dict[str, Any]]) -> str:
    """Create a useful local fallback response when no external LLM is configured."""
    issue_summary = ticket_description.strip().replace("\n", " ")
    if len(issue_summary) > 220:
        issue_summary = issue_summary[:217].rstrip() + "..."

    likely_team = "Support Team"
    if context_tickets:
        first_team = str(context_tickets[0].get("assigned_team", "")).strip()
        if first_team:
            likely_team = first_team

    next_steps: list[str] = [
        "Confirm the exact symptoms, scope, and first observed timestamp.",
        "Collect logs/screenshots and any recent changes before the issue started.",
        "Apply standard diagnostics and provide an ETA for the next update.",
    ]

    if context_tickets:
        similar_ref = []
        for t in context_tickets[:2]:
            tid = str(t.get("ticket_id", "")).strip()
            team = str(t.get("assigned_team", "")).strip()
            if tid and team:
                similar_ref.append(f"{tid} ({team})")
            elif tid:
                similar_ref.append(tid)
        reference_text = ", ".join(similar_ref)
    else:
        reference_text = "No close historical matches were found"

    return (
        "Thanks for reporting this issue.\n\n"
        f"Summary: {issue_summary}\n"
        f"Recommended owner: {likely_team}\n\n"
        "Proposed next steps:\n"
        f"1. {next_steps[0]}\n"
        f"2. {next_steps[1]}\n"
        f"3. {next_steps[2]}\n\n"
        f"Related ticket signals: {reference_text}."
    )


def suggest_response(ticket_description: str, hf_token: str | None = None) -> dict[str, Any]:
    description = (ticket_description or "").strip()
    if not description:
        raise ValueError("Ticket description must not be empty")

    context_tickets = search_similar_tickets(description, top_k=3)
    llm, llm_available, unavailable_reason = _create_llm_client(hf_token)

    if not llm_available or llm is None:
        local_draft = _build_local_response_draft(description, context_tickets)
        return {
            "suggested_response": local_draft,
            "context_tickets": context_tickets,
            "llm_available": False,
            "llm_error": unavailable_reason or "Using local fallback draft",
        }

    prompt_template_text = _load_suggest_prompt_template()
    prompt = prompt_template_text.format(
        ticket_description=description,
        context_tickets=_format_context_tickets(context_tickets),
    )

    try:
        llm_output = llm.invoke(prompt)
        if hasattr(llm_output, "content"):
            suggested_response = str(llm_output.content).strip()
        else:
            suggested_response = str(llm_output).strip()

        if not suggested_response:
            suggested_response = "No suggestion returned by the language model."

        return {
            "suggested_response": suggested_response,
            "context_tickets": context_tickets,
            "llm_available": True,
            "llm_error": None,
        }
    except Exception as invoke_exc:
        error_detail = str(invoke_exc)
        local_draft = _build_local_response_draft(description, context_tickets)
        return {
            "suggested_response": local_draft,
            "context_tickets": context_tickets,
            "llm_available": False,
            "llm_error": f"External LLM unavailable, using local fallback draft. Reason: {error_detail}",
        }
