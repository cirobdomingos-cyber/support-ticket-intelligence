from __future__ import annotations

import io
import os
from io import BytesIO
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
NO_TEAM_COLUMN_OPTION = "(no team column - routing will be disabled)"
AUTO_GENERATE_TICKET_ID_OPTION = "(auto-generate ticket_id)"

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

DEFAULT_REQUIRED_INTERNAL_COLUMNS: list[str] = ["ticket_id", "description", "assigned_team"]


def _load_alias_config() -> tuple[dict[str, str], list[str]]:
    return DEFAULT_PUBLIC_TO_INTERNAL_COLUMNS, DEFAULT_REQUIRED_INTERNAL_COLUMNS


PUBLIC_TO_INTERNAL_COLUMNS, REQUIRED_INTERNAL_COLUMNS = _load_alias_config()


def _normalize_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        col: PUBLIC_TO_INTERNAL_COLUMNS[col]
        for col in df.columns
        if col in PUBLIC_TO_INTERNAL_COLUMNS
    }
    if rename_map:
        df = df.rename(columns=rename_map)

    # Alias normalization can map multiple source columns to one target name.
    # Merge duplicates by taking the first non-null value left-to-right.
    if df.columns.duplicated().any():
        deduped_frames: list[pd.Series] = []
        deduped_names: list[str] = []
        seen: set[str] = set()
        for column_name in df.columns:
            if column_name in seen:
                continue
            same_named = df.loc[:, df.columns == column_name]
            if same_named.shape[1] == 1:
                merged_series = same_named.iloc[:, 0]
            else:
                merged_series = same_named.bfill(axis=1).iloc[:, 0]
            deduped_frames.append(merged_series)
            deduped_names.append(column_name)
            seen.add(column_name)
        df = pd.concat(deduped_frames, axis=1)
        df.columns = deduped_names
    return df


def _synthetic_output_options() -> list[str]:
    preferred: dict[str, str] = {}
    for public_name, internal_name in PUBLIC_TO_INTERNAL_COLUMNS.items():
        if internal_name not in preferred:
            preferred[internal_name] = public_name
        # Prefer canonical names that match the internal schema exactly.
        if public_name == internal_name:
            preferred[internal_name] = public_name
        # Prefer creation_date over created_date when both are available.
        if internal_name == "creation_date" and public_name == "creation_date":
            preferred[internal_name] = public_name
    return sorted(set(preferred.values()))


def _default_index(options: list[str], candidates: list[str]) -> int:
    for candidate in candidates:
        if candidate in options:
            return options.index(candidate)
    return 0


def _render_training_column_mapping(
    available_columns: list[str],
    key_prefix: str,
    intro_text: str,
) -> tuple[list[str], str | None, str | None, str | None]:
    st.markdown(intro_text)

    description_column = st.selectbox(
        "Which column contains ticket description text?",
        options=available_columns,
        index=_default_index(available_columns, ["description", "issue_description"]),
        key=f"{key_prefix}_description",
    )
    team_col_options = [NO_TEAM_COLUMN_OPTION] + available_columns
    assigned_team_column = st.selectbox(
        "Which column is the assigned team? (optional - needed for routing model)",
        options=team_col_options,
        index=_default_index(team_col_options, ["assigned_team", "route_team"]),
        key=f"{key_prefix}_assigned_team",
    )
    ticket_id_options = [AUTO_GENERATE_TICKET_ID_OPTION] + available_columns
    ticket_id_column = st.selectbox(
        "Optional: which column should be used as ticket id?",
        options=ticket_id_options,
        index=_default_index(ticket_id_options, ["ticket_id", "ticket_uuid"]),
        key=f"{key_prefix}_ticket_id",
    )

    selected_columns = st.multiselect(
        "Columns to include",
        options=available_columns,
        default=available_columns,
        key=f"{key_prefix}_columns",
    )

    required_selected = [description_column]
    if assigned_team_column and assigned_team_column != NO_TEAM_COLUMN_OPTION:
        required_selected.append(assigned_team_column)
    if ticket_id_column and ticket_id_column != AUTO_GENERATE_TICKET_ID_OPTION:
        required_selected.append(ticket_id_column)
    missing_selected = [col for col in required_selected if col not in set(selected_columns)]
    if missing_selected:
        st.error("Selected columns must include mapped columns: " + ", ".join(missing_selected))

    return selected_columns, description_column, assigned_team_column, ticket_id_column


def load_dataset() -> pd.DataFrame:
    status_payload = call_status()
    dataset_status = status_payload.get("dataset", {})
    if not bool(dataset_status.get("exists")):
        raise FileNotFoundError(
            "Dataset is not available yet. Use Setup & Training to upload or generate a dataset first."
        )

    response = requests.get(f"{API_URL}/dataset", params={"limit": 5000}, timeout=60)
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list):
        raise ValueError("Unexpected dataset payload returned from API")
    if not rows:
        return pd.DataFrame()
    return _normalize_dataset_columns(pd.DataFrame(rows))


@st.cache_data(ttl=300)
def check_health() -> tuple[bool, str]:
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") == "ok":
            return True, payload.get("models_loaded", False)
        return False, False
    except Exception:
        return False, False


def call_route(description: str) -> dict[str, Any]:
    response = requests.post(
        f"{API_URL}/route",
        json={"description": description},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def call_search(description: str, top_k: int) -> list[dict[str, Any]]:
    response = requests.post(
        f"{API_URL}/search",
        json={"description": description, "top_k": top_k},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def call_sql_query(sql: str, limit: int = 500) -> dict[str, Any]:
    response = requests.post(
        f"{API_URL}/query",
        json={"sql": sql, "limit": limit},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def call_model_performance() -> dict[str, Any]:
    response = requests.get(f"{API_URL}/model-performance", timeout=15)
    response.raise_for_status()
    return response.json()


def call_suggest(description: str) -> dict[str, Any]:
    response = requests.post(
        f"{API_URL}/suggest",
        json={"description": description},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Unexpected suggestion payload")
    return payload


def call_status() -> dict[str, Any]:
    response = requests.get(f"{API_URL}/status", timeout=10)
    response.raise_for_status()
    return response.json()


def safe_call_status() -> dict[str, Any]:
    try:
        return call_status()
    except Exception:
        return {}


def upload_dataset_file(file_bytes: bytes, filename: str) -> dict[str, Any]:
    response = requests.post(
        f"{API_URL}/upload-dataset",
        files={"file": (filename, file_bytes, "text/csv")},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def generate_synthetic_dataset(
    include_columns: list[str],
    description_column: str | None = None,
    assigned_team_column: str | None = None,
    ticket_id_column: str | None = None,
    dataset_name: str | None = None,
    size: int = 50000,
) -> dict[str, Any]:
    response = requests.post(
        f"{API_URL}/generate-dataset",
        json={
            "size": size,
            "include_columns": include_columns,
            "description_column": description_column,
            "assigned_team_column": assigned_team_column,
            "ticket_id_column": ticket_id_column,
            "dataset_name": dataset_name,
        },
        timeout=1200,
    )
    response.raise_for_status()
    return response.json()


def list_snapshots() -> list[dict[str, Any]]:
    response = requests.get(f"{API_URL}/list-snapshots", timeout=30)
    response.raise_for_status()
    return response.json()


def load_snapshot(name: str) -> dict[str, Any]:
    response = requests.post(f"{API_URL}/load-snapshot", json={"name": name}, timeout=60)
    response.raise_for_status()
    return response.json()


def train_all_models() -> dict[str, Any]:
    """
    Train routing models and semantic index with Railway connection failure recovery.
    
    If the /train response fails due to connection issues, verify that training
    actually succeeded on the API side by calling /verify-models.
    """
    try:
        print("[dashboard] Calling /train endpoint (timeout=3600s)...")
        response = requests.post(f"{API_URL}/train", timeout=3600)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Unexpected training response payload")
        print(f"[dashboard] Training succeeded: {payload}")
        return payload
    except (requests.exceptions.ChunkedEncodingError, 
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout) as conn_error:
        # Connection was interrupted, but models might have been trained
        print(f"[dashboard] Connection error during training: {conn_error}")
        print("[dashboard] Attempting to verify if models were actually trained...")
        try:
            verify_response = requests.post(f"{API_URL}/verify-models", timeout=30)
            verify_response.raise_for_status()
            verify_data = verify_response.json()
            print(f"[dashboard] Model verification result: {verify_data}")
            
            if verify_data.get("routing_models_loaded") or verify_data.get("all_models_loaded"):
                print("[dashboard] Models are loaded! Training succeeded despite connection error.")
                return {
                    "success": True,
                    "status": "Training succeeded (verified after connection recovery)",
                    "routing_models_loaded": verify_data.get("routing_models_loaded"),
                    "semantic_search_loaded": verify_data.get("semantic_search_loaded"),
                }
            else:
                print("[dashboard] Models not loaded after verification.")
                raise conn_error
        except Exception as verify_error:
            print(f"[dashboard] Verification also failed: {verify_error}")
            raise conn_error
    except requests.exceptions.HTTPError as http_error:
        print(f"[dashboard] HTTP error during training: {http_error}")
        raise


def verify_models_loaded() -> dict[str, Any]:
    """Verify that routing models are actually loaded in the API."""
    response = requests.post(f"{API_URL}/verify-models", timeout=30)
    response.raise_for_status()
    return response.json()


def clear_all() -> dict[str, Any]:
    response = requests.post(f"{API_URL}/clear-all", timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Unexpected clear-all payload")
    return payload


def build_search_index() -> dict[str, Any]:
    response = requests.post(f"{API_URL}/build-index", timeout=600)
    response.raise_for_status()
    return response.json()


def render_badge(text: str, color: str = "#0d6efd") -> None:
    st.markdown(
        f"<div style='display:inline-block;padding:18px 24px;border-radius:999px;background:{color};color:white;font-size:24px;font-weight:700;'>"
        f"{text}</div>",
        unsafe_allow_html=True,
    )


def render_status_badge(label: str, ready: bool) -> None:
    color = "#198754" if ready else "#dc3545"
    render_badge(label, color=color)


def render_ticket_card(ticket: dict[str, Any]) -> None:
    assigned_team = ticket.get("assigned_team", "Unknown")
    similarity = ticket.get("similarity_score", 0.0)
    ticket_id = ticket.get("ticket_id", "-")
    description = ticket.get("description", "")

    st.markdown("---")
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Ticket ID:** {ticket_id}")
            st.markdown(f"**Assigned Team:** {assigned_team}")
            st.markdown(f"**Description:** {description}")
        with col2:
            st.metric(label="Similarity", value=f"{similarity:.3f}")


def show_setup_training() -> None:
    st.header("Setup & Training")
    preview_df = None
    status_data = {}
    try:
        status_data = call_status()
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not retrieve system status: {exc}")

    dataset_status = status_data.get("dataset", {})
    models_status = status_data.get("models", {})
    faiss_status = status_data.get("faiss_index", {})

    st.subheader("Dataset")
    dataset_mode = st.radio(
        "Choose dataset flow",
        options=[
            "Upload your own CSV",
            "Generate synthetic dataset",
            "Load previously generated dataset",
        ],
        horizontal=True,
    )

    if dataset_mode == "Upload your own CSV":
        st.markdown("**Upload your own CSV file**")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        required_caption_parts: list[str] = []
        for internal_name in REQUIRED_INTERNAL_COLUMNS:
            aliases = [public for public, internal in PUBLIC_TO_INTERNAL_COLUMNS.items() if internal == internal_name]
            display_names = sorted(set([internal_name] + aliases))
            required_caption_parts.append(" / ".join(display_names))
        st.caption("Required columns: " + ", ".join(required_caption_parts))

        with st.expander("Configured alias mapping"):
            alias_rows = sorted(
                (
                    {"public_name": public_name, "internal_name": internal_name}
                    for public_name, internal_name in PUBLIC_TO_INTERNAL_COLUMNS.items()
                ),
                key=lambda x: (x["internal_name"], x["public_name"]),
            )
            st.dataframe(pd.DataFrame(alias_rows), use_container_width=True)

        selected_columns = None
        df_from_file = None
        description_column = None
        assigned_team_column = None
        ticket_id_column = None
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            try:
                df_from_file = pd.read_csv(BytesIO(file_bytes))
            except Exception as exc:
                st.error(f"Could not read uploaded CSV: {exc}")
                df_from_file = None

        if df_from_file is not None:
            available_columns = df_from_file.columns.tolist()
            selected_columns, description_column, assigned_team_column, ticket_id_column = _render_training_column_mapping(
                available_columns,
                key_prefix="upload",
                intro_text="Select the columns to keep for upload.",
            )

        if st.button("Upload & Validate", key="upload_dataset"):
            if uploaded_file is None:
                st.error("Please select a CSV file before uploading.")
            elif df_from_file is None:
                st.error("Could not read the uploaded CSV file.")
            elif selected_columns is None or len(selected_columns) == 0:
                st.error("Please select at least one column to include.")
            elif description_column is None:
                st.error("Please map the description column.")
            else:
                try:
                    subset_df = df_from_file[selected_columns].copy()

                    rename_map: dict[str, str] = {
                        description_column: "description",
                    }
                    if assigned_team_column and assigned_team_column != NO_TEAM_COLUMN_OPTION:
                        rename_map[assigned_team_column] = "assigned_team"
                    if ticket_id_column and ticket_id_column != AUTO_GENERATE_TICKET_ID_OPTION:
                        rename_map[ticket_id_column] = "ticket_id"

                    # Avoid duplicate canonical columns after rename.
                    for source_col, target_col in rename_map.items():
                        if source_col != target_col and target_col in subset_df.columns:
                            raise ValueError(
                                f"Cannot map '{source_col}' to '{target_col}' because '{target_col}' already exists in selected columns. "
                                f"Deselect '{target_col}' or choose another mapping."
                            )

                    subset_df = subset_df.rename(columns=rename_map)

                    if "ticket_id" not in subset_df.columns:
                        subset_df.insert(0, "ticket_id", [f"uploaded-{i}" for i in range(len(subset_df))])

                    subset_bytes = subset_df.to_csv(index=False).encode("utf-8")
                    result = upload_dataset_file(subset_bytes, uploaded_file.name)
                    has_team = "assigned_team" in subset_df.columns
                    st.success(f"Upload succeeded: {result['row_count']} rows validated.")
                    if not has_team:
                        st.info(
                            "No team column was mapped — ticket routing and model training will be disabled. "
                            "Search, KPI, Data Quality, SQL Explorer, and AI Suggestions remain available."
                        )
                    preview_df = subset_df
                except requests.exceptions.HTTPError as exc:
                    st.error(f"Upload failed: {exc.response.text}")
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")

    elif dataset_mode == "Generate synthetic dataset":
        st.markdown("**Generate synthetic dataset**")
        synthetic_dataset_name = st.text_input(
            "Synthetic dataset name",
            value="synthetic-test-dataset",
            help="Optional label used to identify this generated dataset in status and model metadata.",
        )
        synthetic_row_count = st.number_input(
            "Number of rows",
            min_value=100,
            max_value=1_000_000,
            value=50_000,
            step=1000,
            help="How many rows to generate. Larger datasets take longer but improve model accuracy.",
        )
        synthetic_options = _synthetic_output_options()
        st.caption("Choose which public-safe columns to include in the generated dataset and how they should map for training.")
        (
            synthetic_columns,
            synthetic_description_column,
            synthetic_assigned_team_column,
            synthetic_ticket_id_column,
        ) = _render_training_column_mapping(
            synthetic_options,
            key_prefix="synthetic",
            intro_text="Choose which generated columns should be kept and used for training.",
        )
        if st.button("Generate Synthetic Dataset", key="generate_dataset"):
            if len(synthetic_columns) == 0:
                st.error("Please select at least one synthetic column to include.")
            elif synthetic_description_column is None:
                st.error("Please map the description column.")
            else:
                try:
                    with st.spinner("Generating dataset..."):
                        result = generate_synthetic_dataset(
                            synthetic_columns,
                            description_column=synthetic_description_column,
                            assigned_team_column=(
                                ""
                                if synthetic_assigned_team_column == NO_TEAM_COLUMN_OPTION
                                else synthetic_assigned_team_column
                            ),
                            ticket_id_column=(
                                ""
                                if synthetic_ticket_id_column == AUTO_GENERATE_TICKET_ID_OPTION
                                else synthetic_ticket_id_column
                            ),
                            dataset_name=(synthetic_dataset_name.strip() or None),
                            size=int(synthetic_row_count),
                        )
                    st.success(f"Dataset generated with {result['row_count']} rows.")
                    if synthetic_assigned_team_column == NO_TEAM_COLUMN_OPTION:
                        st.info(
                            "No team column was mapped - ticket routing and model training will be disabled. "
                            "Search, KPI, Data Quality, SQL Explorer, and AI Suggestions remain available."
                        )
                    try:
                        preview_df = load_dataset().head(5)
                    except Exception as exc:
                        st.warning(f"Generated dataset created but preview failed: {exc}")
                except requests.exceptions.HTTPError as exc:
                    st.error(f"Generation failed: {exc.response.text}")
                except Exception as exc:
                    st.error(f"Generation failed: {exc}")

    else:
        st.markdown("**Load a previously generated dataset**")
        try:
            snapshots = list_snapshots()
        except Exception:
            snapshots = []

        if not snapshots:
            st.info("No saved datasets yet. Generate a synthetic dataset with a name to create one.")
        else:
            snapshot_names = [s["name"] for s in snapshots]
            snapshot_labels = {
                s["name"]: f"{s['name']}  ({s['row_count']:,} rows)" for s in snapshots
            }
            selected_snapshot = st.selectbox(
                "Saved datasets",
                options=snapshot_names,
                format_func=lambda n: snapshot_labels.get(n, n),
                key="load_snapshot_select",
            )
            if st.button("Load selected dataset", key="load_snapshot_btn"):
                try:
                    with st.spinner(f"Loading '{selected_snapshot}'..."):
                        result = load_snapshot(selected_snapshot)
                    st.success(
                        f"Dataset '{selected_snapshot}' loaded — {result['row_count']:,} rows. Re-train models to use it."
                    )
                    preview_df = load_dataset().head(5)
                except requests.exceptions.HTTPError as exc:
                    st.error(f"Load failed: {exc.response.text}")
                except Exception as exc:
                    st.error(f"Load failed: {exc}")

    if preview_df is not None:
        st.markdown("---")
        st.subheader("Dataset preview")
        st.dataframe(preview_df.head(5))

    st.markdown("---")
    st.subheader("Load and Train")
    st.caption("This action trains routing and rebuilds FAISS in one run.")
    if st.button("Train Routing + Semantic Index", key="train_models"):
        try:
            with st.spinner("Training routing models and rebuilding FAISS (this may take a few minutes)..."):
                result = train_all_models()
            if result.get("success"):
                st.success(result.get("status", "Training completed."))
                if result.get("artifacts_path"):
                    st.caption(
                        "Artifacts path: "
                        f"{result.get('artifacts_path', '-')}, "
                        f"rows: {result.get('row_count', 0)}, "
                        f"vectors: {result.get('vector_count', 0)}"
                    )
            elif result.get("routing_models_loaded"):
                st.success("✓ Training completed and models loaded (connection recovered)")
                st.info("Models verified as loaded after connection recovery")
            else:
                st.error(result.get("status", "Training failed."))
        except (requests.exceptions.ChunkedEncodingError, 
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as conn_exc:
            st.warning(f"⚠ Connection interrupted: {conn_exc}")
            st.info("Models may have trained successfully. Checking verification status...")
            try:
                verify_result = verify_models_loaded()
                if verify_result.get("routing_models_loaded"):
                    st.success("✓ Verification successful! Models are loaded.")
                    st.metric("Routing Models Loaded", "True")
                else:
                    st.error("Models not verified as loaded after training attempt.")
            except Exception as verify_exc:
                st.error(f"Could not verify models: {verify_exc}")
        except requests.exceptions.HTTPError as exc:
            st.error(f"Training failed: {exc.response.text}")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    st.markdown("---")
    st.subheader("Danger Zone")
    st.caption("Clears loaded models, FAISS index, dataset metadata, snapshots, and local generated dataset artifacts.")
    st.warning("Are you sure you want to clear everything? You will have to retrain the models.")

    if hasattr(st, "popover"):
        with st.popover("Clear Everything"):
            st.error("This action cannot be undone.")
            st.write("Are you sure you want to clear everything? You'll have to retrain the models.")
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button("Yes, clear everything", key="confirm_clear_everything_popover", type="primary"):
                    try:
                        result = clear_all()
                        st.success(f"Cleared {result.get('removed_count', 0)} artifacts. Please train models again.")
                        st.rerun()
                    except requests.exceptions.HTTPError as exc:
                        st.error(f"Clear failed: {exc.response.text}")
                    except Exception as exc:
                        st.error(f"Clear failed: {exc}")
            with cancel_col:
                st.button("Cancel", key="cancel_clear_everything_popover")
    else:
        if "confirm_clear_everything" not in st.session_state:
            st.session_state["confirm_clear_everything"] = False

        if st.button("Clear Everything", key="clear_everything_open_confirm"):
            st.session_state["confirm_clear_everything"] = True

        if st.session_state.get("confirm_clear_everything", False):
            st.error("This action cannot be undone.")
            st.write("Are you sure you want to clear everything? You'll have to retrain the models.")
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button("Yes, clear everything", key="confirm_clear_everything_inline", type="primary"):
                    try:
                        result = clear_all()
                        st.session_state["confirm_clear_everything"] = False
                        st.success(f"Cleared {result.get('removed_count', 0)} artifacts. Please train models again.")
                        st.rerun()
                    except requests.exceptions.HTTPError as exc:
                        st.error(f"Clear failed: {exc.response.text}")
                    except Exception as exc:
                        st.error(f"Clear failed: {exc}")
            with cancel_col:
                if st.button("Cancel", key="cancel_clear_everything_inline"):
                    st.session_state["confirm_clear_everything"] = False
                    st.rerun()

    st.markdown("---")
    st.subheader("System Status")
    try:
        status_data = call_status()
        dataset_status = status_data.get("dataset", {})
        models_status = status_data.get("models", {})
        faiss_status = status_data.get("faiss_index", {})
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not refresh status: {exc}")

    if st.button("Refresh status", key="refresh_status"):
        try:
            status_data = call_status()
            dataset_status = status_data.get("dataset", {})
            models_status = status_data.get("models", {})
            faiss_status = status_data.get("faiss_index", {})
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not refresh status: {exc}")

    dataset_ready = bool(dataset_status.get("exists"))
    models_ready = bool(models_status.get("loaded"))
    faiss_ready = bool(faiss_status.get("exists"))

    badge_col1, badge_col2, badge_col3 = st.columns(3)
    with badge_col1:
        render_status_badge("Dataset", dataset_ready)
    with badge_col2:
        render_status_badge("Models", models_ready)
    with badge_col3:
        render_status_badge("Semantic (FAISS)", faiss_ready)

    routing_capable = bool(dataset_status.get("routing_capable", False))
    active_name = dataset_status.get("dataset_name", "-")
    active_source = dataset_status.get("dataset_source", "-")
    active_rows = int(dataset_status.get("row_count", 0))
    faiss_vectors = int(faiss_status.get("vector_count", 0))

    st.markdown("**Current Dataset**")
    st.write("- Name:", active_name)
    st.write("- Source:", active_source)
    st.write("- Rows:", active_rows)
    st.write("- Routing-capable:", routing_capable)

    st.markdown("**Model State**")
    st.write("- Routing models loaded:", models_ready)
    st.write("- Semantic vectors (FAISS):", faiss_vectors)

    training_dataset = models_status.get("training_dataset")
    st.markdown("**Last Training**")
    if isinstance(training_dataset, dict):
        trained_name = training_dataset.get("dataset_name")
        trained_source = training_dataset.get("dataset_source")
        trained_rows = training_dataset.get("row_count", "-")
        trained_at = training_dataset.get("trained_at_utc", "-")
        dataset_hash = str(training_dataset.get("dataset_sha256", ""))

        same_dataset_identity = (
            (trained_name == active_name)
            and (trained_source == active_source)
            and (int(trained_rows) == active_rows if str(trained_rows).isdigit() else False)
        )

        if same_dataset_identity:
            st.write("- Dataset:", "Matches current active dataset")
        else:
            st.write("- Dataset name:", trained_name or "-")
            st.write("- Dataset source:", trained_source or "-")

        st.write("- Trained at (UTC):", trained_at)
        st.write("- Training rows:", trained_rows)
        if dataset_hash:
            st.write("- Dataset hash:", dataset_hash)
    else:
        st.write("- No training metadata available yet")

    if dataset_ready and not routing_capable:
        st.info(
            "Models stay red because this dataset has no assigned_team column. "
            "Routing training is skipped for this dataset. Upload or generate data with team labels to enable routing models."
        )


def show_ticket_routing() -> None:
    st.header("Route")
    description = st.text_area("Ticket description", height=180)
    route_button = st.button("Route Ticket")

    if route_button:
        if not description.strip():
            st.error("Please enter a ticket description before routing.")
            return

        with st.spinner("Routing ticket..."):
            try:
                payload = call_route(description.strip())
                assigned_team = payload.get("assigned_team", "Unknown")
                confidence = float(payload.get("confidence", 0.0))
                all_scores = payload.get("all_scores", {})

                render_badge(assigned_team, color="#0b5ed7")
                st.write("\n")
                st.subheader("Confidence")
                st.progress(min(max(confidence, 0.0), 1.0))
                st.caption(f"Assigned team confidence: {confidence:.2%}")

                if all_scores:
                    score_series = pd.Series(all_scores).sort_values(ascending=False)
                    st.subheader("Team confidence scores")
                    st.bar_chart(score_series)
                else:
                    st.info("No score details were returned.")
            except requests.exceptions.HTTPError as exc:
                st.error(f"API error: {exc.response.text}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Connection error: {exc}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")


def show_similar_cases() -> None:
    st.header("Search")
    query = st.text_area("Query description", height=180)
    top_k = st.slider("Top K results", min_value=1, max_value=10, value=5)
    search_button = st.button("Find Similar Cases")

    if search_button:
        if not query.strip():
            st.error("Please enter a search query.")
            return

        with st.spinner("Searching similar tickets..."):
            try:
                results = call_search(query.strip(), top_k)
                if not results:
                    st.info("No similar cases found.")
                    return

                for ticket in results:
                    render_ticket_card(ticket)
            except requests.exceptions.HTTPError as exc:
                st.error(f"API error: {exc.response.text}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Connection error: {exc}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")


def show_ai_suggestions() -> None:
    st.header("AI Suggestions")
    st.caption("Generate AI-powered response suggestions for support tickets using semantic search context + LLM")

    description = st.text_area(
        "Describe the support ticket",
        height=180,
        key="ai_suggestions_description",
    )
    generate_button = st.button("Generate AI Response", key="ai_suggestions_button")

    if generate_button:
        if not description.strip():
            st.error("Please enter a ticket description before generating an AI response.")
            return

        with st.spinner("Generating AI response suggestion..."):
            try:
                payload = call_suggest(description.strip())
                llm_available = bool(payload.get("llm_available", False))
                suggested_response = str(payload.get("suggested_response", "")).strip()
                context_tickets = payload.get("context_tickets", [])
                llm_error = payload.get("llm_error")

                if llm_available:
                    with st.container(border=True):
                        st.markdown("#### Suggested Agent Response")
                        st.write(suggested_response)
                else:
                    fallback_reason = str(llm_error or "Using local fallback draft")
                    if "not configured" in fallback_reason.lower():
                        st.info("Running local draft mode (no external LLM token configured).")
                    else:
                        st.warning(
                            "External LLM unavailable. Showing local fallback draft instead. "
                            + (f"**Reason:** {fallback_reason}" if fallback_reason else "")
                        )

                    if suggested_response:
                        with st.container(border=True):
                            st.markdown("#### Local Draft Response")
                            st.write(suggested_response)

                if isinstance(context_tickets, list) and context_tickets:
                    st.markdown("**Context tickets used**")
                    for ticket in context_tickets:
                        render_ticket_card(ticket)
            except requests.exceptions.HTTPError as exc:
                st.error(f"API error: {exc.response.text}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Connection error: {exc}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")


_SLA_HOURS: dict[str, int] = {"Critical": 24, "High": 48, "Medium": 72, "Low": 96}
_DEFAULT_SLA_HOURS = 72


def _compute_sla_breach(row: Any, now: pd.Timestamp) -> bool:
    """Return True if the ticket violated its SLA threshold."""
    threshold_h = _SLA_HOURS.get(str(row.get("severity", "")), _DEFAULT_SLA_HOURS)
    threshold_s = threshold_h * 3600
    ttc = row.get("time_to_close_seconds")
    if ttc not in (None, "", float("nan")):
        try:
            return float(ttc) > threshold_s
        except (ValueError, TypeError):
            pass
    # Open ticket — measure elapsed time from creation
    created = row.get("_created_dt")
    if created is not pd.NaT and created is not None:
        elapsed_s = (now - created).total_seconds()
        return elapsed_s > threshold_s
    return False


def show_kpi_analytics() -> None:
    st.header("KPI Analytics")

    try:
        df = load_dataset()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        return

    df = df.copy()
    has_team_col = "assigned_team" in df.columns
    if has_team_col:
        df["assigned_team"] = df["assigned_team"].fillna("Unknown")

    # Parse creation date
    for col in ["creation_date", "creation_datetime", "created_date", "created_timestamp"]:
        if col in df.columns:
            df["_created_dt"] = pd.to_datetime(df[col], errors="coerce")
            break
    else:
        df["_created_dt"] = pd.NaT

    now = pd.Timestamp.now()

    # ── Filters ──────────────────────────────────────────────────────────────
    with st.expander("Filters", expanded=True):
        num_filter_cols = 3 if has_team_col else 2
        filter_cols = st.columns(num_filter_cols)
        sel_team = "All"
        if has_team_col:
            teams = ["All"] + sorted(df["assigned_team"].dropna().unique().tolist())
            sel_team = filter_cols[0].selectbox("Team", teams)

        sev_col_idx = 1 if has_team_col else 0
        date_col_idx = 2 if has_team_col else 1
        severities = ["All"]
        if "severity" in df.columns:
            severities += sorted(df["severity"].dropna().unique().tolist())
        sel_sev = filter_cols[sev_col_idx].selectbox("Severity", severities)

        valid_dates = df["_created_dt"].dropna()
        if not valid_dates.empty:
            min_d = valid_dates.min().date()
            max_d = valid_dates.max().date()
            date_range = filter_cols[date_col_idx].date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                d_start, d_end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
                df = df[(df["_created_dt"] >= d_start) & (df["_created_dt"] <= d_end + pd.Timedelta(days=1))]

    if has_team_col and sel_team != "All":
        df = df[df["assigned_team"] == sel_team]
    if sel_sev != "All" and "severity" in df.columns:
        df = df[df["severity"] == sel_sev]

    if df.empty:
        st.warning("No tickets match the selected filters.")
        return

    # ── Derived columns ───────────────────────────────────────────────────────
    df["_is_open"] = ~df.get("status", pd.Series(dtype=str)).isin(["Resolved", "Closed"])
    df["_is_escalated"] = df.get("status", pd.Series(dtype=str)) == "Escalated"
    df["_sla_breach"] = df.apply(lambda r: _compute_sla_breach(r, now), axis=1)

    has_ttc = "time_to_close_seconds" in df.columns
    closed = df[~df["_is_open"]]
    if has_ttc and not closed.empty:
        ttc_numeric = pd.to_numeric(closed["time_to_close_seconds"], errors="coerce").dropna()
        avg_resolution_h = ttc_numeric.mean() / 3600 if not ttc_numeric.empty else None
    else:
        avg_resolution_h = None

    total = len(df)
    open_count = int(df["_is_open"].sum())
    escalated_pct = df["_is_escalated"].mean() * 100
    sla_breach_pct = df["_sla_breach"].mean() * 100

    # ── Section 1: Top KPIs ───────────────────────────────────────────────────
    st.markdown("### Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Tickets", f"{total:,}")
    m2.metric("Open Tickets", f"{open_count:,}", delta=f"{open_count/total*100:.1f}% of total")
    m3.metric(
        "Avg Resolution Time",
        f"{avg_resolution_h:.1f}h" if avg_resolution_h is not None else "—",
    )
    m4.metric("SLA Breach Rate", f"{sla_breach_pct:.1f}%")
    m5.metric("Escalation Rate", f"{escalated_pct:.1f}%")

    st.markdown("---")

    # ── Section 2: Volume trend + forecast  |  Severity distribution ─────────
    st.markdown("### Volume & Severity")
    col_trend, col_sev = st.columns([3, 2])

    with col_trend:
        if df["_created_dt"].notna().any():
            trend_df = df.dropna(subset=["_created_dt"]).copy()
            trend_df["week"] = trend_df["_created_dt"].dt.to_period("W").apply(lambda r: r.start_time)
            weekly = trend_df.groupby("week").size().reset_index(name="count").sort_values("week")

            # Simple linear forecast for next 4 weeks
            import numpy as np
            x = np.arange(len(weekly))
            if len(x) >= 3:
                coeffs = np.polyfit(x, weekly["count"].values, 1)
                forecast_x = np.arange(len(weekly), len(weekly) + 4)
                forecast_vals = np.polyval(coeffs, forecast_x).clip(0)
                last_date = weekly["week"].iloc[-1]
                forecast_dates = [last_date + pd.Timedelta(weeks=i + 1) for i in range(4)]
                forecast_df = pd.DataFrame({"week": forecast_dates, "count": forecast_vals, "type": "Forecast"})
                weekly["type"] = "Actual"
                combined = pd.concat([weekly, forecast_df], ignore_index=True)
                fig_trend = px.bar(
                    combined[combined["type"] == "Actual"], x="week", y="count",
                    title="Weekly ticket volume + 4-week forecast",
                    labels={"count": "Tickets", "week": ""},
                    color_discrete_sequence=["#4C78A8"],
                )
                fig_trend.add_scatter(
                    x=forecast_df["week"], y=forecast_df["count"],
                    mode="lines+markers", name="Forecast",
                    line=dict(dash="dash", color="#E45756"),
                )
            else:
                fig_trend = px.bar(weekly, x="week", y="count", title="Weekly ticket volume")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No date data available for trend chart.")

    with col_sev:
        if "severity" in df.columns:
            sev_counts = df["severity"].value_counts().reset_index()
            sev_counts.columns = ["severity", "count"]
            color_map = {"Critical": "#E45756", "High": "#F58518", "Medium": "#EECA3B", "Low": "#72B7B2"}
            fig_sev = px.pie(
                sev_counts, names="severity", values="count",
                title="Severity distribution", hole=0.45,
                color="severity", color_discrete_map=color_map,
            )
            fig_sev.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_sev, use_container_width=True)
        else:
            st.info("No severity column in dataset.")

    st.markdown("---")

    # ── Section 3: Team performance bar  |  Status distribution ─────────────
    st.markdown("### Team Performance")
    col_team, col_status = st.columns(2)

    with col_team:
        if not has_team_col:
            st.info("Team performance charts require an 'assigned_team' column.")
        elif has_ttc and not closed.empty:
            team_res = (
                closed.assign(ttc_h=pd.to_numeric(closed["time_to_close_seconds"], errors="coerce") / 3600)
                .groupby("assigned_team")["ttc_h"]
                .mean()
                .dropna()
                .sort_values()
                .reset_index()
            )
            team_res.columns = ["team", "avg_resolution_h"]
            fig_res = px.bar(
                team_res, x="avg_resolution_h", y="team", orientation="h",
                title="Avg resolution time per team (hours)",
                labels={"avg_resolution_h": "Hours", "team": ""},
                color="avg_resolution_h",
                color_continuous_scale="RdYlGn_r",
            )
            fig_res.update_coloraxes(showscale=False)
            st.plotly_chart(fig_res, use_container_width=True)
        else:
            team_counts = df["assigned_team"].value_counts().reset_index()
            team_counts.columns = ["team", "count"]
            fig_tc = px.bar(team_counts, x="team", y="count", title="Tickets per team")
            st.plotly_chart(fig_tc, use_container_width=True)

    with col_status:
        if "status" in df.columns:
            status_counts = df["status"].value_counts().reset_index()
            status_counts.columns = ["status", "count"]
            fig_status = px.bar(
                status_counts, x="status", y="count",
                title="Ticket status distribution",
                labels={"count": "Tickets", "status": ""},
                color="status",
            )
            st.plotly_chart(fig_status, use_container_width=True)
        else:
            st.info("No status column in dataset.")

    st.markdown("---")

    # ── Section 4: Team summary table ─────────────────────────────────────────
    st.markdown("### Team Summary Table")
    team_stats_rows = []
    if not has_team_col:
        st.info("Team summary table requires an 'assigned_team' column.")
    for team, grp in (df.groupby("assigned_team") if has_team_col else []):
        t_total = len(grp)
        t_open = int(grp["_is_open"].sum())
        t_sla = f"{grp['_sla_breach'].mean() * 100:.1f}%"
        t_esc = f"{grp['_is_escalated'].mean() * 100:.1f}%"
        if has_ttc:
            ttc_h = pd.to_numeric(grp.loc[~grp["_is_open"], "time_to_close_seconds"], errors="coerce") / 3600
            avg_h = f"{ttc_h.mean():.1f}h" if not ttc_h.dropna().empty else "—"
        else:
            avg_h = "—"
        team_stats_rows.append({
            "Team": team,
            "Total": t_total,
            "Open": t_open,
            "Avg Resolution": avg_h,
            "SLA Breach": t_sla,
            "Escalation": t_esc,
        })
    if team_stats_rows:
        st.dataframe(
            pd.DataFrame(team_stats_rows).sort_values("Total", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # ── Section 5: Failure modes  |  Channel breakdown ────────────────────────
    st.markdown("### Operational Breakdown")
    col_fail, col_chan = st.columns(2)

    with col_fail:
        fm_col = next((c for c in ["failure_mode", "component", "sr_area"] if c in df.columns), None)
        if fm_col:
            fm_counts = df[fm_col].value_counts().head(10).reset_index()
            fm_counts.columns = [fm_col, "count"]
            fig_fm = px.bar(
                fm_counts, x="count", y=fm_col, orientation="h",
                title=f"Top 10 — {fm_col.replace('_', ' ').title()}",
                labels={"count": "Tickets", fm_col: ""},
            )
            st.plotly_chart(fig_fm, use_container_width=True)

    with col_chan:
        if "ticket_channel" in df.columns:
            ch_counts = df["ticket_channel"].value_counts().reset_index()
            ch_counts.columns = ["channel", "count"]
            fig_ch = px.pie(
                ch_counts, names="channel", values="count",
                title="Tickets by channel", hole=0.4,
            )
            st.plotly_chart(fig_ch, use_container_width=True)
        elif "region" in df.columns:
            reg_counts = df["region"].value_counts().reset_index()
            reg_counts.columns = ["region", "count"]
            fig_reg = px.bar(reg_counts, x="region", y="count", title="Tickets by region")
            st.plotly_chart(fig_reg, use_container_width=True)

    st.markdown("---")

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("### Export")
    output = io.BytesIO()
    summary_data = {
        "Metric": ["Total Tickets", "Open Tickets", "Avg Resolution Time", "SLA Breach Rate", "Escalation Rate"],
        "Value": [
            total,
            open_count,
            f"{avg_resolution_h:.1f}h" if avg_resolution_h is not None else "—",
            f"{sla_breach_pct:.1f}%",
            f"{escalated_pct:.1f}%",
        ],
    }
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
        if team_stats_rows:
            pd.DataFrame(team_stats_rows).to_excel(writer, sheet_name="Team Stats", index=False)
        df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore").head(5000).to_excel(
            writer, sheet_name="Raw Data", index=False
        )
    output.seek(0)
    st.download_button(
        label="Download KPI Report (Excel)",
        data=output,
        file_name="support_ticket_kpi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def show_model_explainability() -> None:
    st.header("Model Performance")
    st.caption("Routing model accuracy, feature importance, and confusion matrix from the last training run")

    try:
        data = call_model_performance()
    except requests.exceptions.HTTPError as exc:
        if exc.response.status_code == 404:
            st.warning("No performance data found. Re-train the routing model in Setup & Training to generate it.")
        else:
            st.error(f"API error: {exc.response.text}")
        return
    except Exception as exc:
        st.error(f"Could not load model performance: {exc}")
        return

    accuracy = data.get("accuracy", 0.0)
    class_names = data.get("class_names", [])
    cm = data.get("confusion_matrix", [])
    features = data.get("feature_importance", [])

    # ── Top metrics ───────────────────────────────────────────────────────────
    st.markdown("### Routing Model Accuracy")
    acc_col, _ = st.columns([1, 3])
    acc_col.metric("Hold-out Accuracy", f"{accuracy * 100:.1f}%")
    st.progress(min(accuracy, 1.0))
    st.caption("Evaluated on a 20% hold-out split at training time (LogisticRegression + TF-IDF).")

    st.markdown("---")

    # ── Feature importance  |  Confusion matrix ───────────────────────────────
    st.markdown("### What Drives Routing Decisions")
    col_feat, col_cm = st.columns(2)

    with col_feat:
        if features:
            feat_df = pd.DataFrame(features).head(20)
            fig_feat = px.bar(
                feat_df.sort_values("importance"),
                x="importance", y="word", orientation="h",
                title="Top 20 keywords by routing influence",
                labels={"importance": "Importance", "word": ""},
                color="importance",
                color_continuous_scale="Blues",
            )
            fig_feat.update_coloraxes(showscale=False)
            fig_feat.update_layout(height=500)
            st.plotly_chart(fig_feat, use_container_width=True)
        else:
            st.info("No feature importance data available.")

    with col_cm:
        if cm and class_names:
            import numpy as np
            cm_arr = np.array(cm)
            # Normalise rows so colours show recall per class
            row_sums = cm_arr.sum(axis=1, keepdims=True).clip(1)
            cm_norm = (cm_arr / row_sums * 100).round(1)
            fig_cm = px.imshow(
                cm_norm,
                x=class_names,
                y=class_names,
                text_auto=True,
                color_continuous_scale="Blues",
                title="Confusion matrix — row-normalised recall (%)",
                labels={"x": "Predicted", "y": "Actual", "color": "Recall %"},
                aspect="auto",
            )
            fig_cm.update_layout(height=500)
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("No confusion matrix data available.")

    st.markdown("---")
    st.caption(
        "This model routes support tickets to the correct engineering team based on free-text descriptions. "
        "TF-IDF converts text to feature vectors; Logistic Regression classifies them. "
        "Feature importance is the sum of absolute coefficients across all classes."
    )


def show_data_quality() -> None:
    st.header("Data Quality")
    st.caption("Dataset health checks — missing values, duplicates, distributions, and outliers")

    try:
        df = load_dataset()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        return

    df = df.copy()

    # ── Overview ──────────────────────────────────────────────────────────────
    st.markdown("### Dataset Overview")
    ov1, ov2, ov3, ov4 = st.columns(4)
    ov1.metric("Rows", f"{len(df):,}")
    ov2.metric("Columns", len(df.columns))
    dup_count = int(df.duplicated().sum())
    ov3.metric("Duplicate rows", dup_count, delta=f"{dup_count/len(df)*100:.1f}%")
    null_pct = df.isnull().mean().mean() * 100
    ov4.metric("Overall null rate", f"{null_pct:.1f}%")

    st.markdown("---")

    # ── Missing values ────────────────────────────────────────────────────────
    st.markdown("### Missing Values per Column")
    null_df = (
        df.isnull().mean()
        .mul(100)
        .reset_index()
        .rename(columns={"index": "column", 0: "null_pct"})
        .sort_values("null_pct", ascending=False)
    )
    null_df.columns = ["column", "null_pct"]
    has_nulls = null_df[null_df["null_pct"] > 0]
    if has_nulls.empty:
        st.success("No missing values found in any column.")
    else:
        fig_null = px.bar(
            has_nulls.sort_values("null_pct"),
            x="null_pct", y="column", orientation="h",
            title="Columns with missing values (%)",
            labels={"null_pct": "Null %", "column": ""},
            color="null_pct", color_continuous_scale="Reds",
        )
        fig_null.update_coloraxes(showscale=False)
        st.plotly_chart(fig_null, use_container_width=True)

    st.markdown("---")

    # ── Key column distributions ───────────────────────────────────────────────
    st.markdown("### Key Column Distributions")
    cat_cols = [c for c in ["severity", "status", "region", "ticket_channel", "customer_type"] if c in df.columns]
    if cat_cols:
        cols = st.columns(min(len(cat_cols), 3))
        for i, col_name in enumerate(cat_cols):
            vc = df[col_name].value_counts().reset_index()
            vc.columns = [col_name, "count"]
            fig = px.bar(vc, x=col_name, y="count", title=col_name.replace("_", " ").title())
            cols[i % 3].plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Resolution time outliers ───────────────────────────────────────────────
    if "time_to_close_seconds" in df.columns:
        st.markdown("### Resolution Time Distribution & Outliers")
        ttc = pd.to_numeric(df["time_to_close_seconds"], errors="coerce").dropna()
        if not ttc.empty:
            ttc_h = ttc / 3600
            q1, q3 = ttc_h.quantile(0.25), ttc_h.quantile(0.75)
            iqr = q3 - q1
            outlier_mask = (ttc_h < q1 - 1.5 * iqr) | (ttc_h > q3 + 1.5 * iqr)
            outlier_count = int(outlier_mask.sum())

            st_col, hist_col = st.columns([1, 3])
            st_col.metric("Median resolution", f"{ttc_h.median():.1f}h")
            st_col.metric("95th percentile", f"{ttc_h.quantile(0.95):.1f}h")
            st_col.metric("Outliers (IQR)", outlier_count, delta=f"{outlier_count/len(ttc_h)*100:.1f}%")

            fig_hist = px.histogram(
                ttc_h.clip(upper=ttc_h.quantile(0.99)), nbins=60,
                title="Resolution time distribution (hours, clipped at 99th pct)",
                labels={"value": "Hours", "count": "Tickets"},
            )
            hist_col.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # ── Schema table ──────────────────────────────────────────────────────────
    st.markdown("### Column Schema")
    schema_rows = []
    for col in df.columns:
        series = df[col]
        schema_rows.append({
            "Column": col,
            "Type": str(series.dtype),
            "Null %": f"{series.isnull().mean() * 100:.1f}%",
            "Unique values": series.nunique(),
            "Sample": str(series.dropna().iloc[0]) if not series.dropna().empty else "—",
        })
    st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)


def show_overview() -> None:
    st.markdown(
        """
        <h1 style='font-size:2.2rem; margin-bottom:0.2rem;'>Support Ticket Intelligence</h1>
        <p style='font-size:1.1rem; color:#888; margin-top:0;'>
        End-to-end AI prototype for support ticket automation — routing, search, analytics, and AI-assisted responses.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Capability cards ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "📊", "KPI Analytics", "SLA breach rate, resolution time, team performance, and 4-week volume forecast."),
        (c2, "🔀", "Ticket Routing", "ML model routes tickets to the right team instantly. >99% accuracy on synthetic data."),
        (c3, "🔍", "Semantic Search", "FAISS + SentenceTransformers — find the most similar historical cases in milliseconds."),
        (c4, "🤖", "AI Suggestions", "HuggingFace Mistral-7B drafts a structured agent response grounded in past resolutions."),
    ]
    for col, icon, title, desc in cards:
        with col:
            st.markdown(
                f"""
                <div style='padding:1.2rem;border:1px solid #e0e0e0;border-radius:10px;height:160px;'>
                <div style='font-size:1.8rem'>{icon}</div>
                <div style='font-weight:700;margin:0.4rem 0 0.3rem;'>{title}</div>
                <div style='font-size:0.85rem;color:#555;'>{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Live stats ────────────────────────────────────────────────────────────
    st.markdown("### What's running right now")
    try:
        status_data = call_status()
        dataset_rows = status_data.get("dataset", {}).get("row_count", 0)
        models_loaded = status_data.get("models", {}).get("loaded", False)
        faiss_vectors = status_data.get("faiss_index", {}).get("vector_count", 0)

        s1, s2, s3 = st.columns(3)
        s1.metric("Tickets in dataset", f"{dataset_rows:,}")
        s2.metric("Routing model", "Ready" if models_loaded else "Not loaded")
        s3.metric("Search index vectors", f"{faiss_vectors:,}")
    except Exception:
        st.info("Could not reach the API to show live stats.")

    st.markdown("---")

    # ── How it works ─────────────────────────────────────────────────────────
    st.markdown("### How it works")
    st.markdown(
        """
        ```
        Synthetic dataset (50 k tickets)
               │
               ├─► TF-IDF + Logistic Regression  ──► /route   (team assignment)
               │
               └─► SentenceTransformers + FAISS   ──► /search  (similar cases)
                                                   ──► /suggest (AI response draft)
        ```
        All powered by a **FastAPI** backend and visualised in this **Streamlit** dashboard.
        """
    )

    st.markdown("---")

    # ── Links ─────────────────────────────────────────────────────────────────
    st.markdown("### Links")
    lc1, lc2, lc3 = st.columns(3)
    lc1.markdown("**GitHub**  \n[cirobdomingos-cyber/support-ticket-intelligence](https://github.com/cirobdomingos-cyber/support-ticket-intelligence)")
    lc2.markdown("**API docs (Swagger)**  \n[/docs](https://support-ticket-intelligence-production-795d.up.railway.app/docs)")
    lc3.markdown("**Built by**  \n[Ciro Beduschi Domingos](https://www.linkedin.com/in/ciro-beduschi-domingos-209b5138/)")


def show_sql_explorer() -> None:
    st.header("SQL Explorer")
    st.caption("Run SELECT queries directly against the tickets table (SQLite)")

    starter_queries = {
        "Tickets per team": "SELECT assigned_team, COUNT(*) AS tickets\nFROM tickets\nGROUP BY assigned_team\nORDER BY tickets DESC",
        "Avg resolution by severity": "SELECT severity, ROUND(AVG(CAST(time_to_close_seconds AS REAL)) / 3600, 1) AS avg_hours\nFROM tickets\nWHERE time_to_close_seconds != ''\nGROUP BY severity\nORDER BY avg_hours DESC",
        "Escalated tickets by region": "SELECT region, COUNT(*) AS escalated\nFROM tickets\nWHERE status = 'Escalated'\nGROUP BY region\nORDER BY escalated DESC",
        "Top failure modes": "SELECT failure_mode, COUNT(*) AS count\nFROM tickets\nGROUP BY failure_mode\nORDER BY count DESC\nLIMIT 10",
        "Open critical tickets": "SELECT ticket_id, description, assigned_team, creation_date\nFROM tickets\nWHERE severity = 'Critical' AND status NOT IN ('Resolved', 'Closed')\nLIMIT 20",
    }

    selected = st.selectbox("Starter queries", ["(write your own)"] + list(starter_queries.keys()))
    default_sql = starter_queries.get(selected, "SELECT * FROM tickets LIMIT 10")

    sql = st.text_area("SQL query", value=default_sql, height=140, key="sql_input")
    limit = st.slider("Max rows", min_value=10, max_value=2000, value=500, step=10)

    if st.button("Run Query", type="primary"):
        if not sql.strip():
            st.error("Please enter a query.")
            return
        try:
            with st.spinner("Executing..."):
                result = call_sql_query(sql.strip(), limit=limit)
            st.success(f"{result['row_count']} rows returned")
            if result["columns"] and result["rows"]:
                result_df = pd.DataFrame(result["rows"], columns=result["columns"])
                st.dataframe(result_df, use_container_width=True, hide_index=True)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download result as CSV",
                    data=csv_bytes,
                    file_name="query_result.csv",
                    mime="text/csv",
                )
            else:
                st.info("Query returned no rows.")
        except requests.exceptions.HTTPError as exc:
            st.error(f"Query error: {exc.response.json().get('detail', exc.response.text)}")
        except Exception as exc:
            st.error(f"Error: {exc}")

    with st.expander("Available columns in the tickets table"):
        st.code(
            "ticket_id, product, component, failure_mode, severity, assigned_team,\n"
            "status, sub_status, description, creation_date, creation_datetime,\n"
            "close_date, close_datetime, time_to_close_seconds, first_queued_seconds,\n"
            "region, customer_type, ticket_channel, creator_country, creator_department,\n"
            "owner_country, owner_department, dealer_id, dealer_name, mileage,\n"
            "error_code, sr_type, sr_area",
            language="text",
        )


def main() -> None:
    st.set_page_config(page_title="Support Ticket Dashboard", layout="wide")
    st.sidebar.title("Support Ticket Dashboard")
    st.sidebar.caption("Powered by the Support Ticket API")

    status_payload = safe_call_status()
    connected = bool(status_payload)
    routing_capable = status_payload.get("dataset", {}).get("routing_capable", False)
    routing_models_loaded = status_payload.get("models", {}).get("loaded", False)
    dataset_ready = status_payload.get("dataset", {}).get("exists", False)
    faiss_ready = status_payload.get("faiss_index", {}).get("exists", False)

    # Basic pages require dataset + FAISS index; routing pages additionally require trained routing models.
    modules_ready = dataset_ready and faiss_ready
    routing_ready = routing_capable and routing_models_loaded

    if connected:
        status_text = "Connected" if modules_ready else "Connected (setting up...)"
        status_color = "✅"
    else:
        status_text = "Disconnected"
        status_color = "⚠️"

    st.sidebar.markdown("### Connection status")
    st.sidebar.write(f"{status_color} {status_text}")
    st.sidebar.write(f"API URL: `{API_URL}`")

    available_pages = ["Overview", "Setup & Training"]
    if modules_ready:
        available_pages += ["KPI", "Data Quality", "SQL Explorer"]
        if routing_ready:
            available_pages += ["Model Performance"]
        available_pages += ["Search"]
        if routing_ready:
            available_pages += ["Route"]
        available_pages += ["AI Suggestions"]

    page = st.sidebar.radio("Navigation", available_pages)

    if page == "Overview":
        show_overview()
    elif page == "Setup & Training":
        show_setup_training()
    elif page == "KPI":
        show_kpi_analytics()
    elif page == "Data Quality":
        show_data_quality()
    elif page == "SQL Explorer":
        show_sql_explorer()
    elif page == "Model Performance":
        show_model_explainability()
    elif page == "Search":
        show_similar_cases()
    elif page == "Route":
        show_ticket_routing()
    else:
        show_ai_suggestions()


if __name__ == "__main__":
    main()
