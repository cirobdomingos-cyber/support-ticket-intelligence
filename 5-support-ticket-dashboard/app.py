from __future__ import annotations

import json
import os
from io import BytesIO
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "https://support-ticket-intelligence-production-795d.up.railway.app")

DEFAULT_PUBLIC_TO_INTERNAL_COLUMNS: dict[str, str] = {
    "ticket_uuid": "ticket_id",
    "issue_description": "description",
    "route_team": "assigned_team",
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
    return df


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
    except Exception as exc:
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


def generate_synthetic_dataset(include_columns: list[str]) -> dict[str, Any]:
    response = requests.post(
        f"{API_URL}/generate-dataset",
        json={"include_columns": include_columns},
        timeout=1200,
    )
    response.raise_for_status()
    return response.json()


def train_routing_models() -> requests.Response:
    response = requests.post(f"{API_URL}/train", stream=True, timeout=None)
    response.raise_for_status()
    return response


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
    left_col, right_col = st.columns(2)
    with left_col:
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
            st.markdown("Select the columns to keep for upload.")
            available_columns = df_from_file.columns.tolist()

            def _default_index(options: list[str], candidates: list[str]) -> int:
                for candidate in candidates:
                    if candidate in options:
                        return options.index(candidate)
                return 0

            description_column = st.selectbox(
                "Which column contains ticket description text?",
                options=available_columns,
                index=_default_index(available_columns, ["description", "issue_description"]),
            )
            assigned_team_column = st.selectbox(
                "Which column is the training target (assigned team)?",
                options=available_columns,
                index=_default_index(available_columns, ["assigned_team", "route_team"]),
            )
            ticket_id_options = ["(auto-generate ticket_id)"] + available_columns
            ticket_id_column = st.selectbox(
                "Optional: which column should be used as ticket id?",
                options=ticket_id_options,
                index=_default_index(ticket_id_options, ["ticket_id", "ticket_uuid"]),
            )

            selected_columns = st.multiselect(
                "Columns to include",
                options=available_columns,
                default=available_columns,
            )

            required_selected = [description_column, assigned_team_column]
            if ticket_id_column and ticket_id_column != "(auto-generate ticket_id)":
                required_selected.append(ticket_id_column)
            missing_selected = [col for col in required_selected if col not in set(selected_columns)]
            if missing_selected:
                st.error("Selected columns must include mapped columns: " + ", ".join(missing_selected))

        if st.button("Upload & Validate", key="upload_dataset"):
            if uploaded_file is None:
                st.error("Please select a CSV file before uploading.")
            elif df_from_file is None:
                st.error("Could not read the uploaded CSV file.")
            elif selected_columns is None or len(selected_columns) == 0:
                st.error("Please select at least one column to include.")
            elif description_column is None or assigned_team_column is None:
                st.error("Please map description and assigned team columns.")
            else:
                try:
                    subset_df = df_from_file[selected_columns].copy()

                    rename_map: dict[str, str] = {
                        description_column: "description",
                        assigned_team_column: "assigned_team",
                    }
                    if ticket_id_column and ticket_id_column != "(auto-generate ticket_id)":
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
                    st.success(f"Upload succeeded: {result['row_count']} rows validated.")
                    preview_df = subset_df
                except requests.exceptions.HTTPError as exc:
                    st.error(f"Upload failed: {exc.response.text}")
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")

    with right_col:
        st.markdown("**Generate synthetic dataset**")
        synthetic_columns = st.multiselect(
            "Synthetic output columns",
            options=sorted(PUBLIC_TO_INTERNAL_COLUMNS.keys()),
            default=sorted(PUBLIC_TO_INTERNAL_COLUMNS.keys()),
            help="Choose which public-safe columns to include in generated dataset.",
        )
        if st.button("Generate Synthetic Dataset", key="generate_dataset"):
            try:
                with st.spinner("Generating dataset..."):
                    result = generate_synthetic_dataset(synthetic_columns)
                st.success(f"Dataset generated with {result['row_count']} rows.")
                try:
                    preview_df = load_dataset().head(5)
                except Exception as exc:
                    st.warning(f"Generated dataset created but preview failed: {exc}")
            except requests.exceptions.HTTPError as exc:
                st.error(f"Generation failed: {exc.response.text}")
            except Exception as exc:
                st.error(f"Generation failed: {exc}")

    if preview_df is not None:
        st.markdown("---")
        st.subheader("Dataset preview")
        st.dataframe(preview_df.head(5))

    st.markdown("---")
    st.subheader("Train Models")
    if st.button("Train Routing Models", key="train_models"):
        try:
            with st.spinner("Training routing models..."):
                response = train_routing_models()
                status_placeholder = st.empty()
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    try:
                        payload = json.loads(raw_line)
                        status_placeholder.text(f"{payload.get('step')}: {payload.get('status')}")
                    except ValueError:
                        status_placeholder.text(raw_line)
                st.success("Training stream completed.")
        except requests.exceptions.HTTPError as exc:
            st.error(f"Training failed: {exc.response.text}")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    st.markdown("---")
    st.subheader("Build Search Index")
    if st.button("Build Semantic Search Index", key="build_index"):
        try:
            with st.spinner("Building FAISS index..."):
                result = build_search_index()
            st.success(f"Index built with {result['vector_count']} vectors.")
        except requests.exceptions.HTTPError as exc:
            st.error(f"Index build failed: {exc.response.text}")
        except Exception as exc:
            st.error(f"Index build failed: {exc}")

    st.markdown("---")
    st.subheader("System Status")
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
        render_status_badge("Search Index", faiss_ready)

    st.write("- Dataset rows: ", dataset_status.get("row_count", 0))
    st.write("- Routing models loaded: ", models_ready)
    st.write("- FAISS vectors: ", faiss_status.get("vector_count", 0))


def show_ticket_routing() -> None:
    st.header("Ticket Routing")
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
    st.header("Similar Cases")
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

    if "assigned_team" not in df.columns:
        st.error("Dataset is missing the assigned_team column.")
        return

    df = df.copy()
    df["assigned_team"] = df["assigned_team"].fillna("Unknown")

    # Try to detect a date column; if none found, let user pick from all columns.
    date_column = None
    for candidate in ["creation_date", "creation_datetime", "close_date", "close_datetime",
                      "created_date", "created_timestamp", "closed_date", "closed_timestamp"]:
        if candidate in df.columns:
            date_column = candidate
            break
    if date_column is None:
        datetime_candidates = df.columns.tolist()
        date_column = st.selectbox(
            "No recognised date column found. Select the date/time column to use for trend chart:",
            options=["(none — skip trend chart)"] + datetime_candidates,
        )
        if date_column == "(none — skip trend chart)":
            date_column = None

    teams = ["All"] + sorted(df["assigned_team"].dropna().unique().tolist())
    selected_team = st.selectbox("Filter by team", teams)
    filtered = df if selected_team == "All" else df[df["assigned_team"] == selected_team]

    team_counts = filtered["assigned_team"].value_counts().rename_axis("team").reset_index(name="count")

    has_date = date_column is not None
    col1, col2, col3 = st.columns(3)
    col1.metric("Total tickets", len(filtered))

    if not team_counts.empty:
        fig_team = px.bar(team_counts, x="team", y="count", title="Tickets per team")
        col2.plotly_chart(fig_team, use_container_width=True)
    else:
        col2.info("No tickets available for the selected team.")

    if has_date:
        date_series = pd.to_datetime(filtered[date_column], errors="coerce")
        trend_df = filtered.copy()
        trend_df["_date"] = date_series
        trend_df = trend_df.dropna(subset=["_date"])
        trend_df["week"] = trend_df["_date"].dt.to_period("W").apply(lambda r: r.start_time)
        weekly = trend_df.groupby("week").size().reset_index(name="count").sort_values("week")
        if not weekly.empty:
            fig_time = px.line(weekly, x="week", y="count", title="Tickets over time")
            col3.plotly_chart(fig_time, use_container_width=True)
        else:
            col3.info("No parseable dates in selected column.")
    else:
        col3.info("No date column selected — trend chart skipped.")

    st.markdown("---")
    st.subheader("Dataset preview")
    st.dataframe(filtered.head(10))


def main() -> None:
    st.set_page_config(page_title="Support Ticket Dashboard", layout="wide")
    st.sidebar.title("Support Ticket Dashboard")
    st.sidebar.caption("Powered by the Support Ticket API")

    connected, models_loaded = check_health()
    if connected:
        status_text = "Connected" if models_loaded else "Connected, model load pending"
        status_color = "✅"
    else:
        status_text = "Disconnected"
        status_color = "⚠️"

    st.sidebar.markdown("### Connection status")
    st.sidebar.write(f"{status_color} {status_text}")
    st.sidebar.write(f"API URL: `{API_URL}`")

    status_payload = safe_call_status()
    modules_ready = (
        status_payload.get("dataset", {}).get("exists", False)
        and status_payload.get("models", {}).get("loaded", False)
        and status_payload.get("faiss_index", {}).get("exists", False)
    )

    available_pages = ["Setup & Training"]
    if modules_ready:
        available_pages += ["Ticket Routing", "Similar Cases", "KPI Analytics"]

    page = st.sidebar.radio("Navigation", available_pages)

    if page == "Setup & Training":
        show_setup_training()
    elif page == "Ticket Routing":
        show_ticket_routing()
    elif page == "Similar Cases":
        show_similar_cases()
    else:
        show_kpi_analytics()


if __name__ == "__main__":
    main()
