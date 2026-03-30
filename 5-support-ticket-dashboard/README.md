# Support Ticket Dashboard

Streamlit dashboard for ticket routing, case similarity search, and high-level support KPIs.

## Setup

```bash
cd 5-support-ticket-dashboard
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
cd 5-support-ticket-dashboard
streamlit run app.py
```

## Docker

```bash
cd 5-support-ticket-dashboard
docker compose up --build
```

Or with legacy Compose:

```bash
docker-compose up --build
```

## Configuration

The dashboard reads `API_URL` from the top of `app.py`:

```python
API_URL = "http://localhost:8000"
```

Update that value if the API is hosted on another host or port.

## Dataset

The dashboard expects the dataset at:

```text
../1-support-ticket-dataset/data/sample_dataset.csv
```

The upload/generation UI reads alias mappings from `../column_aliases.json` and displays
the configured public/internal field list dynamically.

## Pages

- **Ticket Routing** — send a ticket description to `/route` and display the assigned team,
  confidence, and per-team score chart.
- **Similar Cases** — submit a query and view the top similar tickets returned by `/search`.
- **KPI Analytics** — load the sample dataset and show ticket counts, team volume, and time
  series trends.

## Notes

- Start the API before using the dashboard.
- The dashboard handles API errors with user-friendly messages.
- The visualization uses Streamlit layout primitives and Plotly charts for analytics.

