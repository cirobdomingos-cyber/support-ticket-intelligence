# Support Ticket Dashboard

Streamlit dashboard for setup and training, ticket routing, case similarity search, AI response suggestions, and high-level support KPIs.

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

AI suggestions also depend on the API having a valid `HUGGINGFACEHUB_API_TOKEN` configured.

## Dataset

The dashboard expects the dataset at:

```text
../1-support-ticket-dataset/data/sample_dataset.csv
```

The upload/generation UI reads alias mappings from `../column_aliases.json` and displays
the configured public/internal field list dynamically.

## Pages

- **Setup & Training** — upload or generate a dataset, train routing models, build the FAISS index, and review system status.
- **KPI** — load the sample dataset and show ticket counts, team volume, and time series trends.
- **Search** — submit a query and view the top similar tickets returned by `/search`.
- **Route** — send a ticket description to `/route` and display the assigned team, confidence, and per-team score chart.
- **AI Suggestions** — draft an agent response with semantic-search context and the API `/suggest` endpoint.

## AI Suggestions

The AI Suggestions page uses the API suggestion endpoint and shows a fallback message if the API has no HuggingFace token configured.

To enable it on Railway, add:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

You can create a free token at `https://huggingface.co/settings/tokens`.

## Notes

- Start the API before using the dashboard.
- The dashboard handles API errors with user-friendly messages.
- The visualization uses Streamlit layout primitives and Plotly charts for analytics.

