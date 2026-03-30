# Support Ticket API

FastAPI service exposing support ticket routing and semantic search endpoints.

## Setup

From the `4-support-ticket-api/` folder:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run locally

```bash
cd 4-support-ticket-api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Run with Docker Compose

```bash
cd 4-support-ticket-api
docker compose up --build
```

If your environment uses the older Compose command, use:

```bash
docker-compose up --build
```

## New management endpoints

### GET /status

Returns the current dataset, routing model, and FAISS index state.

### POST /upload-dataset

Upload a CSV file with required columns configured in `../column_aliases.json` (`required_internal_columns`).
Public alias names from the same config are also accepted.

### POST /generate-dataset

Generates a synthetic dataset using the dataset generator module.
You can pass optional `include_columns` (internal or public names) in the request body.
When the API is deployed without the full monorepo available (for example, an API-only Railway service),
the endpoint falls back to an in-process generator and writes the dataset under the API service's local
`data/sample_dataset.csv` path.

### POST /train

Starts routing model training and streams progress back as JSON lines.

### POST /build-index

Builds or rebuilds the FAISS index from the currently available dataset.

## Endpoints

### GET /health

Returns:

```json
{
  "status": "ok",
  "models_loaded": true
}
```

### POST /route

Route a ticket description to an assigned team.

Request body:

```json
{
  "description": "Customer reports battery overheating during startup"
}
```

### POST /search

Search for similar past tickets.

Request body:

```json
{
  "description": "engine makes loud knocking noise during acceleration",
  "top_k": 5
}
```

## Notes

- The API loads models at startup and returns `503` until they are ready.
- The service integrates routing predictions and semantic search results.
- Use the dashboard or direct API calls for evaluation.

