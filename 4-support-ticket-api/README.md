# Support Ticket API

FastAPI service exposing support ticket routing and semantic search endpoints.

## Setup

From the 4-support-ticket-api folder:

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

Starts routing model training synchronously, saves artifacts to the API local models directory,
reloads routing resources, and returns a JSON result.

### POST /build-index

Builds or rebuilds the FAISS index from the currently available dataset.

### POST /suggest

Generates an AI response suggestion for a support agent using:
- Semantic search context (top 3 similar tickets)
- A prompt template in prompts/suggest_response.txt
- A configurable LLM provider (OpenAI or HuggingFace)

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

### POST /suggest

Generate an AI-assisted response suggestion for a support ticket.

Request body:

```json
{
  "description": "Customer reports intermittent battery overheating after a software update"
}
```

Response body:

```json
{
  "suggested_response": "Diagnosis summary: ...",
  "context_tickets": [
    {
      "ticket_id": "ticket-001",
      "description": "...",
      "assigned_team": "Electrical Systems",
      "similarity_score": 0.84
    }
  ],
  "llm_available": true
}
```

## Railway environment variables

Set these in Railway service Variables to enable AI suggestions.

### OpenAI (default)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
```

### HuggingFace Hub (optional alternative)

```env
LLM_PROVIDER=huggingface
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
HUGGINGFACE_REPO_ID=google/flan-t5-large
```

If keys are missing, the API returns a graceful fallback with llm_available=false.

## Notes

- The API loads models at startup and returns 503 until they are ready.
- The service integrates routing predictions and semantic search results.
- Use the dashboard or direct API calls for evaluation.

