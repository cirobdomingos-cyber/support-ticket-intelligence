# Support Ticket Intelligence — Claude Code Guidelines

## Project Owner
Ciro Beduschi Domingos — Senior Data Analyst, 15 years at Volvo Group (Python/SQL/Power BI/Databricks).
Goal: portfolio for remote Analytics Engineer / Senior Data Analyst roles ($80k–$130k USD).
This project must look and feel like real production work, not a tutorial.

## What This Project Does
End-to-end AI prototype that automates technical support ticket triage:
- Automatic ticket routing via ML (>99% accuracy)
- Semantic search over historical cases (FAISS + SentenceTransformers)
- Agent response drafting via LLM (HuggingFace — free tier)
- Operational KPI dashboard (Streamlit)

**Live demo:** lavish-connection-production-5385.up.railway.app
**API docs:** support-ticket-intelligence-production-795d.up.railway.app/docs

## Project Structure
```
1-support-ticket-dataset/     # Synthetic data generator (~50k tickets, 38 columns)
2-support-ticket-routing-ml/  # TF-IDF + LogisticRegression model training
3-support-ticket-semantic-search/ # Sentence embeddings + FAISS index
4-support-ticket-api/         # FastAPI — serves everything over HTTP
5-support-ticket-dashboard/   # Streamlit — full UI
```

## Tech Stack
- **Language:** Python 3.12 (do NOT use Python 3.14 — many packages lack wheels for it)
- **API:** FastAPI + Uvicorn
- **Dashboard:** Streamlit
- **ML:** scikit-learn (TF-IDF + LogReg), SentenceTransformers, FAISS
- **LLM:** HuggingFace Inference API (Qwen/Qwen2.5-7B-Instruct-1M by default)
- **Deploy:** Railway (2 separate services: API and Dashboard)
- **CI:** GitHub Actions

## Running Locally (Windows PowerShell)

### API (port 8000)
```powershell
cd 4-support-ticket-api
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

### Dashboard (port 8501)
```powershell
cd 5-support-ticket-dashboard
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:API_URL = "http://localhost:8000"
.venv\Scripts\python.exe -m streamlit run app.py
```

## Environment Variables
- `API_URL` — API base URL (default: http://localhost:8000)
- `HUGGINGFACEHUB_API_TOKEN` — HuggingFace token for AI suggestions (optional; falls back to local draft without it)
- `HUGGINGFACE_REPO_ID` — LLM model ID (default: Qwen/Qwen2.5-7B-Instruct)
- `DUCKDB_PATH` — path to DuckDB file for analytics layer (module 6)

## Deploy (Railway)
- 2 services: `4-support-ticket-api` and `5-support-ticket-dashboard`
- Each has its own `Dockerfile` and `railway.json`
- API healthcheck at `/health`
- Push to `main` = automatic production deploy
- A "dev" Railway environment exists for testing before promoting to production

## Branch Workflow
- `main` = production (Railway auto-deploys on push)
- Never commit directly to main — always use a PR
- Use worktrees for isolated development: `claude --worktree dev/task-name`
- Dev branches map to the Railway "dev" environment for live testing

## Planned Next Module: dbt + DuckDB (module 6)
Analytics layer on top of ticket data:
- dbt Core + DuckDB adapter, Python 3.12
- Seed: support_tickets.csv (50k rows, 38 cols)
- Models: staging → intermediate (SLA breach logic) → marts
- Marts: ticket_kpis, team_workload, dealer_performance, product_defects
- Dashboard KPI page reads from DuckDB marts (replaces pandas runtime transforms)
- Deploy: multi-stage Dockerfile — dbt-builder stage produces dev.duckdb, baked into dashboard image
- **Always develop on a separate branch, validate in Railway dev, then PR to main**

## Architectural Decisions (do not revert without reason)
- Python 3.12 for local venvs — Python 3.14 breaks pandas/pydantic without Visual Studio
- Use `python -m <module>` instead of .exe shims — more reliable on Windows
- LLM via HuggingFace free tier — zero cost, works on Railway without extra config
- DuckDB (zero infrastructure) for analytics layer instead of Postgres
- `requirements.txt` uses `>=` for packages with native extensions — avoids build failures on new Python versions

## Code Standards
- Commit messages in English, explain the "why" not the "what"
- PRs always from a branch, never direct push to main
- PowerShell scripts use `py -3.12` explicitly to avoid picking up Python 3.14
- All new code and comments in English

## What This Project Demonstrates (interview context)
- End-to-end ML pipeline: data → model → API → UI
- Analytics engineering with dbt layered modeling
- Containerised deploy on real cloud infrastructure (Railway)
- Clean separation of concerns: API / UI / analytics as independent layers
- Production quality: tests, CI, documentation, no TODOs left behind
