# Support Ticket Intelligence

> End-to-end AI prototype that automates support ticket routing, surfaces similar past cases,
> and drafts agent responses — deployed live on Railway.

**Live demo:** [lavish-connection-production-5385.up.railway.app](https://lavish-connection-production-5385.up.railway.app)

---

## The Business Problem

Technical support teams spend significant time on manual work that machines can handle:

| Problem | This project's solution |
|---|---|
| Agents manually decide which team should own a ticket | **ML routing model** classifies tickets automatically with >99% accuracy |
| Agents re-solve problems that were solved before | **Semantic search** retrieves the 5 most similar historical cases |
| New agents don't know how to respond | **AI suggestion** drafts a response grounded in past resolutions |
| Managers have no visibility into team performance or SLA health | **KPI dashboard** shows resolution time, SLA breach rate, and escalation rate |

---

## Architecture

```
Synthetic Dataset Generator
        │
        ▼
┌───────────────────────┐      ┌──────────────────────────────────┐
│  2 · Routing ML       │      │  3 · Semantic Search             │
│  TF-IDF + LogReg      │      │  SentenceTransformers + FAISS    │
│  >99% accuracy        │      │  cosine similarity retrieval     │
└──────────┬────────────┘      └─────────────┬────────────────────┘
           │                                 │
           └──────────────┬──────────────────┘
                          ▼
              ┌───────────────────────┐
              │  4 · FastAPI          │
              │  /route  /search      │
              │  /suggest  /status    │
              │  /model-performance   │
              └──────────┬────────────┘
                         ▼
              ┌───────────────────────┐
              │  5 · Streamlit        │
              │  KPI · Model Perf     │
              │  Search · Route       │
              │  AI Suggestions       │
              └───────────────────────┘
```

---

## Dashboard Pages

### KPI Analytics
Operational metrics with filters for team, severity, and date range:
- Total tickets, open count, avg resolution time, SLA breach rate, escalation rate
- Weekly volume trend with 4-week linear forecast
- Severity distribution (donut chart)
- Avg resolution time per team (horizontal bar)
- Status distribution, top failure modes, channel breakdown
- Full team summary table with per-team SLA and escalation rates

### Model Performance
Transparency into the routing model:
- Hold-out accuracy score
- Top 20 TF-IDF keywords that drive routing decisions
- Row-normalised confusion matrix (recall per class)

### Search
Semantic similarity search — find historical tickets most similar to a query, ranked by cosine similarity.

### Route
Enter a ticket description and get an instant team assignment with confidence scores for every team.

### AI Suggestions
Generates a structured agent response using the 3 most similar historical tickets as context, powered by HuggingFace Mistral-7B (free tier).

---

## Quick Start (one command)

**Windows:**
```bat
git clone https://github.com/cirobdomingos-cyber/support-ticket-intelligence.git
cd support-ticket-intelligence
setup.bat
```

**Linux / Mac:**
```bash
git clone https://github.com/cirobdomingos-cyber/support-ticket-intelligence.git
cd support-ticket-intelligence
bash setup.sh
```

Then start the two services:

```bash
# Terminal 1 — API (auto-generates dataset + trains models on first boot)
cd 4-support-ticket-api
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Dashboard
cd 5-support-ticket-dashboard
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

> **First boot:** The API auto-generates a 50 000-row synthetic dataset, trains the routing model,
> and builds the FAISS index automatically. No manual setup step required.

---

## Manual Setup (step by step)

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Generate dataset
python 1-support-ticket-dataset/generator/generate_dataset.py

# 3. Train routing models
python 2-support-ticket-routing-ml/src/train_baselines.py

# 4. Build semantic search index
python 3-support-ticket-semantic-search/src/semantic_search.py

# 5. Start API
cd 4-support-ticket-api && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 6. Start dashboard
cd 5-support-ticket-dashboard && streamlit run app.py
```

---

## Module Summary

| Folder | Purpose | Status |
|---|---|---|
| `1-support-ticket-dataset` | Synthetic dataset generation (50 k rows, 30+ fields) | ✅ |
| `2-support-ticket-routing-ml` | TF-IDF + Logistic Regression routing model | ✅ |
| `3-support-ticket-semantic-search` | SentenceTransformers embeddings + FAISS index | ✅ |
| `4-support-ticket-api` | FastAPI — routing, search, AI suggestion, KPI data, model performance | ✅ |
| `5-support-ticket-dashboard` | Streamlit — KPI, Model Performance, Search, Route, AI Suggestions | ✅ |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/status` | Dataset, model, and index status |
| POST | `/route` | Predict team for a ticket description |
| POST | `/search` | Find top-K similar historical tickets |
| POST | `/suggest` | Generate AI response suggestion |
| GET | `/model-performance` | Feature importance + confusion matrix |
| POST | `/train` | Retrain routing model |
| POST | `/build-index` | Rebuild FAISS semantic index |
| POST | `/generate-dataset` | Generate new synthetic dataset |
| POST | `/upload-dataset` | Upload a custom CSV dataset |

---

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `API_URL` | (Railway internal URL) | Dashboard → API URL |
| `HUGGINGFACEHUB_API_TOKEN` | — | Required to enable AI Suggestions |
| `HUGGINGFACE_REPO_ID` | `mistralai/Mistral-7B-Instruct-v0.2` | LLM model for suggestions |

---

## Tech Stack

`Python 3.11` · `FastAPI` · `Streamlit` · `pandas` · `scikit-learn` · `SentenceTransformers` · `FAISS` · `LangChain` · `HuggingFace` · `Docker` · `Railway`

---

## About

Built by [Ciro Beduschi Domingos](https://www.linkedin.com/in/ciro-beduschi-domingos-209b5138/) as a portfolio project demonstrating end-to-end applied ML and data analytics engineering.
