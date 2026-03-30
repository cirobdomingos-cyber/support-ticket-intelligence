# Support Ticket Intelligence

An end-to-end AI prototype for support ticket automation, built as a modular
pipeline with dataset generation, routing models, semantic search, API service,
and dashboard visualization.

---

## Quick start (one command)

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

The setup script will:
1. Install all Python requirements
2. Generate the synthetic dataset
3. Train the routing models
4. Build the FAISS search index

Once setup finishes, start the two services:

```bash
# Terminal 1 — API
cd 4-support-ticket-api
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Dashboard
cd 5-support-ticket-dashboard
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

This repository uses numbered module folders to keep the project structure clear:

- `1-support-ticket-dataset/` — synthetic ticket data generator
- `2-support-ticket-routing-ml/` — routing model training and inference
- `3-support-ticket-semantic-search/` — semantic search with embeddings and FAISS
- `4-support-ticket-api/` — FastAPI service exposing routing and search endpoints
- `5-support-ticket-dashboard/` — Streamlit dashboard for operators and managers

---

## Why this project

Technical support teams often have good ticket capture systems but lack the
intelligence to route issues, reuse past solutions, and track operational metrics.
This project proves a practical architecture for:

- automatic ticket routing
- case similarity retrieval
- API-based integration
- lightweight analytics dashboard

---

## Architecture
```
1-support-ticket-dataset/  →  2-support-ticket-routing-ml/  +  3-support-ticket-semantic-search/  →  4-support-ticket-api/  →  5-support-ticket-dashboard/
``` 

---

## Module summary

| Folder | Purpose | Status |
|---|---|---|
| `1-support-ticket-dataset` | Synthetic dataset generation and export | ✅ Working |
| `2-support-ticket-routing-ml` | Ticket routing model baselines and training | ✅ Working |
| `3-support-ticket-semantic-search` | Similar-case search using embeddings and FAISS | ✅ Working |
| `4-support-ticket-api` | FastAPI endpoints for route and search | ✅ Working |
| `5-support-ticket-dashboard` | Streamlit dashboard for routing and analytics | ✅ Working |

---

## Manual setup (step by step)

1. Clone the repository:

```bash
git clone https://github.com/cirobdomingos-cyber/support-ticket-intelligence.git
cd support-ticket-intelligence
```

2. Install all requirements:

```bash
python -m pip install -r requirements.txt
```

3. Generate the dataset:

```bash
python 1-support-ticket-dataset/generator/generate_dataset.py
```

4. Train the routing models:

```bash
python 2-support-ticket-routing-ml/src/train_baselines.py
```

5. Build the semantic search index:

```bash
python 3-support-ticket-semantic-search/src/semantic_search.py
```

6. Start the API service:

```bash
cd 4-support-ticket-api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

7. Run the dashboard:

```bash
cd 5-support-ticket-dashboard
streamlit run app.py
```

---

## Notes

- You can install all module dependencies with the root `requirements.txt`.
- Alternatively, install per-module requirements inside each numbered folder.
- Use the module folder names exactly as listed above when navigating or running
  commands.
- Column aliases are centralized in `column_aliases.json` at repository root.
- Update `column_aliases.json` to customize public/internal field names per company.
- The project is designed as a prototype and uses synthetic data for feature
  exploration and pipeline validation.

---

## Tech stack

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `SentenceTransformers` · `FAISS` · `FastAPI` · `Streamlit` · `Docker`

---

## Contact

Built by [Ciro Beduschi Domingos](https://www.linkedin.com/in/ciro-beduschi-domingos-209b5138/).