# Support Ticket Intelligence

An end-to-end AI platform for technical support operations — built from real-world experience
supporting heavy truck fleets across Latin America.

This project combines **automated ticket routing**, **semantic case retrieval**, and
**analytics dashboards** into a modular system designed to replace reactive, manual
support workflows with data-driven intelligence.

---

## The Problem

Enterprise technical support systems are good at recording cases. They are bad at everything
else: routing tickets to the right team, surfacing similar past issues, detecting emerging
failure patterns, or giving managers real visibility into operations.

This platform is the secondary intelligence layer that fixes that.

---

## Architecture
```
Dataset generation  →  ML routing  +  Semantic search  →  REST API  →  Dashboard
   (synthetic)          (classifier)    (FAISS embeddings)  (FastAPI)   (Streamlit)
```

### Modules

| Module | Description | Status |
|---|---|---|
| `1-dataset` | Synthetic support ticket generator with metadata, VIN, dealer info, lifecycle labels | ✅ Complete |
| `2-routing-ml` | Text classification baselines: TF-IDF + LogReg, XGBoost, DistilBERT | ✅ Baselines working |
| `3-semantic-search` | Sentence embeddings + FAISS index for similar case retrieval | ✅ Working |
| `4-api` | FastAPI service exposing routing and search as REST endpoints | 🔄 In progress |
| `5-dashboard` | Streamlit KPI dashboard with routing insights and case analytics | 📅 Planned |

---

## Quick Start
```bash
git clone https://github.com/cirobdomingos-cyber/support-ticket-intelligence
cd support-ticket-intelligence
pip install -r requirements.txt

# Generate dataset
python 1-dataset/generate_dataset.py

# Train routing models
python 2-routing-ml/train_baselines.py

# Run semantic search
python 3-semantic-search/semantic_search.py
```

---

## Design Decisions

**Why synthetic data?**
Real support ticket data is proprietary and contains sensitive vehicle and customer
information. The synthetic generator is designed to replicate realistic distributions
of ticket types, team assignments, and metadata — making it useful for pipeline
development and model architecture validation without exposing production data.

Accuracy metrics on this dataset reflect dataset structure, not real-world generalization.
The value of this project is the **architecture and pipeline**, not the benchmark numbers.

**Why multiple model approaches?**
TF-IDF + LogReg gives a fast, interpretable baseline. XGBoost handles structured metadata
features. DistilBERT captures semantic nuance in free-text descriptions. Comparing all three
makes the tradeoffs explicit — latency vs accuracy vs explainability.

**Why FAISS?**
Support engineers waste significant time re-solving problems that have already been solved.
Vector similarity search lets an engineer query a new ticket and instantly retrieve the
5 most similar past cases — including resolution steps.

---

## Roadmap

- [x] Synthetic dataset generation
- [x] Text classification baselines (TF-IDF, XGBoost)
- [x] Semantic similarity search (FAISS)
- [ ] Fix DistilBERT training and save pipeline
- [ ] FastAPI service with `/route` and `/search` endpoints
- [ ] Docker containerization
- [ ] Streamlit dashboard (KPIs, routing analytics, case explorer)
- [ ] LLM-assisted response suggestion (LangChain + OpenAI)
- [ ] CI/CD with GitHub Actions

---

## Background

This project grew out of 6 years spent in heavy truck technical support, followed by
another 6 years building data solutions for the same domain. The gap between what support
systems *record* and what they *enable* is large — and entirely solvable with modern
ML and search tooling.

---

## Tech Stack

`Python` · `scikit-learn` · `XGBoost` · `Transformers (DistilBERT)` · `SentenceTransformers`
· `FAISS` · `FastAPI` · `Streamlit` · `Docker` · `pandas` · `GitHub Actions`

---

*Built by [Ciro Beduschi Domingos](https://www.linkedin.com/in/seu-linkedin) —
Senior Data & AI Professional with 15 years at Volvo Trucks.*