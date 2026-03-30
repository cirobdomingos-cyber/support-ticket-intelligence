#!/usr/bin/env bash
# setup.sh — one-command bootstrap for support-ticket-intelligence
# Usage: bash setup.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "================================================="
echo "  Support Ticket Intelligence — Setup"
echo "================================================="
echo ""

# ── 1. Requirements ──────────────────────────────────
echo "[1/4] Installing Python requirements..."
python -m pip install --upgrade pip --quiet
python -m pip install -r "$REPO_ROOT/requirements.txt" --quiet
echo "      Done."
echo ""

# ── 2. Generate synthetic dataset ───────────────────
echo "[2/4] Generating synthetic dataset..."
python "$REPO_ROOT/1-support-ticket-dataset/generator/generate_dataset.py" \
    --output "$REPO_ROOT/1-support-ticket-dataset/data/sample_dataset.csv"
echo "      Done."
echo ""

# ── 3. Train routing models ──────────────────────────
echo "[3/4] Training routing models (this may take a few minutes)..."
python "$REPO_ROOT/2-support-ticket-routing-ml/src/train_baselines.py"
echo "      Done."
echo ""

# ── 4. Build FAISS search index ──────────────────────
echo "[4/4] Building FAISS semantic search index..."
python "$REPO_ROOT/3-support-ticket-semantic-search/src/semantic_search.py"
echo "      Done."
echo ""

echo "================================================="
echo "  Setup complete!"
echo ""
echo "  Start the API:"
echo "    cd 4-support-ticket-api"
echo "    uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "  Start the dashboard (new terminal):"
echo "    cd 5-support-ticket-dashboard"
echo "    streamlit run app.py"
echo ""
echo "  Open dashboard: http://localhost:8501"
echo "  Open API docs:  http://localhost:8000/docs"
echo "================================================="
