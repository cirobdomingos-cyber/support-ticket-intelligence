# run.ps1 — Start the full Support Ticket Intelligence stack
# Usage: .\run.ps1
# Run from the repo root directory.

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Support Ticket Intelligence — Startup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: dbt build (runs here, blocks until done) ─────────────────────────
Write-Host "[1/3] Building dbt analytics layer..." -ForegroundColor Yellow

Set-Location "$Root\6-dbt-analytics"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "      Creating venv..." -ForegroundColor Gray
    python -m venv .venv
    Write-Host "      Installing dbt-duckdb..." -ForegroundColor Gray
    & ".\.venv\Scripts\pip.exe" install dbt-duckdb --quiet
}

if (-not (Test-Path "dbt_packages")) {
    Write-Host "      Installing dbt packages..." -ForegroundColor Gray
    & ".\.venv\Scripts\dbt.exe" deps --profiles-dir . --quiet
}

Write-Host "      Running dbt build..." -ForegroundColor Gray
& ".\.venv\Scripts\dbt.exe" build --profiles-dir .

if (-not $?) {
    Write-Host "dbt build failed. Fix the errors above before continuing." -ForegroundColor Red
    exit 1
}

Write-Host "[1/3] dbt build complete -> dev.duckdb ready" -ForegroundColor Green
Write-Host ""

# ── Step 2: API (new window) ──────────────────────────────────────────────────
Write-Host "[2/3] Starting API on port 8000 (new window)..." -ForegroundColor Yellow

$apiScript = @"
`$ErrorActionPreference = 'Continue'
Write-Host ''
Write-Host '=== API Service ===' -ForegroundColor Cyan
Set-Location '$Root\4-support-ticket-api'

if (-not (Test-Path '.venv\Scripts\python.exe')) {
    Write-Host 'Creating venv...' -ForegroundColor Gray
    python -m venv .venv
    Write-Host 'Installing dependencies (this may take several minutes on first run)...' -ForegroundColor Gray
    .\.venv\Scripts\pip.exe install -r requirements.txt
}

Write-Host 'Starting uvicorn...' -ForegroundColor Green
Write-Host 'API docs: http://localhost:8000/docs' -ForegroundColor White
Write-Host ''
.\.venv\Scripts\uvicorn.exe main:app --reload --port 8000
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $apiScript

# ── Step 3: Dashboard (new window) ───────────────────────────────────────────
Write-Host "[3/3] Starting Dashboard on port 8501 (new window)..." -ForegroundColor Yellow

$dashScript = @"
`$ErrorActionPreference = 'Continue'
Write-Host ''
Write-Host '=== Dashboard Service ===' -ForegroundColor Cyan
Set-Location '$Root\5-support-ticket-dashboard'

if (-not (Test-Path '.venv\Scripts\python.exe')) {
    Write-Host 'Creating venv...' -ForegroundColor Gray
    python -m venv .venv
    Write-Host 'Installing dependencies...' -ForegroundColor Gray
    .\.venv\Scripts\pip.exe install -r requirements.txt
}

`$env:DUCKDB_PATH = '$Root\6-dbt-analytics\dev.duckdb'
`$env:API_URL     = 'http://localhost:8000'

Write-Host 'Starting Streamlit...' -ForegroundColor Green
Write-Host 'Dashboard: http://localhost:8501' -ForegroundColor White
Write-Host ''
.\.venv\Scripts\streamlit.exe run app.py
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $dashScript

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "All services are starting in separate windows." -ForegroundColor Green
Write-Host ""
Write-Host "  Dashboard : http://localhost:8501" -ForegroundColor White
Write-Host "  API docs  : http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "First run: the API window will download ~2 GB of ML models." -ForegroundColor Yellow
Write-Host "The dashboard is usable immediately — KPI Analytics works right away." -ForegroundColor Yellow
Write-Host "Routing / Search / AI require the API to finish loading." -ForegroundColor Yellow
Write-Host ""
