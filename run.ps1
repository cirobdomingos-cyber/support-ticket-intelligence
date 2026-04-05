# run.ps1 — Start the full Support Ticket Intelligence stack
# Usage (from repo root): .\run.ps1

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

Write-Host ""
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "  Support Ticket Intelligence — Startup"  -ForegroundColor Cyan
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host ""

# ── Step 1: dbt build (blocks until done) ────────────────────────────────────
Write-Host "[1/3] Building dbt analytics layer..." -ForegroundColor Yellow

Set-Location "$Root\6-dbt-analytics"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "      Creating venv..." -ForegroundColor Gray
    py -3.12 -m venv .venv
    Write-Host "      Installing dbt-duckdb..." -ForegroundColor Gray
    .\.venv\Scripts\python.exe -m pip install dbt-duckdb --quiet
}

$dbtPy = ".\.venv\Scripts\python.exe"

if (-not (Test-Path "dbt_packages")) {
    Write-Host "      Installing dbt packages..." -ForegroundColor Gray
    & $dbtPy -m dbt deps --profiles-dir . --quiet
}

& $dbtPy -m dbt build --profiles-dir .

if ($LASTEXITCODE -ne 0) {
    Write-Host "dbt build failed. Fix the errors above before continuing." -ForegroundColor Red
    exit 1
}

Write-Host "[1/3] Done — dev.duckdb ready" -ForegroundColor Green
Write-Host ""

# ── Step 2: API (new window) ──────────────────────────────────────────────────
Write-Host "[2/3] Starting API on port 8000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-File", "$Root\start_api.ps1"

# ── Step 3: Dashboard (new window) ───────────────────────────────────────────
Write-Host "[3/3] Starting Dashboard on port 8501..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-File", "$Root\start_dashboard.ps1"

Write-Host ""
Write-Host "Done. Two new windows are opening." -ForegroundColor Green
Write-Host ""
Write-Host "  Dashboard : http://localhost:8501" -ForegroundColor White
Write-Host "  API docs  : http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "First run: API window downloads ~2 GB of ML models. Be patient." -ForegroundColor Yellow
Write-Host "KPI Analytics works immediately. Routing/Search/AI need the API ready." -ForegroundColor Yellow
