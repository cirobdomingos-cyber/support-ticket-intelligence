Set-Location "$PSScriptRoot\5-support-ticket-dashboard"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating venv..." -ForegroundColor Gray
    python -m venv .venv
    Write-Host "Installing dependencies..." -ForegroundColor Gray
    .\.venv\Scripts\pip.exe install -r requirements.txt
}

$env:DUCKDB_PATH = "$PSScriptRoot\6-dbt-analytics\dev.duckdb"
$env:API_URL     = "http://localhost:8000"

Write-Host "Dashboard starting at http://localhost:8501" -ForegroundColor Green
.\.venv\Scripts\streamlit.exe run app.py
