Set-Location "$PSScriptRoot\5-support-ticket-dashboard"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating venv..." -ForegroundColor Gray
    python -m venv .venv
}

$py = ".\.venv\Scripts\python.exe"

if (-not (Test-Path ".venv\Scripts\streamlit.exe") -and -not (Test-Path ".venv\Scripts\streamlit")) {
    Write-Host "Installing dependencies..." -ForegroundColor Gray
    & $py -m pip install -r requirements.txt
}

$env:DUCKDB_PATH = "$PSScriptRoot\6-dbt-analytics\dev.duckdb"
$env:API_URL     = "http://localhost:8000"

Write-Host "Dashboard starting at http://localhost:8501" -ForegroundColor Green
& $py -m streamlit run app.py
