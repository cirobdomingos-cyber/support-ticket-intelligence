Set-Location "$PSScriptRoot\4-support-ticket-api"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating venv..." -ForegroundColor Gray
    python -m venv .venv
    Write-Host "Installing dependencies (first run takes several minutes)..." -ForegroundColor Gray
    .\.venv\Scripts\pip.exe install -r requirements.txt
}

Write-Host "API starting at http://localhost:8000" -ForegroundColor Green
Write-Host "API docs at   http://localhost:8000/docs" -ForegroundColor White
.\.venv\Scripts\uvicorn.exe main:app --reload --port 8000
