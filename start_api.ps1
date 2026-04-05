Set-Location "$PSScriptRoot\4-support-ticket-api"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating venv..." -ForegroundColor Gray
    py -3.12 -m venv .venv
}

$py = ".\.venv\Scripts\python.exe"

if (-not (Test-Path ".venv\Scripts\uvicorn.exe") -and -not (Test-Path ".venv\Scripts\uvicorn")) {
    Write-Host "Installing dependencies (first run takes several minutes)..." -ForegroundColor Gray
    & $py -m pip install -r requirements.txt
}

Write-Host "API starting at http://localhost:8000" -ForegroundColor Green
Write-Host "API docs at   http://localhost:8000/docs" -ForegroundColor White
& $py -m uvicorn main:app --reload --port 8000
