@echo off
REM setup.bat — one-command bootstrap for support-ticket-intelligence
REM Usage: setup.bat

setlocal enabledelayedexpansion

set "REPO_ROOT=%~dp0"
REM Strip trailing backslash
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

echo.
echo =================================================
echo   Support Ticket Intelligence — Setup
echo =================================================
echo.

REM ── 1. Requirements ──────────────────────────────
echo [1/4] Installing Python requirements...
python -m pip install --upgrade pip --quiet
if errorlevel 1 goto :error
python -m pip install -r "%REPO_ROOT%\requirements.txt" --quiet
if errorlevel 1 goto :error
echo       Done.
echo.

REM ── 2. Generate synthetic dataset ────────────────
echo [2/4] Generating synthetic dataset...
python "%REPO_ROOT%\1-support-ticket-dataset\generator\generate_dataset.py" ^
    --output "%REPO_ROOT%\1-support-ticket-dataset\data\sample_dataset.csv"
if errorlevel 1 goto :error
echo       Done.
echo.

REM ── 3. Train routing models ───────────────────────
echo [3/4] Training routing models (this may take a few minutes)...
python "%REPO_ROOT%\2-support-ticket-routing-ml\src\train_baselines.py"
if errorlevel 1 goto :error
echo       Done.
echo.

REM ── 4. Build FAISS search index ───────────────────
echo [4/4] Building FAISS semantic search index...
python "%REPO_ROOT%\3-support-ticket-semantic-search\src\semantic_search.py"
if errorlevel 1 goto :error
echo       Done.
echo.

echo =================================================
echo   Setup complete!
echo.
echo   Start the API (run in a terminal):
echo     cd 4-support-ticket-api
echo     uvicorn main:app --reload --host 0.0.0.0 --port 8000
echo.
echo   Start the dashboard (run in a new terminal):
echo     cd 5-support-ticket-dashboard
echo     streamlit run app.py
echo.
echo   Open dashboard: http://localhost:8501
echo   Open API docs:  http://localhost:8000/docs
echo =================================================
goto :eof

:error
echo.
echo ERROR: Setup failed at the step above. Check the output above for details.
exit /b 1
