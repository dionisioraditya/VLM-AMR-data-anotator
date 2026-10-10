@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv" (
    echo Membuat virtual environment Python...
    python -m venv .venv
)

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment gagal dibuat atau python tidak ditemukan.
    exit /b 1
)

.venv\Scripts\python.exe -c "import PySide6" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Dependensi belum terinstall. Mengunduh dan menginstall dari requirements.txt...
    .venv\Scripts\pip.exe install -r requirements.txt
)

echo Menjalankan AMR VLM Dataset Maker...
.venv\Scripts\python.exe main.py %*
endlocal
