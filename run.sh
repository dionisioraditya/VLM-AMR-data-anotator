#!/usr/bin/env bash
# Quick launcher script for AMR VLM Dataset Maker

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ ! -d ".venv" ]; then
    echo "Membuat virtual environment Python..."
    python3 -m venv .venv
fi

# Cek apakah PySide6 sudah terinstall di .venv
if ! .venv/bin/python -c "import PySide6" 2>/dev/null; then
    echo "Dependensi belum terinstall. Mengunduh dan menginstall dari requirements.txt..."
    .venv/bin/pip install -r requirements.txt
fi

echo "Menjalankan AMR VLM Dataset Maker..."
.venv/bin/python main.py "$@"
