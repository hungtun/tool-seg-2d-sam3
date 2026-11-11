#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

# Find suitable Python binary (Python >= 3.11 required for SAM 3 Transformers)
if [ -x "/usr/local/bin/python3.13" ]; then
  PYTHON_BIN="/usr/local/bin/python3.13"
elif [ -x "/opt/homebrew/bin/python3.13" ]; then
  PYTHON_BIN="/opt/homebrew/bin/python3.13"
elif [ -x "/usr/local/bin/python3.12" ]; then
  PYTHON_BIN="/usr/local/bin/python3.12"
elif command -v python3.13 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.13)"
elif command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.12)"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.11)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Error: Python 3 not found."
  exit 1
fi

echo "==> Creating virtual environment using $PYTHON_BIN ($($PYTHON_BIN --version))..."
rm -rf "$VENV_DIR"
"$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
echo "==> Installing dependencies from requirements.txt..."
python -m pip install -r "$ROOT_DIR/requirements.txt"

echo ""
echo "=================================================================="
echo "Installation complete with $(python --version)!"
echo "Activate the environment using:"
echo "  source .venv/bin/activate"
echo ""
echo "Verify CLI:"
echo "  python cli.py --help"
echo "=================================================================="
