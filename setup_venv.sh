#!/usr/bin/env bash
# Creates a Python virtual environment and installs all project dependencies.
# Run once from the project root:
#   chmod +x setup_venv.sh
#   ./setup_venv.sh

set -e  # exit immediately if any command fails

VENV_DIR=".venv"
PYTHON=${PYTHON:-python3}   # override by running: PYTHON=python3.11 ./setup_venv.sh

# ── Create the virtual environment ───────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' already exists — skipping creation."
else
    echo "Creating virtual environment in '$VENV_DIR' ..."
    $PYTHON -m venv "$VENV_DIR"
    echo "Done."
fi

# ── Activate ──────────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "Activated: $(which python)"

# ── Upgrade pip ───────────────────────────────────────────────────────────────
pip install --upgrade pip

# ── Install dependencies ──────────────────────────────────────────────────────
echo "Installing dependencies from requirements.txt ..."
pip install -r requirements.txt

echo ""
echo "Setup complete. To activate the environment in your shell, run:"
echo "  source $VENV_DIR/bin/activate"
