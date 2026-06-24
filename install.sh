#!/usr/bin/env bash
# =============================================================================
# install.sh — musemender one-shot setup for Ubuntu
# Run once from inside the musemender directory:
#   cd musemender && bash install.sh
# =============================================================================
set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[musemender]${RESET} $*"; }
success() { echo -e "${GREEN}[✓]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[!]${RESET} $*"; }
fail()    { echo -e "${RED}[✗] $*${RESET}"; exit 1; }

echo -e "\n${BOLD}musemender — install${RESET}\n"

# ── 1. Must run from the musemender root ─────────────────────────────────────
if [[ ! -f "backend/app.py" ]]; then
    fail "Run this from inside the musemender directory:\n    cd musemender && bash install.sh"
fi

# ── 2. Python version check (3.10+ required) ─────────────────────────────────
info "Checking Python version..."
PY=$(command -v python3 || true)
if [[ -z "$PY" ]]; then
    fail "python3 not found. Install it: sudo apt install python3"
fi

PY_VER=$($PY -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PY -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PY -c "import sys; print(sys.version_info.minor)")

if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 10 ]]; }; then
    fail "Python 3.10+ required. Found: $PY_VER\nOn Ubuntu 20.04: sudo apt install python3.10"
fi
success "Python $PY_VER"

# ── 3. System packages ────────────────────────────────────────────────────────
info "Installing system packages (ffmpeg, python3-venv)..."
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg python3-venv python3-pip
success "System packages installed"

# ── 4. Virtual environment ────────────────────────────────────────────────────
if [[ -d "venv" ]]; then
    warn "venv already exists — skipping creation. Delete it and re-run to start fresh."
else
    info "Creating virtual environment..."
    $PY -m venv venv
    success "Virtual environment created at ./venv"
fi

source venv/bin/activate

# ── 5. Pip packages ───────────────────────────────────────────────────────────
info "Upgrading pip..."
pip install --upgrade pip --quiet

info "Installing Python packages (this will take a few minutes on first run)..."
info "  torch + torchaudio are large — ~800MB download"
pip install -r requirements.txt

success "Python packages installed"

# ── 6. Required directories ───────────────────────────────────────────────────
info "Ensuring required directories exist..."
mkdir -p data uploads
success "Directories: data/, uploads/"

# ── 7. Verify critical imports ────────────────────────────────────────────────
info "Verifying imports..."
python3 - <<'PYCHECK'
import sys

failures = []

checks = [
    ("torch",         "import torch; assert torch.__version__"),
    ("torchaudio",    "import torchaudio"),
    ("transformers",  "from transformers import ClapModel, ClapProcessor"),
    ("librosa",       "import librosa"),
    ("faiss",         "import faiss"),
    ("flask",         "from flask import Flask"),
    ("flask_cors",    "from flask_cors import CORS"),
    ("mutagen",       "from mutagen import File"),
    ("numpy",         "import numpy"),
    ("tqdm",          "from tqdm import tqdm"),
    ("soundfile",     "import soundfile"),
    ("werkzeug",      "from werkzeug.utils import secure_filename"),
]

for name, stmt in checks:
    try:
        exec(stmt)
        print(f"  \033[32m✓\033[0m  {name}")
    except Exception as e:
        print(f"  \033[31m✗\033[0m  {name}: {e}")
        failures.append(name)

if failures:
    print(f"\n\033[31mFailed imports: {', '.join(failures)}\033[0m")
    sys.exit(1)
else:
    print("\n\033[32mAll imports OK\033[0m")
PYCHECK

success "All imports verified"

# ── 8. ffmpeg check ───────────────────────────────────────────────────────────
if command -v ffmpeg &>/dev/null; then
    FFMPEG_VER=$(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')
    success "ffmpeg $FFMPEG_VER"
else
    warn "ffmpeg not found on PATH — .wma/.aac/.m4a conversion will fail"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}Installation complete.${RESET}"
echo ""
echo -e "${BOLD}Next steps:${RESET}"
echo ""
echo -e "  ${CYAN}1. Index your music library:${RESET}"
echo -e "     bash index.sh /path/to/your/Music"
echo ""
echo -e "  ${CYAN}2. Start the app:${RESET}"
echo -e "     bash start.sh"
echo ""
echo -e "  ${CYAN}3. Open in browser:${RESET}"
echo -e "     http://localhost:5000"
echo ""
echo -e "  ${YELLOW}Note: The first index run downloads the CLAP model (~600MB).${RESET}"
echo -e "  ${YELLOW}Each song takes 15-45s to embed on CPU — run it overnight for large libraries.${RESET}"
echo ""
