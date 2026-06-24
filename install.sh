#!/usr/bin/env bash
# =============================================================================
# install.sh — musemender one-shot setup for Ubuntu
# Works whether files are flat in musemender/ or already in subfolders.
# =============================================================================
set -euo pipefail

# ── Always work from the directory this script lives in ──────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[musemender]${RESET} $*"; }
success() { echo -e "${GREEN}[✓]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[!]${RESET} $*"; }
fail()    { echo -e "${RED}[✗] $*${RESET}"; exit 1; }

echo -e "\n${BOLD}musemender — install${RESET}"
echo -e "Working directory: $SCRIPT_DIR\n"

# ── 1. Create subdirectory structure ─────────────────────────────────────────
info "Creating directory structure..."
mkdir -p backend frontend data uploads
success "Directories ready"

# ── 2. Move flat files into correct subfolders if needed ─────────────────────
BACKEND_FILES=(app.py config.py db.py extractor.py indexer.py similarity.py)
FRONTEND_FILES=(index.html)

MOVED=0

for f in "${BACKEND_FILES[@]}"; do
    if [[ -f "$SCRIPT_DIR/$f" ]]; then
        mv "$SCRIPT_DIR/$f" "$SCRIPT_DIR/backend/$f"
        info "  Moved $f → backend/"
        MOVED=$((MOVED + 1))
    fi
done

for f in "${FRONTEND_FILES[@]}"; do
    if [[ -f "$SCRIPT_DIR/$f" ]]; then
        mv "$SCRIPT_DIR/$f" "$SCRIPT_DIR/frontend/$f"
        info "  Moved $f → frontend/"
        MOVED=$((MOVED + 1))
    fi
done

if [[ $MOVED -gt 0 ]]; then
    success "Moved $MOVED file(s) into correct subfolders"
else
    info "Files already in subfolders — nothing to move"
fi

# ── 3. Verify all required files are present ─────────────────────────────────
info "Checking all required files..."

MISSING=()
for f in "${BACKEND_FILES[@]}"; do
    if [[ ! -f "backend/$f" ]]; then
        MISSING+=("backend/$f")
    fi
done
for f in "${FRONTEND_FILES[@]}"; do
    if [[ ! -f "frontend/$f" ]]; then
        MISSING+=("frontend/$f")
    fi
done
if [[ ! -f "requirements.txt" ]]; then
    MISSING+=("requirements.txt")
fi

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo -e "${RED}[✗] The following required files are missing:${RESET}"
    for f in "${MISSING[@]}"; do
        echo -e "    ${RED}✗${RESET}  $f"
    done
    echo ""
    echo -e "    Make sure you downloaded all files from the project before running install."
    echo -e "    Expected files in $SCRIPT_DIR:"
    echo -e "      app.py  config.py  db.py  extractor.py  indexer.py  similarity.py"
    echo -e "      index.html  requirements.txt  install.sh  start.sh  index.sh"
    exit 1
fi

success "All required files present"

# ── 4. Python version check (3.10+ required) ─────────────────────────────────
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

# ── 5. System packages ────────────────────────────────────────────────────────
info "Installing system packages (ffmpeg, python3-venv)..."
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg python3-venv python3-pip
success "System packages installed"

# ── 6. Virtual environment ────────────────────────────────────────────────────
if [[ -d "venv" ]]; then
    warn "venv already exists — skipping creation. Delete venv/ and re-run to start fresh."
else
    info "Creating virtual environment..."
    $PY -m venv venv
    success "Virtual environment created at $SCRIPT_DIR/venv"
fi

source venv/bin/activate

# ── 7. Pip packages ───────────────────────────────────────────────────────────
info "Upgrading pip..."
pip install --upgrade pip --quiet

info "Installing Python packages (this will take a few minutes — torch is ~800MB)..."
pip install -r requirements.txt
success "Python packages installed"

# ── 8. Verify critical imports ────────────────────────────────────────────────
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

# ── 9. ffmpeg check ───────────────────────────────────────────────────────────
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
echo -e "     bash $SCRIPT_DIR/index.sh /path/to/your/Music"
echo ""
echo -e "  ${CYAN}2. Start the app:${RESET}"
echo -e "     bash $SCRIPT_DIR/start.sh"
echo ""
echo -e "  ${CYAN}3. Open in browser:${RESET}"
echo -e "     http://localhost:5000"
echo ""
echo -e "  ${YELLOW}Note: First index run downloads the CLAP model (~600MB).${RESET}"
echo -e "  ${YELLOW}Each song takes 15-45s to embed on CPU.${RESET}"
echo ""
