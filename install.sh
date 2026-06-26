#!/usr/bin/env bash
# =============================================================================
# install.sh — musemender setup
# Run from inside the musemender folder: bash install.sh
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[musemender]${RESET} $*"; }
success() { echo -e "${GREEN}[✓]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[!]${RESET} $*"; }
fail()    { echo -e "${RED}[✗]${RESET} $*"; exit 1; }

echo -e "\n${BOLD}musemender — install${RESET}\n"

# ── 1. Create directory structure ─────────────────────────────────────────────
info "Creating directory structure..."
mkdir -p backend frontend data uploads
chmod 755 data uploads 2>/dev/null || true
chown -R "$(whoami):$(whoami)" data uploads 2>/dev/null || true
success "Directories ready"

# ── 2. Move flat files into subfolders if needed ──────────────────────────────
BACKEND_FILES=(app.py config.py db.py extractor.py indexer.py similarity.py valence.py)
MOVED=0

for f in "${BACKEND_FILES[@]}"; do
    [[ -f "$f" ]] && mv "$f" "backend/$f" && info "  Moved $f → backend/" && MOVED=$((MOVED+1))
done
[[ -f "index.html" ]] && mv "index.html" "frontend/index.html" && info "  Moved index.html → frontend/" && MOVED=$((MOVED+1)) || true

[[ $MOVED -gt 0 ]] && success "Moved $MOVED file(s)" || info "Files already in subfolders"

# ── 3. Check all required files are present ───────────────────────────────────
info "Checking required files..."
MISSING=()
for f in "${BACKEND_FILES[@]}"; do
    [[ ! -f "backend/$f" ]] && MISSING+=("backend/$f")
done
[[ ! -f "frontend/index.html" ]] && MISSING+=("frontend/index.html")
[[ ! -f "requirements.txt" ]]    && MISSING+=("requirements.txt")

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo -e "${RED}[✗] Missing files:${RESET}"
    for f in "${MISSING[@]}"; do echo -e "    ${RED}✗${RESET}  $f"; done
    echo ""
    echo -e "    Make sure all project files are in this folder before running install."
    echo -e "    Expected: app.py config.py db.py extractor.py indexer.py similarity.py valence.py"
    echo -e "            index.html requirements.txt install.sh start.sh index.sh"
    exit 1
fi
success "All source files present"

# ── 4. Detect best available Python 3.10+ ────────────────────────────────────
# Tries all candidates so it works across Ubuntu 20.04 / 22.04 / 24.04
info "Finding Python 3.10+..."
PY=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        VER=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        MAJOR=$(echo "$VER" | cut -d. -f1)
        MINOR=$(echo "$VER" | cut -d. -f2)
        if [[ "$MAJOR" -eq 3 ]] && [[ "$MINOR" -ge 10 ]]; then
            PY="$candidate"
            success "Found $candidate ($VER)"
            break
        fi
    fi
done

if [[ -z "$PY" ]]; then
    echo -e "${RED}[✗] No Python 3.10+ found.${RESET}"
    echo ""
    echo -e "  Ubuntu 22.04 / 24.04:  ${CYAN}sudo apt install python3 python3-venv${RESET}"
    echo -e "  Ubuntu 20.04:          ${CYAN}sudo add-apt-repository ppa:deadsnakes/ppa${RESET}"
    echo -e "                         ${CYAN}sudo apt update && sudo apt install python3.10 python3.10-venv${RESET}"
    exit 1
fi

# ── 5. System packages ────────────────────────────────────────────────────────
info "Installing system packages..."
sudo apt-get update -qq 2>/dev/null || warn "apt-get update had warnings — continuing"
sudo apt-get install -y -qq ffmpeg python3-venv python3-pip 2>/dev/null || \
    sudo apt-get install -y ffmpeg python3-venv python3-pip
success "System packages installed"

# ── 6. Virtual environment ────────────────────────────────────────────────────
# --copies: copies the interpreter binary instead of symlinking
# so the venv still works if the folder is moved or the drive remounts
if [[ -d "venv" ]]; then
    if [[ ! -f "venv/bin/python3" ]]; then
        warn "venv looks broken — rebuilding..."
        rm -rf venv
        "$PY" -m venv --copies venv
        success "venv rebuilt"
    else
        warn "venv already exists — skipping. Delete venv/ and re-run to rebuild."
    fi
else
    info "Creating virtual environment..."
    "$PY" -m venv --copies venv
    success "venv created"
fi

source venv/bin/activate

# ── 7. Python packages ────────────────────────────────────────────────────────
info "Upgrading pip..."
pip install --upgrade pip --quiet
info "Installing packages (torch is ~800MB on first run)..."
pip install -r requirements.txt
success "Packages installed"

# ── 8. Verify all imports ─────────────────────────────────────────────────────
info "Verifying imports..."
python3 - <<'PYCHECK'
import sys
failures = []
checks = [
    ("torch",        "import torch; assert torch.__version__"),
    ("torchaudio",   "import torchaudio"),
    ("transformers", "from transformers import ClapModel, ClapProcessor"),
    ("librosa",      "import librosa"),
    ("faiss",        "import faiss"),
    ("flask",        "from flask import Flask"),
    ("flask_cors",   "from flask_cors import CORS"),
    ("gunicorn",     "import gunicorn"),
    ("mutagen",      "from mutagen import File"),
    ("numpy",        "import numpy"),
    ("tqdm",         "from tqdm import tqdm"),
    ("soundfile",    "import soundfile"),
    ("werkzeug",     "from werkzeug.utils import secure_filename"),
]
for name, stmt in checks:
    try:
        exec(stmt)
        print(f"  \033[32m✓\033[0m  {name}")
    except Exception as e:
        print(f"  \033[31m✗\033[0m  {name}: {e}")
        failures.append(name)
if failures:
    print(f"\n\033[31mFailed: {', '.join(failures)}\033[0m")
    sys.exit(1)
print("\n\033[32mAll imports OK\033[0m")
PYCHECK
success "All imports verified"

# ── 9. ffmpeg ─────────────────────────────────────────────────────────────────
command -v ffmpeg &>/dev/null \
    && success "ffmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')" \
    || warn "ffmpeg not found — .wma/.aac/.m4a conversion will fail"

# ── 10. SQLite write test ─────────────────────────────────────────────────────
info "Testing SQLite write..."
python3 - <<'DBTEST'
import sqlite3, os, sys
try:
    conn = sqlite3.connect(os.path.join("data", "_test.db"))
    conn.execute("CREATE TABLE IF NOT EXISTS t (x INTEGER)")
    conn.close()
    os.unlink(os.path.join("data", "_test.db"))
    print("  \033[32m✓\033[0m  SQLite write test passed")
except Exception as e:
    print(f"  \033[31m✗\033[0m  SQLite write FAILED: {e}")
    print("      Fix: sudo chown -R $(whoami) data/")
    sys.exit(1)
DBTEST

# ── 11. Final checklist ───────────────────────────────────────────────────────
echo ""
info "Final checklist..."
ISSUES=()
for f in "${BACKEND_FILES[@]}"; do [[ ! -f "backend/$f" ]] && ISSUES+=("MISSING: backend/$f"); done
[[ ! -f "frontend/index.html" ]] && ISSUES+=("MISSING: frontend/index.html")
[[ ! -f "requirements.txt" ]]    && ISSUES+=("MISSING: requirements.txt")
[[ ! -d "venv" ]]                && ISSUES+=("MISSING: venv/")
[[ ! -w "data" ]]                && ISSUES+=("NOT WRITABLE: data/")
[[ ! -w "uploads" ]]             && ISSUES+=("NOT WRITABLE: uploads/")

if [[ ${#ISSUES[@]} -gt 0 ]]; then
    echo -e "${RED}[✗] Issues found:${RESET}"
    for i in "${ISSUES[@]}"; do echo -e "    ${RED}✗${RESET}  $i"; done
    exit 1
fi

echo -e "  ${GREEN}✓${RESET}  backend/    (7 Python files)"
echo -e "  ${GREEN}✓${RESET}  frontend/   (index.html)"
echo -e "  ${GREEN}✓${RESET}  data/       (writable)"
echo -e "  ${GREEN}✓${RESET}  uploads/    (writable)"
echo -e "  ${GREEN}✓${RESET}  venv/       ($(venv/bin/python3 --version))"
echo -e "  ${GREEN}✓${RESET}  SQLite write test passed"

echo ""
echo -e "${GREEN}${BOLD}Installation complete.${RESET}"
echo ""
echo -e "  ${CYAN}1. Index your library:${RESET}  bash index.sh /path/to/Music"
echo -e "  ${CYAN}2. Start the app:${RESET}        bash start.sh"
echo -e "  ${CYAN}3. Open browser:${RESET}         http://localhost:5000"
echo ""
echo -e "  ${YELLOW}First index run downloads CLAP (~600MB). Each song: 15-45s on CPU.${RESET}"
echo ""
