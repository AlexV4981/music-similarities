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

# ── 1. Create subdirectory structure with correct permissions ─────────────────
info "Creating directory structure..."
mkdir -p "$SCRIPT_DIR/backend" \
         "$SCRIPT_DIR/frontend" \
         "$SCRIPT_DIR/data" \
         "$SCRIPT_DIR/uploads"

# Ensure the current user owns all dirs (fixes SQLite permission errors)
chown -R "$(whoami):$(whoami)" "$SCRIPT_DIR/data" "$SCRIPT_DIR/uploads" 2>/dev/null || true
chmod 755 "$SCRIPT_DIR/data" "$SCRIPT_DIR/uploads"

success "Directories ready (data/ and uploads/ are writable)"

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

# ── 3. Verify all required source files are present ──────────────────────────
info "Checking all required files..."

MISSING=()
for f in "${BACKEND_FILES[@]}"; do
    [[ ! -f "$SCRIPT_DIR/backend/$f" ]] && MISSING+=("backend/$f")
done
for f in "${FRONTEND_FILES[@]}"; do
    [[ ! -f "$SCRIPT_DIR/frontend/$f" ]] && MISSING+=("frontend/$f")
done
[[ ! -f "$SCRIPT_DIR/requirements.txt" ]] && MISSING+=("requirements.txt")

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo -e "${RED}[✗] The following required files are missing:${RESET}"
    for f in "${MISSING[@]}"; do
        echo -e "    ${RED}✗${RESET}  $f"
    done
    echo ""
    echo -e "    Make sure you downloaded all files before running install."
    echo -e "    Expected files in $SCRIPT_DIR:"
    echo -e "      app.py  config.py  db.py  extractor.py  indexer.py  similarity.py"
    echo -e "      index.html  requirements.txt  install.sh  start.sh  index.sh"
    exit 1
fi

success "All required source files present"

# ── 4. Python version check (3.10+ required) ─────────────────────────────────
info "Checking Python version..."
PY=$(command -v python3 || true)
[[ -z "$PY" ]] && fail "python3 not found. Install it: sudo apt install python3"

PY_VER=$($PY -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PY -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PY -c "import sys; print(sys.version_info.minor)")

if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 10 ]]; }; then
    fail "Python 3.10+ required. Found: $PY_VER\nOn Ubuntu 20.04: sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.10"
fi
success "Python $PY_VER"

# ── 5. System packages ────────────────────────────────────────────────────────
info "Installing system packages (ffmpeg, python3-venv)..."
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg python3-venv python3-pip
success "System packages installed"

# ── 6. Virtual environment ────────────────────────────────────────────────────
if [[ -d "$SCRIPT_DIR/venv" ]]; then
    warn "venv already exists — skipping creation. Delete venv/ and re-run to start fresh."
else
    info "Creating virtual environment..."
    $PY -m venv "$SCRIPT_DIR/venv"
    success "Virtual environment created"
fi

source "$SCRIPT_DIR/venv/bin/activate"

# ── 7. Pip packages ───────────────────────────────────────────────────────────
info "Upgrading pip..."
pip install --upgrade pip --quiet

info "Installing Python packages (this will take a few minutes — torch is ~800MB)..."
pip install -r "$SCRIPT_DIR/requirements.txt"
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

# ── 10. Final sanity check — everything in place and writable ────────────────
echo ""
info "Running final checks..."

FINAL_ISSUES=()

# Source files
for f in "${BACKEND_FILES[@]}"; do
    [[ ! -f "$SCRIPT_DIR/backend/$f" ]] && FINAL_ISSUES+=("MISSING: backend/$f")
done
[[ ! -f "$SCRIPT_DIR/frontend/index.html" ]] && FINAL_ISSUES+=("MISSING: frontend/index.html")
[[ ! -f "$SCRIPT_DIR/requirements.txt" ]]    && FINAL_ISSUES+=("MISSING: requirements.txt")

# Runtime directories exist and are writable
for dir in data uploads; do
    [[ ! -d "$SCRIPT_DIR/$dir" ]]    && FINAL_ISSUES+=("MISSING DIR: $dir/")
    [[ ! -w "$SCRIPT_DIR/$dir" ]]    && FINAL_ISSUES+=("NOT WRITABLE: $dir/ — run: sudo chown -R $(whoami) $SCRIPT_DIR/$dir")
done

# venv exists
[[ ! -f "$SCRIPT_DIR/venv/bin/python3" ]] && FINAL_ISSUES+=("MISSING: venv (virtualenv not created)")

# SQLite write test
python3 - <<DBTEST 2>/dev/null || FINAL_ISSUES+=("SQLITE ERROR: cannot write to data/library.db — check permissions on data/")
import sqlite3, os
db = os.path.join("$SCRIPT_DIR", "data", "test_write.db")
conn = sqlite3.connect(db)
conn.execute("CREATE TABLE IF NOT EXISTS t (x INTEGER)")
conn.close()
os.unlink(db)
DBTEST

if [[ ${#FINAL_ISSUES[@]} -gt 0 ]]; then
    echo ""
    echo -e "${RED}[✗] Final check failed — issues found:${RESET}"
    for issue in "${FINAL_ISSUES[@]}"; do
        echo -e "    ${RED}✗${RESET}  $issue"
    done
    echo ""
    echo -e "    Fix the above and re-run ${CYAN}bash install.sh${RESET}"
    exit 1
fi

echo ""
echo -e "  ${GREEN}✓${RESET}  backend/          (6 Python files)"
echo -e "  ${GREEN}✓${RESET}  frontend/         (index.html)"
echo -e "  ${GREEN}✓${RESET}  data/             (writable — SQLite + FAISS index)"
echo -e "  ${GREEN}✓${RESET}  uploads/          (writable — temp files)"
echo -e "  ${GREEN}✓${RESET}  venv/             (Python packages)"
echo -e "  ${GREEN}✓${RESET}  requirements.txt"
echo -e "  ${GREEN}✓${RESET}  SQLite write test passed"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}Installation complete. Everything is in place.${RESET}"
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
