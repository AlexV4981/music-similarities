#!/usr/bin/env bash
# =============================================================================
# start.sh — start the musemender web server
# Can be run from anywhere:
#   bash ~/musemender/start.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'
YELLOW='\033[1;33m'; BOLD='\033[1m'; RESET='\033[0m'

if [[ ! -d "venv" ]]; then
    echo -e "${RED}[✗] venv not found. Run install.sh first:${RESET}"
    echo -e "    bash $SCRIPT_DIR/install.sh"
    exit 1
fi

source venv/bin/activate

echo -e "\n${BOLD}musemender${RESET}"

if [[ ! -f "data/vectors.index" ]]; then
    echo -e "${YELLOW}[!] No index found. Index your music library first:${RESET}"
    echo -e "    bash $SCRIPT_DIR/index.sh /path/to/your/Music"
    echo -e "${YELLOW}    (The app will still start — use the Re-index tab in the UI)${RESET}\n"
fi

echo -e "${GREEN}[✓] Open in browser: http://localhost:5000${RESET}"
echo -e "    Press Ctrl+C to stop\n"

cd backend
python3 app.py
