#!/usr/bin/env bash
# =============================================================================
# start.sh — start the musemender web server
# Usage: bash start.sh
# =============================================================================
set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'; RESET='\033[0m'; BOLD='\033[1m'

if [[ ! -f "backend/app.py" ]]; then
    echo -e "${RED}[✗] Run this from the musemender directory: cd musemender && bash start.sh${RESET}"
    exit 1
fi

if [[ ! -d "venv" ]]; then
    echo -e "${RED}[✗] venv not found. Run install.sh first.${RESET}"
    exit 1
fi

source venv/bin/activate

echo -e "\n${BOLD}musemender${RESET}"
echo -e "${CYAN}Starting server...${RESET}"

# Check if index exists
if [[ ! -f "data/vectors.index" ]]; then
    echo -e "\n${RED}[!] No index found.${RESET}"
    echo -e "    Index your music library first:"
    echo -e "    ${CYAN}bash index.sh /path/to/your/Music${RESET}\n"
fi

echo -e "${GREEN}[✓] Open in browser: http://localhost:5000${RESET}"
echo -e "    Press Ctrl+C to stop\n"

cd backend
python3 app.py
