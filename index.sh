#!/usr/bin/env bash
# =============================================================================
# index.sh — index your music library
# Can be run from anywhere:
#   bash ~/musemender/index.sh /path/to/Music
#   bash ~/musemender/index.sh /path/to/Music --update
#   bash ~/musemender/index.sh /path/to/Music --clean
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CYAN='\033[0;36m'; RED='\033[0;31m'; YELLOW='\033[1;33m'
BOLD='\033[1m'; RESET='\033[0m'

if [[ ! -d "venv" ]]; then
    echo -e "${RED}[✗] venv not found. Run install.sh first:${RESET}"
    echo -e "    bash $SCRIPT_DIR/install.sh"
    exit 1
fi

if [[ $# -lt 1 ]]; then
    echo -e "${RED}[✗] Usage:${RESET}"
    echo -e "    bash index.sh /path/to/Music            # full index"
    echo -e "    bash index.sh /path/to/Music --update   # skip unchanged files"
    echo -e "    bash index.sh /path/to/Music --clean    # remove deleted files"
    exit 1
fi

MUSIC_DIR="$1"
EXTRA_ARGS="${@:2}"

if [[ ! -d "$MUSIC_DIR" ]]; then
    echo -e "${RED}[✗] Directory not found: $MUSIC_DIR${RESET}"
    exit 1
fi

source venv/bin/activate

echo -e "\n${BOLD}musemender — indexer${RESET}"
echo -e "${CYAN}Music directory:${RESET} $MUSIC_DIR"

if [[ "${EXTRA_ARGS:-}" == *"--update"* ]]; then
    echo -e "${CYAN}Mode:${RESET} update (skip unchanged files)"
elif [[ "${EXTRA_ARGS:-}" == *"--clean"* ]]; then
    echo -e "${CYAN}Mode:${RESET} clean (remove stale entries)"
else
    echo -e "${CYAN}Mode:${RESET} full index"
fi

echo -e "${YELLOW}[!] First run downloads the CLAP model (~600MB). Each song takes 15-45s on CPU.${RESET}\n"

cd backend
python3 indexer.py "$MUSIC_DIR" ${EXTRA_ARGS:-}
