#!/usr/bin/env bash
# ===============================================================================
# Triune Studio & AI Engine - Automated Linux / macOS / WSL Installer
# ===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}===============================================================================${NC}"
echo -e "${CYAN}         TRIUNE STUDIO - LINUX / MACOS / WSL AUTOMATED INSTALLER${NC}"
echo -e "${CYAN}===============================================================================${NC}"
echo ""

# Detect Python interpreter
PY_BIN=""
for cand in python3 python python3.12 python3.11 python3.10; do
    if command -v "$cand" >/dev/null 2>&1; then
        PY_BIN="$cand"
        break
    fi
done

if [ -z "$PY_BIN" ]; then
    echo -e "${RED}[ERROR] Python 3 was not found in PATH.${NC}"
    echo "Please install Python 3.10, 3.11, or 3.12 using your package manager:"
    echo "  Ubuntu/Debian: sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
    echo "  Fedora:        sudo dnf install -y python3 python3-pip"
    echo "  Arch Linux:    sudo pacman -S python python-pip"
    echo "  macOS:         brew install python@3.11"
    exit 1
fi

echo -e "${GREEN}[✓] Found Python: $($PY_BIN --version) ($PY_BIN)${NC}"
echo ""

# Launch installer.py
exec "$PY_BIN" "$SCRIPT_DIR/installer.py" "$@"
