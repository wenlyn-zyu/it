#!/bin/bash
# Claude Code wrapper for zhuwl2022/it
# Sources .env and launches Claude from project directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source .env to export API keys to environment
set -a
source "$SCRIPT_DIR/.env"
set +a

# Find and run Claude (via nvm)
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"

exec claude "$@"
