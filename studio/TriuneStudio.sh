#!/usr/bin/env bash
cd "$(dirname "$0")/.."
if [ -f "studio/studio_env/bin/python" ]; then
    studio/studio_env/bin/python scripts/launch_studio.py "$@"
elif [ -f ".venv/bin/python" ]; then
    .venv/bin/python scripts/launch_studio.py "$@"
else
    python3 scripts/launch_studio.py "$@"
fi
