#!/usr/bin/env bash
# Starts the Data Synthesizer API + dashboard on one origin.
# Usage: ./run.sh [port]   (default 8765)
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-8765}"
PY="${VIRTUAL_ENV:-.venv}/bin/python"
if [ ! -x "$PY" ]; then PY="python3"; fi
echo "Starting Data Synthesizer API + dashboard on http://127.0.0.1:${PORT}/"
"$PY" api/server.py "$PORT" &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null" EXIT
sleep 1
URL="http://127.0.0.1:${PORT}/"
if command -v open >/dev/null 2>&1; then open "$URL"
elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
else echo "Open ${URL} in your browser."
fi
wait $SERVER_PID
