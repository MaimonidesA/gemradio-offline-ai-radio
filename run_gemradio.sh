#!/usr/bin/env bash
# GemRadio — start the station.
#
#   ./run_gemradio.sh            front panel (default)
#   ./run_gemradio.sh doctor     check every component
#   ./run_gemradio.sh scan       re-index the music library
#   ./run_gemradio.sh headless   broadcast in the terminal
#
# Everything runs locally: no network calls leave this machine.

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# Nothing in this project may reach the internet.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true
export NO_PROXY="*"
export PYTHONDONTWRITEBYTECODE=1

if ! command -v python3 >/dev/null 2>&1; then
    echo "GemRadio needs python3." >&2
    exit 1
fi

# Ollama holds the DJ. Start it if it is installed but not listening.
if ! curl -sf --max-time 2 "${GEMRADIO_OLLAMA_HOST:-http://127.0.0.1:11434}/api/tags" >/dev/null 2>&1; then
    if command -v ollama >/dev/null 2>&1; then
        echo "Starting the local Ollama daemon…"
        (ollama serve >/dev/null 2>&1 &)
        for _ in $(seq 1 20); do
            sleep 0.5
            curl -sf --max-time 2 "${GEMRADIO_OLLAMA_HOST:-http://127.0.0.1:11434}/api/tags" >/dev/null 2>&1 && break
        done
    else
        echo "Note: Ollama is not running — the DJ will use its built-in fallback links."
    fi
fi

exec python3 -m gemradio "$@"
