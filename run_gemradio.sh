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

OLLAMA_URL="${GEMRADIO_OLLAMA_HOST:-http://127.0.0.1:11434}"

ollama_up() {
    curl -sf --max-time 2 "$OLLAMA_URL/api/tags" >/dev/null 2>&1
}

wait_for_ollama() {
    for _ in $(seq 1 30); do
        ollama_up && return 0
        sleep 0.5
    done
    return 1
}

# Ollama holds the DJ.  If it is not listening we must be careful how we start
# it: on a machine where Ollama is a system service, its models live under the
# service account, and launching our own `ollama serve` would take port 11434
# from the service, leave it in a restart loop and hide every installed model.
# So prefer the service, and only run our own daemon when there is no service
# to run.
if ! ollama_up; then
    if command -v systemctl >/dev/null 2>&1 \
       && systemctl list-unit-files ollama.service >/dev/null 2>&1 \
       && systemctl cat ollama.service >/dev/null 2>&1; then
        echo "Starting the Ollama system service…"
        systemctl start ollama >/dev/null 2>&1 || true
        if ! wait_for_ollama; then
            echo "Ollama is installed as a system service but is not responding."
            echo "Start it with:  sudo systemctl start ollama"
            echo "The DJ will use its built-in fallback links until it is up."
        fi
    elif command -v ollama >/dev/null 2>&1; then
        echo "Starting a local Ollama daemon…"
        (ollama serve >/dev/null 2>&1 &)
        wait_for_ollama || \
            echo "Ollama did not come up — the DJ will use its built-in fallback links."
    else
        echo "Note: Ollama is not installed — the DJ will use its built-in fallback links."
    fi
fi

exec python3 -m gemradio "$@"
