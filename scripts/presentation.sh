#!/usr/bin/env bash
set -Eeuo pipefail

# Demo-Night deck launcher.
# Serves the "Sovereign world models" reveal.js deck over http (reveal.js needs
# http://, not file://) and opens it in the browser. The deck is a static page,
# so this only needs python3, no backend.
#
# Keys once it is open:  F fullscreen  ·  S speaker view (notes + timer)  ·  Esc overview
#
# Common overrides:
#   PRESENTATION_PORT=8090 scripts/presentation.sh
#   PRESENTATION_BROWSER_APP="Google Chrome" scripts/presentation.sh
#   PRESENTATION_OPEN_BROWSER=0 scripts/presentation.sh   # just serve, do not open
#   PRESENTATION_SPEAKER=1 scripts/presentation.sh        # also open speaker view

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DECK_DIR="${PRESENTATION_DIR:-${ROOT_DIR}/docs/plans/hackathons/codex-hackathon-paris-june/presentation}"
HOST="${PRESENTATION_HOST:-127.0.0.1}"
PORT="${PRESENTATION_PORT:-8088}"
BASE_URL="${PRESENTATION_BASE_URL:-http://${HOST}:${PORT}}"
DECK_URL="${PRESENTATION_URL:-${BASE_URL}/}"
OPEN_BROWSER="${PRESENTATION_OPEN_BROWSER:-1}"
OPEN_SPEAKER="${PRESENTATION_SPEAKER:-0}"
WAIT_FOR_CTRL_C="${PRESENTATION_WAIT:-1}"
SERVER_PID=""
STARTED_SERVER=0

info() { printf '[deck] %s\n' "$*"; }
fail() { printf '[deck] ERROR: %s\n' "$*" >&2; exit 1; }

cleanup() {
  if [[ "${STARTED_SERVER}" == "1" && -n "${SERVER_PID}" ]]; then
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      info "Stopping deck server pid ${SERVER_PID}"
      kill "${SERVER_PID}" 2>/dev/null || true
      # wait for a graceful exit, then force it so the port never leaks
      for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "${SERVER_PID}" 2>/dev/null || break
        sleep 0.2
      done
      kill -9 "${SERVER_PID}" 2>/dev/null || true
      wait "${SERVER_PID}" 2>/dev/null || true
    fi
  fi
  STARTED_SERVER=0
}
trap cleanup EXIT INT TERM

# Returns 0 only when the URL is served by OUR deck (so we never reuse, or open,
# a foreign server squatting on the port).
deck_is_ready() {
  curl -fsS "${DECK_URL}" 2>/dev/null | grep -q "Sovereign world models"
}

wait_for_deck() {
  local deadline="${PRESENTATION_STARTUP_TIMEOUT:-20}"
  local start
  start="$(date +%s)"
  while true; do
    if deck_is_ready; then
      info "Deck ready: ${DECK_URL}"
      return 0
    fi
    if (( "$(date +%s)" - start >= deadline )); then
      return 1
    fi
    sleep 0.4
  done
}

open_url() {
  local url="$1"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    if [[ -n "${PRESENTATION_BROWSER_APP:-}" ]]; then
      open -a "${PRESENTATION_BROWSER_APP}" "${url}"
    else
      open "${url}"
    fi
    return 0
  fi
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${url}" >/dev/null 2>&1 &
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 -m webbrowser -t "${url}" >/dev/null 2>&1 || true
    return 0
  fi
  info "No browser opener found. Open this manually: ${url}"
}

open_tabs() {
  if [[ "${OPEN_BROWSER}" == "0" ]]; then
    info "Browser opening disabled. Open: ${DECK_URL}"
    return 0
  fi
  open_url "${DECK_URL}"
  if [[ "${OPEN_SPEAKER}" == "1" ]]; then
    sleep 0.4
    info "Tip: speaker view also opens from any slide with the S key."
  fi
}

main() {
  command -v curl >/dev/null 2>&1 || fail "curl is required"
  command -v python3 >/dev/null 2>&1 || fail "python3 is required to serve the deck"
  [[ -f "${DECK_DIR}/index.html" ]] || fail "deck not found at ${DECK_DIR}/index.html"

  info "Deck: ${DECK_DIR}"
  info "URL:  ${DECK_URL}"

  if deck_is_ready; then
    info "Reusing the deck already served on ${BASE_URL}"
  else
    # Fail clearly if something foreign already holds the port.
    if curl -fsS "${BASE_URL}/" >/dev/null 2>&1; then
      fail "${BASE_URL} is already in use by another server. Set PRESENTATION_PORT to a free port."
    fi
    info "Serving on ${HOST}:${PORT}"
    python3 -m http.server "${PORT}" --bind "${HOST}" --directory "${DECK_DIR}" >/dev/null 2>&1 &
    SERVER_PID="$!"
    STARTED_SERVER=1
    wait_for_deck || fail "deck did not come up at ${DECK_URL} (check that port ${PORT} is free)"
  fi

  open_tabs

  info "Deck is open. F fullscreen · S speaker view · Esc overview · arrows to navigate."
  if [[ "${STARTED_SERVER}" == "1" ]]; then
    info "Press Ctrl+C here to stop the server when you are done."
  fi
  if [[ "${WAIT_FOR_CTRL_C}" == "1" ]]; then
    if [[ "${STARTED_SERVER}" == "1" ]]; then
      wait "${SERVER_PID}"
    else
      while true; do sleep 3600; done
    fi
  fi
}

main "$@"
