#!/usr/bin/env bash
set -Eeuo pipefail

# Live hackathon launcher.
# Starts the Lensemble demo backend, opens the federated orchestrator dashboard,
# and opens the surprise-meter page. The federated demo server serves both web
# surfaces from the same process.
#
# Common overrides:
#   DEMO_PORT=8765 scripts/demo.sh
#   DEMO_BROWSER_APP="Google Chrome" scripts/demo.sh
#   DEMO_OPEN_BROWSER=0 scripts/demo.sh
#   DEMO_SURPRISE_URL="http://127.0.0.1:8765/web/surprise-meter/?engine=fallback&mode=post" scripts/demo.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${DEMO_HOST:-127.0.0.1}"
PORT="${DEMO_PORT:-8765}"
BASE_URL="${DEMO_BASE_URL:-http://${HOST}:${PORT}}"
ORCHESTRATOR_URL="${DEMO_ORCHESTRATOR_URL:-${BASE_URL}/web/federated-demo/}"
SURPRISE_URL="${DEMO_SURPRISE_URL:-${BASE_URL}/web/surprise-meter/?engine=auto&mode=post}"
OPEN_BROWSER="${DEMO_OPEN_BROWSER:-1}"
WAIT_FOR_CTRL_C="${DEMO_WAIT:-1}"
DEPLOYMENT_TARGET="${DEMO_DEPLOYMENT_TARGET:-local}"
LOG_DIR="${DEMO_LOG_DIR:-${ROOT_DIR}/runs/demo}"
LOG_FILE="${LOG_DIR}/demo-server.log"
SERVER_PID=""
STARTED_SERVER=0

info() {
  printf '[demo] %s\n' "$*"
}

fail() {
  printf '[demo] ERROR: %s\n' "$*" >&2
  if [[ -f "${LOG_FILE}" ]]; then
    printf '\n[demo] Last server log lines:\n' >&2
    tail -n 80 "${LOG_FILE}" >&2 || true
  fi
  exit 1
}

cleanup() {
  if [[ "${STARTED_SERVER}" == "1" && -n "${SERVER_PID}" ]]; then
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      info "Stopping demo server pid ${SERVER_PID}"
      kill "${SERVER_PID}" 2>/dev/null || true
      wait "${SERVER_PID}" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT INT TERM

wait_for_url() {
  local url="$1"
  local label="$2"
  local deadline="${DEMO_STARTUP_TIMEOUT:-45}"
  local start
  start="$(date +%s)"
  while true; do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      info "${label} ready: ${url}"
      return 0
    fi
    if (( "$(date +%s)" - start >= deadline )); then
      return 1
    fi
    sleep 0.5
  done
}

server_is_ready() {
  curl -fsS "${ORCHESTRATOR_URL}" >/dev/null 2>&1 &&
    curl -fsS "${SURPRISE_URL}" >/dev/null 2>&1
}

open_tabs() {
  if [[ "${OPEN_BROWSER}" == "0" ]]; then
    info "Browser opening disabled"
    return 0
  fi

  if [[ "$(uname -s)" == "Darwin" ]]; then
    if [[ -n "${DEMO_BROWSER_APP:-}" ]]; then
      open -a "${DEMO_BROWSER_APP}" "${ORCHESTRATOR_URL}"
      sleep 0.4
      open -a "${DEMO_BROWSER_APP}" "${SURPRISE_URL}"
    else
      open "${ORCHESTRATOR_URL}"
      sleep 0.4
      open "${SURPRISE_URL}"
    fi
    return 0
  fi

  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${ORCHESTRATOR_URL}" >/dev/null 2>&1 &
    sleep 0.4
    xdg-open "${SURPRISE_URL}" >/dev/null 2>&1 &
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    python3 -m webbrowser -t "${ORCHESTRATOR_URL}" >/dev/null 2>&1 || true
    sleep 0.4
    python3 -m webbrowser -t "${SURPRISE_URL}" >/dev/null 2>&1 || true
    return 0
  fi

  info "Could not find a browser opener. Open these manually:"
  info "  ${ORCHESTRATOR_URL}"
  info "  ${SURPRISE_URL}"
}

main() {
  cd "${ROOT_DIR}"
  mkdir -p "${LOG_DIR}"

  command -v uv >/dev/null 2>&1 || fail "uv is required"
  command -v curl >/dev/null 2>&1 || fail "curl is required"

  info "Repository: ${ROOT_DIR}"
  info "Orchestrator: ${ORCHESTRATOR_URL}"
  info "Surprise meter: ${SURPRISE_URL}"
  info "Server log: ${LOG_FILE}"

  if server_is_ready; then
    info "Reusing existing demo server on ${BASE_URL}"
  else
    local server_cmd=(
      uv run lensemble demo federated
      --host "${HOST}"
      --port "${PORT}"
      --deployment-target "${DEPLOYMENT_TARGET}"
    )
    if [[ -n "${DEMO_PUBLIC_BASE_URL:-}" ]]; then
      server_cmd+=(--public-base-url "${DEMO_PUBLIC_BASE_URL}")
    fi
    if [[ "${DEMO_PUBLIC_DEMO:-0}" == "1" ]]; then
      server_cmd+=(--public-demo)
    fi

    : >"${LOG_FILE}"
    info "Starting federated demo server on ${HOST}:${PORT}"
    "${server_cmd[@]}" >"${LOG_FILE}" 2>&1 &
    SERVER_PID="$!"
    STARTED_SERVER=1

    wait_for_url "${ORCHESTRATOR_URL}" "orchestrator" ||
      fail "orchestrator did not become ready at ${ORCHESTRATOR_URL}"
    wait_for_url "${SURPRISE_URL}" "surprise meter" ||
      fail "surprise meter did not become ready at ${SURPRISE_URL}"
  fi

  open_tabs

  info "Presentation tabs are open."
  info "Press Ctrl+C in this terminal to stop the server when you are done."
  if [[ "${WAIT_FOR_CTRL_C}" == "1" ]]; then
    if [[ "${STARTED_SERVER}" == "1" ]]; then
      wait "${SERVER_PID}"
    else
      while true; do sleep 3600; done
    fi
  fi
}

main "$@"
