#!/usr/bin/env bash
set -euo pipefail

BACKEND="${LLAMA_BACKEND:-python}"
MODEL="${LLAMA_MODEL_PATH:-/models/model.gguf}"
CTX="${LLAMA_CONTEXT:-2048}"
THREADS="${LLAMA_THREADS:-4}"
NBATCH="${LLAMA_BATCH:-128}"
SERVER_HOST="${LLAMA_SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${LLAMA_SERVER_PORT:-8080}"
UVICORN_WORKERS="${UVICORN_WORKERS:-2}"
EXTRA_FLAGS="${LLAMA_SERVER_EXTRA_FLAGS:-}"

WAIT_FOR_MODEL_TIMEOUT="${WAIT_FOR_MODEL_TIMEOUT:-600}"
WAIT_FOR_MODEL_POLL="${WAIT_FOR_MODEL_POLL:-5}"

echo "Waiting for model at ${MODEL} (timeout=${WAIT_FOR_MODEL_TIMEOUT}s, poll=${WAIT_FOR_MODEL_POLL}s)..."
deadline=$((SECONDS + WAIT_FOR_MODEL_TIMEOUT))
while [ ! -s "${MODEL}" ]; do
  if [ "${SECONDS}" -ge "${deadline}" ]; then
    echo "Model ${MODEL} not found after ${WAIT_FOR_MODEL_TIMEOUT}s" >&2
    exit 1
  fi
  sleep "${WAIT_FOR_MODEL_POLL}"
done
echo "Model found: ${MODEL}"

if [[ "${BACKEND}" == "server" ]]; then
  echo "Starting standalone llama-server (backend=server)..."
  /usr/local/bin/llama-server -m "${MODEL}" --host "${SERVER_HOST}" --port "${SERVER_PORT}" -c "${CTX}" -t "${THREADS}" --batch-size "${NBATCH}" ${EXTRA_FLAGS} &
  # Point proxy at local server if not already set
  export LLAMA_SERVER_URL="${LLAMA_SERVER_URL:-http://127.0.0.1:${SERVER_PORT}}"
fi

echo "Starting uvicorn with backend=${BACKEND} ..."
exec python3 -m uvicorn serving.api:app --host 0.0.0.0 --port 8001 --workers "${UVICORN_WORKERS}"
