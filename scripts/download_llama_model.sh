#!/usr/bin/env bash
set -euo pipefail

MODEL_BASENAME="$(basename "${LLAMA_MODEL_PATH:-}")"
TARGET_DIR="${LLAMA_MODEL_DIR:-data/models}"

if [[ -z "${MODEL_BASENAME}" ]]; then
  echo "LLAMA_MODEL_PATH is not set in the environment" >&2
  exit 1
fi

HF_MODEL_REPO_DEFAULT="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
HF_MODEL_REPO="${HF_MODEL_REPO:-$HF_MODEL_REPO_DEFAULT}"
HF_MODEL_FILE="${HF_MODEL_FILE:-$MODEL_BASENAME}"

mkdir -p "$TARGET_DIR"
export HF_MODEL_REPO HF_MODEL_FILE TARGET_DIR

echo "Downloading $HF_MODEL_FILE from $HF_MODEL_REPO to $TARGET_DIR..."

python - <<'PY'
import os
from huggingface_hub import hf_hub_download

repo = os.environ["HF_MODEL_REPO"]
filename = os.environ["HF_MODEL_FILE"]
target_dir = os.environ["TARGET_DIR"]
token = os.environ.get("HF_TOKEN") or None

path = hf_hub_download(
    repo_id=repo,
    filename=filename,
    local_dir=target_dir,
    local_dir_use_symlinks=False,
    token=token,
)
print(f"Done. Saved to {path}")
PY
