#!/usr/bin/env bash
set -euo pipefail

# Apply the training kustomization multiple times with different env files.
# Each env file should contain the same keys as infrastructure/k8s/nautilus/training/training.env.

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 ENV_FILE [ENV_FILE...]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="$ROOT_DIR/infrastructure/k8s/nautilus/training"
NAMESPACE="cse239fall2025"

# Validate env files exist.
for env_file in "$@"; do
  if [ ! -f "$env_file" ]; then
    echo "Env file not found: $env_file" >&2
    exit 1
  fi
done

tmpdirs=()
cleanup() {
  for d in "${tmpdirs[@]:-}"; do
    rm -rf "$d"
  done
}
trap cleanup EXIT

for env_file in "$@"; do
  echo "Applying training stack with params from: $env_file"
  tmpdir="$(mktemp -d)"
  tmpdirs+=("$tmpdir")

  # Stage a temp copy of the training kustomization with the desired env file.
  cp -R "$TRAIN_DIR"/. "$tmpdir"/
  cp "$env_file" "$tmpdir/training.env"

  # Set replicas in the StatefulSet based on TRAINER_REPLICAS from the env file (defaults to 3).
  train_replicas="$(grep -E '^TRAINER_REPLICAS=' "$env_file" | tail -n1 | cut -d= -f2-)"
  train_replicas="${train_replicas:-3}"
  perl -0777 -pi -e 's/(replicas:\s*)\d+/\1'"${train_replicas}"'/' "$tmpdir/statefulset-training.yaml"

  kubectl apply -k "$tmpdir"

  # Restart to pick up new env/configmap values.
  kubectl -n "$NAMESPACE" rollout restart statefulset/smirvaki-trainer
done
