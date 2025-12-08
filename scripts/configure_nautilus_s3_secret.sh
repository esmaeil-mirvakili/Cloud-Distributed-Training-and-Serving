#!/usr/bin/env bash

# Helper script to create or update the Nautilus S3 credential secret used by
# infrastructure/k8s/nautilus/job-export-model.yaml.
#
# Usage:
#   ./scripts/configure_nautilus_s3_secret.sh --namespace cse239fall2025 --env-file .env
# Options:
#   --namespace <ns>     Kubernetes namespace (default: cse239fall2025)
#   --secret-name <name> Secret name (default: nautilus-s3-credentials)
#   --env-file <path>    Path to a .env file with ACCESS_KEY/SECRET_KEY (required).

set -euo pipefail

NAMESPACE="cse239fall2025"
SECRET_NAME="nautilus-s3-credentials"
ENV_FILE=""
ACCESS_KEY=""
SECRET_KEY=""

usage() {
  grep '^#' "$0" | cut -c 4-
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    --secret-name)
      SECRET_NAME="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      ;;
  esac
done

if [[ -z "${ENV_FILE}" ]]; then
  echo "--env-file is required (point it to a file defining ACCESS_KEY/SECRET_KEY or S3_ACCESS_KEY/S3_SECRET_KEY)." >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Specified env file '${ENV_FILE}' not found." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

ACCESS_KEY="${ACCESS_KEY:-${S3_ACCESS_KEY:-}}"
SECRET_KEY="${SECRET_KEY:-${S3_SECRET_KEY:-}}"

if [[ -z "${ACCESS_KEY}" || -z "${SECRET_KEY}" ]]; then
  echo "Error: ACCESS_KEY/SECRET_KEY (or S3_ACCESS_KEY/S3_SECRET_KEY) must be set in ${ENV_FILE}." >&2
  exit 1
fi

echo "Creating/updating secret ${SECRET_NAME} in namespace ${NAMESPACE}..."

kubectl -n "${NAMESPACE}" create secret generic "${SECRET_NAME}" \
  --from-literal=access_key="${ACCESS_KEY}" \
  --from-literal=secret_key="${SECRET_KEY}" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Secret ${SECRET_NAME} configured."
