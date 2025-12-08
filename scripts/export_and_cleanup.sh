#!/usr/bin/env bash
# Uploads the model artifacts stored on the Nautilus PVC to S3 and deletes PVCs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAMESPACE="${NAMESPACE:-cse239fall2025}"
JOB_MANIFEST="${JOB_MANIFEST:-infrastructure/k8s/nautilus/job-export-model.yaml}"
JOB_NAME="${JOB_NAME:-smirvaki-export-model}"
PVC_DATA="${PVC_DATA:-smirvaki-training-data}"
PVC_OUTPUTS="${PVC_OUTPUTS:-smirvaki-training-outputs}"

echo "Applying ${JOB_MANIFEST} to start S3 export job..."
kubectl apply -f "${REPO_ROOT}/${JOB_MANIFEST}"

echo "Waiting for job/${JOB_NAME} to complete..."
kubectl -n "${NAMESPACE}" wait --for=condition=complete --timeout=1h "job/${JOB_NAME}"

echo "Cleaning up export job..."
kubectl -n "${NAMESPACE}" delete job "${JOB_NAME}" --ignore-not-found

echo "Deleting PVCs ${PVC_DATA} and ${PVC_OUTPUTS}..."
kubectl -n "${NAMESPACE}" delete pvc "${PVC_DATA}" "${PVC_OUTPUTS}"

echo "Export and cleanup finished."
