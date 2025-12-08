#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-llm-training-serving}"
REGION="${AWS_REGION:-us-east-1}"
BUCKET_NAME="${S3_BUCKET_NAME:-llm-training-outputs}"
ROLE_NAME="${IRSA_ROLE_NAME:-llm-training-irsa}"
S3_POLICY_NAME="${S3_POLICY_NAME:-llm-training-s3}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-smirvaki-trainer}"
CONFIRM=yes # do not change

info() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }

require() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing dependency: $1" >&2
    exit 1
  fi
}

main() {
  if [ "${CONFIRM:-}" != "yes" ]; then
    echo "Set CONFIRM=yes to proceed with deleting cluster, bucket, and IAM role." >&2
    exit 1
  fi

  require aws
  require eksctl
  require kubectl
  require helm

  info "Deleting S3 bucket ${BUCKET_NAME} (skip with KEEP_BUCKET=yes)"
  if [ "${KEEP_BUCKET:-}" != "yes" ]; then
    if aws s3api head-bucket --bucket "$BUCKET_NAME" >/dev/null 2>&1; then
      aws s3 rb "s3://${BUCKET_NAME}" --force || warn "Failed to delete bucket ${BUCKET_NAME}"
    else
      warn "Bucket ${BUCKET_NAME} not found; skipping"
    fi
  else
    info "Keeping bucket ${BUCKET_NAME}"
  fi

  if [ "${KEEP_IAM:-}" != "yes" ]; then
    if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
      info "Detaching inline policy ${S3_POLICY_NAME} and deleting IAM role ${ROLE_NAME}"
      aws iam delete-role-policy --role-name "$ROLE_NAME" --policy-name "$S3_POLICY_NAME" >/dev/null 2>&1 || warn "Failed to delete inline policy ${S3_POLICY_NAME}"
      aws iam delete-role --role-name "$ROLE_NAME" || warn "Failed to delete role ${ROLE_NAME}"
    else
      warn "IAM role ${ROLE_NAME} not found; skipping"
    fi
  else
    info "Keeping IAM role ${ROLE_NAME}"
  fi

  info "Deleting namespace llm-training (if present)"
  kubectl delete namespace llm-training --ignore-not-found

  info "Uninstalling cluster-autoscaler (if present)"
  helm uninstall cluster-autoscaler -n kube-system >/dev/null 2>&1 || true
  eksctl delete iamserviceaccount --cluster "$CLUSTER_NAME" --region "$REGION" --name cluster-autoscaler --namespace kube-system --wait >/dev/null 2>&1 || true
  if [ "${KEEP_IAM:-}" != "yes" ]; then
    ca_role="cluster-autoscaler-${CLUSTER_NAME}"
    info "Deleting autoscaler IAM role ${ca_role}"
    aws iam delete-role-policy --role-name "${ca_role}" --policy-name "${ca_role}-inline" >/dev/null 2>&1 || true
    aws iam delete-role --role-name "${ca_role}" >/dev/null 2>&1 || true
  fi

  info "Deleting EKS cluster ${CLUSTER_NAME} in ${REGION}"
  eksctl delete cluster --name "$CLUSTER_NAME" --region "$REGION"
  info "Cleanup complete."
}

main "$@"
