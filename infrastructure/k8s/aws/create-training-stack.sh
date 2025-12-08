#!/usr/bin/env bash
set -euo pipefail
export AWS_PAGER=""
export PAGER="${PAGER:-cat}"

CLUSTER_NAME="${CLUSTER_NAME:-llm-training-serving}"
REGION="${AWS_REGION:-us-east-1}"
EKS_CONFIG="${EKS_CONFIG:-$(dirname "$0")/eks-cluster.yaml}"
BUCKET_NAME="${S3_BUCKET_NAME:-llm-training-outputs}"
ROLE_NAME="${IRSA_ROLE_NAME:-llm-training-irsa}"
S3_POLICY_NAME="${S3_POLICY_NAME:-llm-training-s3}"
CONTEXT_NAME="${K8S_CONTEXT_NAME:-}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-smirvaki-trainer}"

info() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }

require() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing dependency: $1" >&2
    exit 1
  fi
}

ensure_cluster() {
  if eksctl get cluster --name "$CLUSTER_NAME" --region "$REGION" >/dev/null 2>&1; then
    info "EKS cluster ${CLUSTER_NAME} already exists in ${REGION}"
  else
    info "Creating EKS cluster ${CLUSTER_NAME} in ${REGION} using ${EKS_CONFIG}"
    eksctl create cluster -f "$EKS_CONFIG"
  fi
  info "Updating kubeconfig"
  aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$REGION" >/dev/null
  if [ -z "$CONTEXT_NAME" ]; then
    local account
    account=$(aws sts get-caller-identity --query Account --output text)
    CONTEXT_NAME="arn:aws:eks:${REGION}:${account}:cluster/${CLUSTER_NAME}"
  fi
  info "Setting kubectl context to ${CONTEXT_NAME}"
  kubectl config use-context "$CONTEXT_NAME" >/dev/null
}

ensure_nodegroup() {
  local name="$1" type="$2" desired="$3" min="$4" max="$5"
  if eksctl get nodegroup --cluster "$CLUSTER_NAME" --region "$REGION" --name "$name" >/dev/null 2>&1; then
    info "Scaling nodegroup ${name} to desired=${desired}, min=${min}, max=${max}"
    eksctl scale nodegroup --cluster "$CLUSTER_NAME" --region "$REGION" --name "$name" --nodes "$desired" --nodes-min "$min" --nodes-max "$max"
  else
    info "Creating nodegroup ${name} (${type})"
    eksctl create nodegroup --cluster "$CLUSTER_NAME" --region "$REGION" \
      --name "$name" --node-type "$type" \
      --nodes "$desired" --nodes-min "$min" --nodes-max "$max"
  fi
}

ensure_bucket() {
  if aws s3api head-bucket --bucket "$BUCKET_NAME" >/dev/null 2>&1; then
    info "S3 bucket ${BUCKET_NAME} already exists"
  else
    info "Creating S3 bucket ${BUCKET_NAME} in ${REGION}"
    if [ "$REGION" = "us-east-1" ]; then
      aws s3api create-bucket --bucket "$BUCKET_NAME"
    else
      aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION" \
        --create-bucket-configuration LocationConstraint="$REGION"
    fi
    aws s3api put-bucket-encryption --bucket "$BUCKET_NAME" --server-side-encryption-configuration '{
      "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
    }' || warn "Could not set default encryption on ${BUCKET_NAME}"
  fi
}

main() {
  require aws
  require eksctl
  require kubectl
  require helm

  info "Using cluster=${CLUSTER_NAME}, region=${REGION}, bucket=${BUCKET_NAME}"
  ensure_cluster
  ensure_namespace
  ensure_nodegroup base t3.small 3 2 32
  ensure_nodegroup gpu g5.xlarge 1 1 8
  ensure_bucket
  ensure_iam_role
  ensure_autoscaler_iam
  ensure_autoscaler_sa
  ensure_autoscaler
  ensure_serviceaccount
  info "Provisioning complete. Apply manifests with: kubectl apply -k infrastructure/k8s/aws/training_data && kubectl apply -k infrastructure/k8s/aws/training"
}

ensure_iam_role() {
  local account oidc issuer tmp_policy
  account=$(aws sts get-caller-identity --query Account --output text)
  issuer=$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" --query "cluster.identity.oidc.issuer" --output text)
  oidc="${issuer##*/}"

  info "Associating OIDC provider for cluster (id=${oidc})"
  eksctl utils associate-iam-oidc-provider --cluster "$CLUSTER_NAME" --region "$REGION" --approve >/dev/null

  cat > "$(dirname "$0")/trust.json" <<EOF
{
  "Version":"2012-10-17",
  "Statement":[{
    "Effect":"Allow",
    "Principal":{"Federated":"arn:aws:iam::${account}:oidc-provider/oidc.eks.${REGION}.amazonaws.com/id/${oidc}"},
    "Action":"sts:AssumeRoleWithWebIdentity",
    "Condition":{
      "StringEquals":{
        "oidc.eks.${REGION}.amazonaws.com/id/${oidc}:sub":"system:serviceaccount:llm-training:${SERVICE_ACCOUNT}",
        "oidc.eks.${REGION}.amazonaws.com/id/${oidc}:aud":"sts.amazonaws.com"
      }
    }
  }]
}
EOF

  tmp_policy=$(mktemp)
  cat > "$tmp_policy" <<EOF
{
  "Version":"2012-10-17",
  "Statement":[
    {"Effect":"Allow","Action":["s3:ListBucket"],"Resource":"arn:aws:s3:::${BUCKET_NAME}"},
    {"Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:DeleteObject"],"Resource":"arn:aws:s3:::${BUCKET_NAME}/*"}
  ]
}
EOF

  if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
    info "Updating assume-role policy for ${ROLE_NAME}"
    aws iam update-assume-role-policy --role-name "$ROLE_NAME" --policy-document "file://$(dirname "$0")/trust.json"
  else
    info "Creating IAM role ${ROLE_NAME}"
    aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document "file://$(dirname "$0")/trust.json"
  fi

  info "Attaching inline S3 policy ${S3_POLICY_NAME} to ${ROLE_NAME} for bucket ${BUCKET_NAME}"
  aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name "$S3_POLICY_NAME" --policy-document "file://${tmp_policy}"
  rm -f "$tmp_policy"
}

ensure_namespace() {
  if kubectl get namespace llm-training >/dev/null 2>&1; then
    info "Namespace llm-training already exists"
  else
    info "Creating namespace llm-training"
    kubectl create namespace llm-training
  fi
}

ensure_serviceaccount() {
  local account role_arn
  account=$(aws sts get-caller-identity --query Account --output text)
  role_arn="arn:aws:iam::${account}:role/${ROLE_NAME}"

  if kubectl -n llm-training get sa "${SERVICE_ACCOUNT}" >/dev/null 2>&1; then
    info "ServiceAccount ${SERVICE_ACCOUNT} exists; patching IRSA annotation"
    kubectl -n llm-training patch sa "${SERVICE_ACCOUNT}" -p "{\"metadata\":{\"annotations\":{\"eks.amazonaws.com/role-arn\":\"${role_arn}\"}}}"
  else
    info "Creating ServiceAccount ${SERVICE_ACCOUNT} with IRSA annotation"
    kubectl create serviceaccount "${SERVICE_ACCOUNT}" -n llm-training --dry-run=client -o yaml \
      | kubectl apply -f -
    kubectl -n llm-training annotate sa "${SERVICE_ACCOUNT}" "eks.amazonaws.com/role-arn=${role_arn}" --overwrite
  fi
}

ensure_autoscaler_iam() {
  local account issuer oidc role_name inline_policy
  account=$(aws sts get-caller-identity --query Account --output text)
  issuer=$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" --query "cluster.identity.oidc.issuer" --output text)
  oidc="${issuer##*/}"
  role_name="cluster-autoscaler-${CLUSTER_NAME}"

  cat > "$(dirname "$0")/autoscaler-trust.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Federated": "arn:aws:iam::${account}:oidc-provider/oidc.eks.${REGION}.amazonaws.com/id/${oidc}"
    },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "oidc.eks.${REGION}.amazonaws.com/id/${oidc}:sub": "system:serviceaccount:kube-system:cluster-autoscaler",
        "oidc.eks.${REGION}.amazonaws.com/id/${oidc}:aud": "sts.amazonaws.com"
      }
    }
  }]
}
EOF

  inline_policy=$(mktemp)
  cat > "$inline_policy" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "autoscaling:DescribeAutoScalingGroups",
        "autoscaling:DescribeAutoScalingInstances",
        "autoscaling:DescribeLaunchConfigurations",
        "autoscaling:DescribeTags",
        "autoscaling:SetDesiredCapacity",
        "autoscaling:TerminateInstanceInAutoScalingGroup",
        "autoscaling:UpdateAutoScalingGroup",
        "ec2:DescribeLaunchTemplateVersions"
      ],
      "Resource": "*"
    }
  ]
}
EOF

  if aws iam get-role --role-name "$role_name" >/dev/null 2>&1; then
    info "Updating assume-role policy for ${role_name}"
    aws iam update-assume-role-policy --role-name "$role_name" --policy-document "file://$(dirname "$0")/autoscaler-trust.json"
  else
    info "Creating IAM role ${role_name} for cluster-autoscaler"
    aws iam create-role --role-name "$role_name" --assume-role-policy-document "file://$(dirname "$0")/autoscaler-trust.json" >/dev/null
  fi

  info "Attaching inline autoscaler policy to ${role_name}"
  aws iam put-role-policy --role-name "$role_name" --policy-name "${role_name}-inline" --policy-document "file://${inline_policy}"
  rm -f "$inline_policy"
}

ensure_autoscaler() {
  info "Installing/Upgrading cluster-autoscaler via Helm"
  helm repo add autoscaler https://kubernetes.github.io/autoscaler >/dev/null
  helm repo update autoscaler >/dev/null
  helm upgrade --install cluster-autoscaler autoscaler/cluster-autoscaler \
    -n kube-system \
    --set autoDiscovery.clusterName="$CLUSTER_NAME" \
    --set awsRegion="$REGION" \
    --set rbac.serviceAccount.create=false \
    --set rbac.serviceAccount.name=cluster-autoscaler \
    --set extraArgs.skip-nodes-with-local-storage=false \
    --set extraArgs.scale-down-unneeded-time=5m \
    --set extraArgs.scan-interval=10s >/dev/null
}

ensure_autoscaler_sa() {
  local account role_name role_arn
  account=$(aws sts get-caller-identity --query Account --output text)
  role_name="cluster-autoscaler-${CLUSTER_NAME}"
  role_arn="arn:aws:iam::${account}:role/${role_name}"

  if kubectl -n kube-system get sa cluster-autoscaler >/dev/null 2>&1; then
    info "ServiceAccount kube-system/cluster-autoscaler exists; patching IRSA annotation"
    kubectl -n kube-system patch sa cluster-autoscaler -p "{\"metadata\":{\"annotations\":{\"eks.amazonaws.com/role-arn\":\"${role_arn}\"}}}"
  else
    info "Creating ServiceAccount kube-system/cluster-autoscaler with IRSA annotation"
    kubectl create serviceaccount cluster-autoscaler -n kube-system --dry-run=client -o yaml \
      | kubectl apply -f -
    kubectl -n kube-system annotate sa cluster-autoscaler "eks.amazonaws.com/role-arn=${role_arn}" --overwrite
  fi
}

main "$@"
