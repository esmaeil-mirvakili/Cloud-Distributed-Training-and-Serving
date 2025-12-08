# AWS Training Stack (EKS)

Kustomize manifests to run the training stack on EKS with GPU nodes and **S3-backed data/outputs** (no EFS required). Prometheus/Grafana are bundled and scrape the trainer `/metrics` endpoint.

## Prerequisites
- EKS cluster with the NVIDIA device plugin installed; GPU node group with `nvidia.com/gpu` capacity.
- S3 bucket for shards and outputs (defaults in env files: `s3://llm-training-outputs/{data,outputs}`).
- IRSA role with S3 access; set `eks.amazonaws.com/role-arn` in `serviceaccount.yaml` and set `S3_*` + `AWS_REGION` in the env files.
- Training images pushed to a pullable registry (ECR recommended): `training_stack` and `training_preprocess`.

## Quickstart
1) Set values:
   - Set your IRSA role ARN in `serviceaccount.yaml`.
   - Edit `training_data/dataset.env` for dataset/model and `S3_DATA_PREFIX`.
   - Edit `training/training.env` (and overlay envs) for model/training knobs, `AWS_REGION`, `S3_DATA_PREFIX`, and `S3_OUTPUT_PREFIX`.
   - Point images in `training/statefulset-training.yaml` and `training_data/job-preprocess.yaml` to your registry if needed.

```bash
aws s3 mb s3://llm-training-outputs

kubectl apply -f infrastructure/k8s/aws/serviceaccount.yaml

```


2) Prepare data (namespace + SA + preprocess job; writes shards to S3):
```bash
kubectl apply -k infrastructure/k8s/aws/training_data
kubectl -n llm-training wait --for=condition=complete job/smirvaki-preprocess-shards
kubectl -n llm-training logs job/smirvaki-preprocess-shards -f
```

3) Launch training (pods download shards from S3 to emptyDir, then upload outputs back to S3):
```bash
# 1 GPU
kubectl apply -k infrastructure/k8s/aws/training
# or scale out
# kubectl apply -k infrastructure/k8s/aws/training_2_gpu
# kubectl apply -k infrastructure/k8s/aws/training_4_gpu
# kubectl apply -k infrastructure/k8s/aws/training_8_gpu
```

4) Metrics dashboards:
```bash
kubectl -n llm-training port-forward svc/smirvaki-prometheus 9090:9090
kubectl -n llm-training port-forward svc/smirvaki-grafana 3000:3000
# Login: admin / changeme (see grafana-deployment.yaml)
```

5) Cleanup:
```bash
kubectl delete -k infrastructure/k8s/aws/training
kubectl delete -k infrastructure/k8s/aws/training_data
```

## Serving (same cluster)
- Manifests live at `infrastructure/k8s/aws/serving`.
- Update `serving/model.env` with your model path and `S3_MODEL_PREFIX` (defaults to `s3://llm-training-outputs/model`).
- Model download pulls directly from S3 in both the download Job and the serving initContainer; no PVC required (`emptyDir` + S3 sync).
- The model download Job and serving Deployment reuse the `smirvaki-trainer` service account (IRSA) for S3 access.
- Overlays:
  - CPU: `infrastructure/k8s/aws/serving-cpu` (avoid GPU nodes)
  - GPU: `infrastructure/k8s/aws/serving-gpu` (schedule on GPU nodes, request 1 GPU, sets `LLAMA_GPU_LAYERS=-1`, uses GPU image `docker.io/esmaeilmirvakili/llama-serving-gpu:latest`)

Apply:
```bash
kubectl apply -k infrastructure/k8s/aws/serving
kubectl -n llm-training wait --for=condition=complete job/smirvaki-llama-model-download
kubectl -n llm-training get deploy,svc smirvaki-llama-serving
# port-forward to test
kubectl -n llm-training port-forward deploy/smirvaki-llama-serving 8000:8000

# Or use overlays:
# CPU-only
# kubectl apply -k infrastructure/k8s/aws/serving-cpu
# GPU
# kubectl apply -k infrastructure/k8s/aws/serving-gpu
```

## Notes
- The preprocess job installs `awscli` in-container and syncs shards to `S3_DATA_PREFIX`; training pods sync that prefix to `/data` before running and push `/outputs` to `S3_OUTPUT_PREFIX/rank<N>` after completion.
- Ensure the IRSA role attached to `smirvaki-trainer` allows `s3:GetObject/PutObject` on your prefixes.
- Prometheus targets 8 ranks by default; extend the list in `prometheus-deployment.yaml` if you scale higher.
