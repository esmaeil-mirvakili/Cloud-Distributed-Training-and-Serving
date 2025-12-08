# AWS Serving Stack (EKS)

Serves a GGUF model on the same EKS cluster as training. Models are pulled from S3 (no PVCs). Prometheus + Grafana included.

## Prerequisites
- EKS cluster reachable by kubectl (namespace `llm-training`).
- IRSA on `smirvaki-trainer` with S3 read/write to your model prefix.
- Model available in S3 or downloadable from Hugging Face.
- Images accessible: CPU `docker.io/esmaeilmirvakili/llama-serving:latest`, GPU `docker.io/esmaeilmirvakili/llama-serving-gpu:latest` (build from `infrastructure/docker/Dockerfile.llama-python-gpu`).

## Config
- `serving/model.env`:
  - `S3_MODEL_PREFIX` – S3 prefix to store/fetch the model (e.g., `s3://llm-training-outputs/model`).
  - `S3_MODEL_FILE` – target filename in S3 (e.g., `model.gguf`).
  - `HF_MODEL_REPO`, `HF_MODEL_FILE` – Hugging Face source to upload to S3 (job uses optional `HF_TOKEN` secret `huggingface-token`/`token`).
  - `LLAMA_MODEL_PATH` – where the serving pod expects the file (defaults `/models/model.gguf`).
- Overlays:
  - CPU: `infrastructure/k8s/aws/serving-cpu`
  - GPU: `infrastructure/k8s/aws/serving-gpu` (requests 1 GPU, tolerates GPU taint, sets `LLAMA_GPU_LAYERS=-1`, uses the GPU image).

## Deploy
1) Push/ensure model:
```bash
kubectl apply -k infrastructure/k8s/aws/serving          # creates config + download job + base deploy
kubectl -n llm-training wait --for=condition=complete job/smirvaki-llama-model-download
```
   - The job downloads from HF -> uploads to `S3_MODEL_PREFIX/S3_MODEL_FILE`.

2) Serving pod waits for the S3 object, downloads to `/models`, then starts API+nginx.
   - Switch overlays:
```bash
# CPU-only
kubectl apply -k infrastructure/k8s/aws/serving-cpu
# GPU
kubectl apply -k infrastructure/k8s/aws/serving-gpu
```

3) Verify & test:
```bash
kubectl -n llm-training get pods,svc smirvaki-llama-serving
kubectl -n llm-training port-forward deploy/smirvaki-llama-serving 8000:8000
```

## Observability
- Prometheus: `kubectl -n llm-training port-forward svc/smirvaki-prometheus 9090:9090`
- Grafana: `kubectl -n llm-training port-forward svc/smirvaki-serving-grafana 3000:3000`
  - Login: `admin / changeme`
  - Dashboard preloaded (`serving.json`).
- HPA uses custom metrics (inflight, p95 latency, GPU utilization) plus CPU; requires Prometheus Adapter for the custom metrics.

## Cleanup
```bash
kubectl delete -k infrastructure/k8s/aws/serving
kubectl -n llm-training delete job smirvaki-llama-model-download --ignore-not-found
```

## Build GPU image
```bash
docker build --platform linux/amd64 -t docker.io/esmaeilmirvakili/llama-serving-gpu:latest -f infrastructure/docker/Dockerfile.llama-python-gpu .
docker push docker.io/esmaeilmirvakili/llama-serving-gpu:latest
```

## Notes
- Ensure `S3_MODEL_PREFIX`/`S3_MODEL_FILE` match the uploaded model; the init container fails fast if missing.
- For private HF repos, create `huggingface-token` secret in `llm-training` with key `token`.***
