## Nautilus serving deployment (llama.cpp via Python bindings)

This overlay runs llama.cpp through the Python bindings (FastAPI wrapper) for the Nautilus cluster. It includes:

- `pvc-model.yaml` — RWX PVC for model weights mounted at `/models` (name: `smirvaki-llama-model-pvc`).
- `job-model-download.yaml` — one-shot job using `llama-model-downloader:nautilus` to pull the GGUF into the PVC (job name: `smirvaki-llama-model-download`). Expects optional `huggingface-token` secret with key `token` for gated models.
- `deployment.yaml` — three replicas of `llama-serving:latest`, mounting the model PVC (deployment name: `smirvaki-llama-serving`) and running the FastAPI wrapper (python bindings) behind an in-pod NGINX proxy.
- `nginx-config.yaml` — NGINX config that load-balances to the local FastAPI process on 8001 and serves traffic on 8000.
- `service.yaml` — ClusterIP for the serving pods (service name: `smirvaki-llama-serving`), exposing HTTP (80->8000) via NGINX and Prometheus metrics on 2112 (served by the Python API at `/metrics`).
- `ingress.yaml` — routes `/` to the service via ingress-nginx (host `llama.nautilus.local`, adjust for your DNS/TLS; ingress name: `smirvaki-llama-serving`).
- `prometheus-deployment.yaml` — namespace-scoped Prometheus scraping itself and the Python wrapper metrics on port 2112.
- `hpa.yaml` — HorizontalPodAutoscaler for `smirvaki-llama-serving` (min 2, max 6 replicas, CPU target 70%).
- `prometheus-adapter-config.yaml` — rules for Prometheus Adapter to expose custom metrics (e.g., `llama_request_latency_p95_seconds`) for HPA.

Usage:
1. Build and push images to Docker Hub (namespace: `esmaeilmirvakili`, targeting amd64). The serving image now uses Python bindings:
   ```
   docker buildx build --platform linux/amd64 -t docker.io/esmaeilmirvakili/llama-serving:latest -f infrastructure/docker/Dockerfile.llama-python . --push
   docker buildx build --platform linux/amd64 -t docker.io/esmaeilmirvakili/llama-model-downloader:latest -f infrastructure/docker/Dockerfile.model-downloader . --push
   ```
2. (Optional) Create a Hugging Face token secret in the namespace:
   ```
   kubectl -n cse239fall2025 create secret generic huggingface-token --from-literal=token=<hf_token>
   ```
3. Apply manifests via kustomize (uses `kustomization.yaml` + `model.env` to generate the ConfigMap; name hashes are disabled so the ConfigMap is `smirvaki-llama-model-config`):
   ```
   kubectl apply -k infrastructure/k8s/nautilus/serving
   ```
   - `model.env` includes `LLAMA_GPU_LAYERS` (default 0). Set it >0 to offload layers when you deploy the GPU-enabled image/build; leave 0 for CPU-only.
   - Backend selection:
     - `LLAMA_BACKEND=python` (default) runs llama-cpp-python bindings in-process (model path required).
     - `LLAMA_BACKEND=server` proxies to a standalone llama-server at `LLAMA_SERVER_URL` (default `http://127.0.0.1:8080`); adjust `LLAMA_SERVER_TIMEOUT` as needed.
   - HPA is included by default (CPU utilization target). Requires metrics-server in the cluster.
   - For custom metrics (queue length, p95 latency) HPA requires Prometheus Adapter. Apply `prometheus-adapter-config.yaml` to your adapter deployment or equivalent configuration.
4. Wait for the download job to complete, then the deployment pods should start and mount the model.
5. Test via ingress:
   ```
   kubectl -n cse239fall2025 port-forward svc/smirvaki-llama-serving 8000:80

   curl -X POST http://localhost:8000/completion \
     -H "Content-Type: application/json" \
     -d '{"prompt":"hi","n_predict":32}'
   ```
   For the OpenAI-compatible chat endpoint (if enabled):
   ```
   curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "local-llama",
       "messages": [{"role": "user", "content": "Say hello"}],
       "max_tokens": 64,
       "temperature": 0.7,
       "stream": false
     }'
   ```

Metrics:
- The Python wrapper exports Prometheus metrics at `/metrics` on `svc/smirvaki-llama-serving:2112` (service forwards to the wrapper on port 8000).
- Prometheus is included here; port-forward to inspect targets/metrics:
  ```
  kubectl -n cse239fall2025 port-forward svc/smirvaki-prometheus 9090:9090
  open http://localhost:9090/targets
  ```
Adjust storage class (`pvc-model.yaml`), ingress host/TLS (`ingress.yaml`), resources, and model args as needed.
