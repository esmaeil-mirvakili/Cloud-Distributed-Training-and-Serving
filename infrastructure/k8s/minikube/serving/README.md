## Minikube serving deployment (llama.cpp via Python bindings)

Minikube-friendly copy of the Nautilus serving stack with lighter resources, default `standard` hostPath storage, CPU-only HPA, and a local ingress host (`llama.minikube.local`). Components:
- `pvc-model.yaml` — RWO PVC for the model at `/models` (name: `smirvaki-llama-model-pvc`, `standard` StorageClass).
- `job-model-download.yaml` — pulls the GGUF into the PVC with `llama-model-downloader:latest` (optional `huggingface-token` secret).
- `deployment.yaml` — `llama-serving:latest` + in-pod NGINX proxy; defaults to server mode behind port 8000; mounts the model PVC.
- `nginx-config.yaml`, `service.yaml`, `ingress.yaml` — NGINX config, ClusterIP, and ingress (`llama.minikube.local`).
- `prometheus-deployment.yaml`, `grafana-deployment.yaml` — namespace-scoped monitoring scraping `/metrics` on the API container (2112).
- `hpa.yaml` — CPU-based autoscaler (min 1, max 3; needs metrics-server).
- `model.env` — config for model path/repo, backend selection, threads/context, etc. (ConfigMap name `smirvaki-llama-model-config`).

### Prerequisites
- Minikube running with the ingress and metrics-server addons:
  ```
  minikube addons enable ingress
  minikube addons enable metrics-server
  ```
- `kubectl` pointed at the minikube context.
- (Optional) Add `/etc/hosts` entry: `127.0.0.1 llama.minikube.local` if using ingress + `minikube tunnel`.

### Build/push images
- Use the public images referenced in the manifests, **or** build locally into the minikube daemon:
  ```
  eval "$(minikube docker-env)"
  docker build -t docker.io/esmaeilmirvakili/llama-serving:latest -f infrastructure/docker/Dockerfile.llama-python .
  docker build -t docker.io/esmaeilmirvakili/llama-model-downloader:latest -f infrastructure/docker/Dockerfile.model-downloader .
  ```
- To change model defaults, edit `infrastructure/k8s/minikube/serving/model.env` before applying.

### Deploy
```
kubectl apply -k infrastructure/k8s/minikube/serving
kubectl -n llm-serving wait --for=condition=complete job/smirvaki-llama-model-download
kubectl -n llm-serving get pods -l app=smirvaki-llama-serving
```
- Optional HF token secret (before applying):  
  `kubectl -n llm-serving create secret generic huggingface-token --from-literal=token=<hf_token>`

### Access
- Port-forward:
  ```
  kubectl -n llm-serving port-forward svc/smirvaki-llama-serving 8000:80
  curl -X POST http://localhost:8000/completion \
    -H "Content-Type: application/json" \
    -d '{"prompt":"hi","n_predict":32}'
  ```
- Ingress: run `minikube tunnel` (or use `minikube ip` with a hosts entry) and hit `http://llama.minikube.local/`.
- Metrics:
  ```
  kubectl -n llm-serving port-forward svc/smirvaki-prometheus 9090:9090
  kubectl -n llm-serving port-forward svc/smirvaki-serving-grafana 3000:3000
  ```

### Notes
- PVC is RWO; keep replicas low unless you switch to an RWX storage class.
- Serving pod requests 1 CPU / 2 Gi and can burst to 2 CPU / 4 Gi; adjust in `deployment.yaml` if you need tighter limits.
- HPA is CPU-only to avoid needing Prometheus Adapter; tune limits in `hpa.yaml`.
- The default model is SmolLM2 135M Instruct Q4_K_M (very small download). Override repo/path in `model.env` if you need higher quality.
