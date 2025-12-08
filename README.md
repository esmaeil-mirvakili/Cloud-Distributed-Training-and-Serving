# Cloud-Distributed-Training-and-Serving

Quick start to run the serving stack on Minikube and chat through the local web UI.

## Prerequisites
- Minikube with `ingress` and `metrics-server` addons:  
  `minikube addons enable ingress && minikube addons enable metrics-server`
- `kubectl` pointed at the Minikube context.
- Optional: Hugging Face token if the model repo is private.

## 1) Start Minikube
```bash
minikube start --cpus 6 --memory 7g
```

## 2) Deploy the serving stack (CPU by default), it can take a few mins to download the model and run the llama.cpp
```bash
kubectl apply -k infrastructure/k8s/minikube/serving
kubectl -n llm-serving get pods -l app=smirvaki-llama-serving
```

## 3) Port-forward the service
```bash
kubectl -n llm-serving port-forward svc/smirvaki-llama-serving 8000:80
```

## 4) Run the local web UI and chat
```bash
python scripts/web_chat_ui.py --llama-server http://127.0.0.1:8000 --port 8080
```
Then open `http://127.0.0.1:8080` in your browser. Each send issues a single-turn call to `/v1/chat/completions` on the serving stack.

## Teardown
```bash
kubectl delete -k infrastructure/k8s/minikube/serving
minikube delete  # optional
```
