# Training Stack Implementation Plan

This document describes the phased implementation plan for the training stack, aligned with the repository structure:

- `infrastructure/`
  - `docker/`
  - `k8s/`
  - `terraform/`
- `scripts/`
- `src/`
  - `training/`
- `pyproject.toml`

---

## Phase 1 – Python Training Code (`src/training`, `pyproject.toml`, `scripts/`)

### 1.1. Dependencies & Project Wiring

- In `pyproject.toml`, add (at minimum):
  - `torch`, `transformers`, `datasets`, `peft`, `deepspeed`
  - `prometheus-client`, `psutil`, `pynvml` (or `nvidia-ml-py` in the future)
  - `pyyaml`, `hydra-core`
- Define the main training entrypoint as a console script:
  - `train-llm = training.cli:main`

### 1.2. Dataset-Agnostic Data Layer

- Implement a formatter abstraction:

  - `ExampleFormatter` (abstract base class) with:
    - `format_example(example: Dict) -> Tuple[str, str]` returning `(prompt, target)`.
  - `InstructionFormatter` implementing:
    - Configurable `instruction_field`, `input_field`, `target_field`, and optional `template`.

- Implement dataset-agnostic utilities in `src/training/data.py`:
  - `load_hf_dataset(dataset_name, split, subset=None)`
  - `tokenize_dataset(ds, tokenizer, formatter, max_length)`
  - `shard_dataset(ds, num_shards, shard_id)`
  - `get_dataloader(tokenized, batch_size, shuffle, num_workers)`
  - `preprocess_and_save_shards(dataset_name, formatter, model_name, output_dir, split, subset, max_length, num_shards)`
  - `load_sharded_dataset(data_dir, shard_id)` (using `torch.load(..., weights_only=False)`)

### 1.3. Model & LoRA / PEFT

- Implement `src/training/modeling.py`:
  - `load_base_model(model_name, device_map="auto")`
  - `load_tokenizer(model_name)` (set `pad_token` to `eos_token` if missing)
  - `apply_lora(model, r, lora_alpha, lora_dropout, target_modules)` using `peft.LoraConfig`.

### 1.4. Metrics & Observability

- Implement `src/training/metrics.py`:
  - Prometheus metrics:
    - `training_loss` (Gauge)
    - `step_time_seconds` (Summary)
    - `cpu_utilization_percent`, `process_rss_memory_bytes`
    - Optional GPU metrics (`gpu_utilization_percent`, `gpu_memory_used_bytes`) via `pynvml` guarded by runtime checks.
  - `start_metrics_server(port)` that:
    - `start_http_server(port)`
    - Spawns a background thread to collect CPU/RSS (and GPU if available) periodically.
  - Ensure NVML failures on non-GPU machines (e.g. macOS) are handled gracefully and only disable GPU metrics, not the whole training.

### 1.5. Training Loop (Baseline + DeepSpeed)

- Implement `src/training/train.py` with:

  - `TrainConfig` dataclass:
    - `model_name`, `data_dir`, `shard_id`
    - `batch_size`, `num_epochs`, `lr`, `weight_decay`, `warmup_ratio`
    - `output_dir`, `use_lora`
    - `deepspeed_config` (dict or `None`)
    - `metrics_port`, `max_steps`

  - `run_training(cfg: TrainConfig)`:
    - Start metrics server.
    - Load sharded dataset and create dataloader.
    - Load base model; optionally wrap with LoRA.
    - Build optimizer and scheduler.
    - If `cfg.deepspeed_config` is provided:
      - Use `deepspeed.initialize(model=model, model_parameters=optimizer_grouped_parameters, config=cfg.deepspeed_config)`.
      - Use the returned engine for `backward` and `step`.
    - Else:
      - Standard single-process PyTorch training loop.
    - Log per-step:
      - Loss (to Prometheus).
      - Step time (to Prometheus).
    - Respect `max_steps` for quick tests.
    - Save:
      - LoRA adapter or full fine-tuned model under `output_dir`.
      - `summary.json` with config, global_step, total_time_seconds.

### 1.6. Hydra-Based CLI & Config

- Replace manual argparse CLI with Hydra:

  - `conf/config.yaml` (or `src/training/conf/config.yaml`) defines:
    - `mode: preprocess | train`
    - `dataset` (name, split, subset)
    - `formatter`:
      - `_target_: training.formatting.InstructionFormatter`
      - `instruction_field`, `input_field`, `target_field`, `template`
    - `model` (name, max_length)
    - `preprocess` (output_dir, num_shards)
    - `train` (data_dir, shard_id, batch_size, num_epochs, lr, weight_decay, warmup_ratio, output_dir, use_lora, metrics_port, max_steps, deepspeed_config_path)

- Implement `src/training/cli.py`:

  - Use `@hydra.main(config_path="conf", config_name="config")`.
  - Instantiate formatter via `formatter: ExampleFormatter = instantiate(cfg.formatter)`.
  - If `cfg.mode == "preprocess"`:
    - Call `preprocess_and_save_shards(...)`.
  - If `cfg.mode == "train"`:
    - Load DeepSpeed config from `cfg.train.deepspeed_config_path` (if not `null`).
    - Build `TrainConfig` and call `run_training(...)`.

### 1.7. Helper Scripts (`scripts/`)

- Optional scripts for convenience (Hydra override style):

  - `scripts/preprocess_dolly.sh`:

    ```bash
    train-llm mode=preprocess
    ```

  - `scripts/run_local_baseline.sh`:

    ```bash
    train-llm mode=train train.max_steps=100
    ```

- These wrap the Hydra config and allow quick local tests.

---

## Phase 2 – Containerization (`infrastructure/docker`)

### 2.1. Training Docker Image

- Create `infrastructure/docker/Dockerfile.training`:

  - Base image: a CUDA-enabled PyTorch image (e.g. `pytorch/pytorch:2.x-cuda11x-cudnn8-runtime`) for GPU environments.
  - Install OS dependencies: `git`, `curl`, `python3`, `pip`, `psutil`, `pynvml` (or `nvidia-ml-py`).
  - Copy project files:
    - `pyproject.toml`
    - `src/`
    - `scripts/`
    - `conf/`
  - Install Python package with `pip install .`.
  - Expose metrics port `8000`.
  - Set `ENTRYPOINT`:
    - Either directly to `train-llm`, or to a small `/app/scripts/entrypoint.sh` that invokes `train-llm` with arguments from env vars.

### 2.2. Local Container Sanity Test

- Build image:

  ```bash
  docker build -t training-stack:local -f infrastructure/docker/Dockerfile.training .
  ```

- Run locally (GPU host):

  ```bash
  docker run --gpus all     -v /local/data:/data     -v /local/outputs:/outputs     training-stack:local     train-llm mode=train train.data_dir=/data train.output_dir=/outputs
  ```

- Verify:
  - Training starts and finishes on a small number of steps.
  - `/metrics` endpoint works inside the container.

---

## Phase 3 – Nautilus Deployment (Kubernetes, `infrastructure/k8s/nautilus`)

### 3.1. Kubernetes Manifests

Create `infrastructure/k8s/nautilus` with:

- Namespace:
  - Use the Nautilus-provisioned namespace (e.g. `cse239fall2025`). No manifest is stored here because the namespace is managed centrally.

- `deepspeed_zero2.json`:
  - Source-of-truth DeepSpeed config loaded into the `smirvaki-trainer-configs` ConfigMap via:
    ```bash
    kubectl -n cse239fall2025 create configmap smirvaki-trainer-configs \
      --from-file=deepspeed_zero2.json=infrastructure/k8s/nautilus/deepspeed_zero2.json \
      --dry-run=client -o yaml | kubectl apply -f -
    ```

- `secret-registry.yaml`:
  - Credentials for pulling the training Docker image from the registry.

- `pvc-data.yaml`:
  - PersistentVolumeClaim(s) for:
    - `/data` – preprocessed shards.
    - `/outputs` – model checkpoints and run summaries.

- `job-training.yaml`:
  - Defines a `Job`:
    - Uses training image.
    - Requests GPU resources (`nvidia.com/gpu`).
    - Mounts PVCs (`/data`, `/outputs`) and ConfigMaps.
    - Sets env vars for:
      - `HYDRA_FULL_ERROR`, `CLOUD=nautilus`, `DATA_DIR`, `OUTPUT_DIR`, `METRICS_PORT`, etc.
    - Runs `train-llm mode=train ...` with env-based overrides.

### 3.2. Prometheus Integration on Nautilus

- Add Prometheus scrape annotations to the pod template in `job-training.yaml`:

  ```yaml
  metadata:
    annotations:
      prometheus.io/scrape: "true"
      prometheus.io/port: "8000"
      prometheus.io/path: "/metrics"
  ```

- Ensure the Nautilus cluster’s Prometheus is configured to scrape annotated pods.

- Build Grafana dashboards for:
  - `training_loss`.
  - `step_time_seconds`.
  - Resource usage (CPU / memory / GPU via Prometheus + kube-state-metrics / node exporter).

### 3.3. Nautilus Workflow

- Preprocess Dolly (or any dataset) on Nautilus:
  - Run a temporary `Job` that calls `train-llm mode=preprocess` and writes shards to `/data`.

- Launch training Jobs:
  - Full SFT with DeepSpeed (multi-GPU).
  - LoRA with DeepSpeed.
  - Optional single-GPU baseline.

- Validate:
  - Models and summaries written to `/outputs`.
  - Metrics visible in Prometheus/Grafana.

---

## Phase 4 – Cloud (GCP) via Kubernetes & Terraform (`infrastructure/terraform`, `infrastructure/k8s/gcp`)

### 4.1. Terraform: GKE, GPU Node Pool, Storage

In `infrastructure/terraform/gcp`:

- Define:
  - `google_container_cluster` (GKE cluster).
  - `google_container_node_pool` for GPU nodes (e.g. `n1-standard-8` with T4/L4 GPUs).
  - Networking resources (`google_compute_network` and subnetwork) if needed.

- Storage:
  - `google_storage_bucket` for:
    - Preprocessed dataset shards.
    - Model checkpoints and summaries (if not using PVs).

- IAM:
  - Service account for training workloads.
  - Permissions for accessing the bucket.
  - Optionally use Workload Identity bound to a Kubernetes service account.

### 4.2. Monitoring Stack in GCP

- Either:
  - Deploy a Prometheus/Grafana stack using `helm_release` (e.g. kube-prometheus-stack), or
  - Use an existing monitoring solution in the cluster.

- Ensure Prometheus is configured to scrape pods with the standard scrape annotations.

### 4.3. Kubernetes Manifests for GCP

Create `infrastructure/k8s/gcp`:

- Similar manifests to Nautilus, with environment-specific differences:

  - `namespace.yaml` (e.g. `llm-training`).
  - PVC definitions that use GCP-specific `StorageClass`, or use GCS via CSI driver.
  - `job-training.yaml`:
    - Uses the same training image.
    - Sets `CLOUD=gcp` environment variable.
    - Same command pattern: `train-llm mode=train ...`.

- Data handling options:
  - Option A:
    - Preprocess Dolly in-cluster using a K8s Job, write shards to a PVC.
  - Option B:
    - Preprocess locally or elsewhere, upload shards to GCS, and have training jobs download from GCS to local storage on startup.

### 4.4. GCP Workflow

- `terraform apply` in `infrastructure/terraform/gcp`:
  - Brings up cluster, GPU nodepool, bucket, and optionally monitoring.

- Configure `kubectl` to target the GKE cluster.

- Apply manifests:

  ```bash
  kubectl apply -f infrastructure/k8s/gcp/
  ```

- Run:
  - Preprocessing Job (if not using precomputed shards).
  - Training Jobs for:
    - Full SFT.
    - LoRA.
    - Single-GPU baseline.

- Validate:
  - Checkpoints and summaries are written to PVC or GCS.
  - Metrics are visible via Prometheus/Grafana.

---

## Phase 5 – Cost & Performance Comparison (Nautilus vs GCP)

### 5.1. Metrics Collection Per Run

For each run (per environment, per mode):

- `summary.json` contents:
  - `config` (including batch size, num_epochs, model_name, etc.).
  - `global_step`.
  - `total_time_seconds`.

- Additional telemetry from Prometheus:
  - CPU, memory, GPU utilization over time.
  - Number of GPUs and nodes used.

- Logs (from pod logs or centralized logging) for troubleshooting.

### 5.2. Cost Calculation

Implement `scripts/calc_costs.py`:

- Inputs:
  - Run summaries (paths to `summary.json`).
  - Node type and pricing info (hardcoded or config-driven).

- For GCP:
  - Use on-demand node-hour pricing per machine type + GPU type.
  - Cost per run:
    - `cost = (node_hour_price * node_hours_used)`.

- For Nautilus:
  - Compute “virtual cost” by mapping Nautilus node specs to equivalent GCP machine type and applying the same pricing.

- Output:
  - CSV or Markdown table summarizing:
    - Environment (Nautilus vs GCP).
    - Run type (SFT vs LoRA, single-GPU vs DeepSpeed).
    - Total time, throughput (tokens / second or steps / second), estimated cost.

### 5.3. Summary & Reporting

- Combine:
  - Performance metrics (throughput, time-to-train).
  - Utilization metrics (GPU/CPU/memory).
  - Monetary and virtual costs.

- Produce a concise report (Markdown or notebook) that shows:

  - Where Nautilus is cost-effective vs GCP and vice versa.
  - The benefit of DeepSpeed multi-GPU vs single-GPU.
  - The benefit of LoRA vs full SFT in terms of:
    - Total time.
    - Resource usage.
    - Cost.
