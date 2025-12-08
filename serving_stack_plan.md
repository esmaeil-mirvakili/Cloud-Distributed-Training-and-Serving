Fair question. If you want llama.cpp “as is” with NGINX in front, here’s a revised plan that does exactly that and still satisfies the requirements (autoscaling, metrics, Nautilus, GCP, Terraform).

I’ll assume:

* llama.cpp runs in **server mode** and exposes its own HTTP API.
* **NGINX** is the HTTP load balancer:

  * Locally: container or docker-compose.
  * In Kubernetes: **ingress-nginx** (or equivalent).
* `src/serving` is used for **supporting scripts, exporters, k6 configs, cost scripts**, not an app server.

Repo layout you gave:

* `infrastructure/`

  * `docker/`
  * `k8s/`
  * `terraform/`
* `scripts/`
* `src/serving/`
* `pyproject.toml`

---

## Phase 0 · Decide on the serving contract & metrics plan

**Goal:** Fix the architecture so you don’t keep changing it mid-implementation.

1. **Use llama.cpp’s own HTTP API** as the serving endpoint.

   * Verify:

     * It can be run as an HTTP server in a container.
     * You can configure model path, context length, GPUs, etc via flags/env.
   * Decide how you’ll call it:

     * Either use its OpenAI-compatible mode (if supported in your version).
     * Or accept its native JSON API and adapt k6 scripts to that.

2. **API contract for clients & k6:**

   * Define this in `src/serving/api_contract.md` (or similar):

     * Request shape (prompt, parameters).
     * Response shape (text, tokens).
   * This is just documentation; the server is llama.cpp, not Python.

3. **Metrics strategy without a Python app:**

   * Cluster-level & LB metrics:

     * Use **ingress-nginx** Prometheus metrics for:

       * Requests per second.
       * Latency histograms.
       * In-flight requests / open connections per upstream pod.
   * Resource metrics:

     * Use **kube-state-metrics + node-exporter + DCGM** (if you care about GPU).
   * Model-specific metrics (tokens/sec, etc):

     * Option A: If llama.cpp exposes any metrics endpoint or logs token counts, write a **sidecar exporter** in `src/serving/exporter.py` that:

       * Scrapes stats or tails logs.
       * Exposes Prometheus metrics on `/metrics` (sidecar only, not in request path).
     * Option B: If that’s too much, derive tokens/sec from workload + average tokens per prompt (rough, but may be “good enough” for this project).

4. **HPA signals decision:**

   * Primary:

     * **NGINX ingress metrics**:

       * Per-upstream RPS.
       * P95 latency.
   * Secondary:

     * CPU utilization from Kubernetes resource metrics.
     * GPU utilization (via DCGM exporter) if using GPUs.
   * HPA will read:

     * CPU resource metrics directly.
     * RPS / latency via Prometheus-adapter from NGINX metrics (and exporter if you implement it).

Once you commit to this, stop thinking about a Python API server. It’s now **llama.cpp + NGINX + exporters.**

---

## Phase 1 · Local single-instance llama.cpp container

**Goal:** Run one llama.cpp HTTP server in Docker on your laptop.

1. **Base Dockerfile for llama.cpp server**
   `infrastructure/docker/Dockerfile.llamacpp`:

   * Stage 1: build llama.cpp.
   * Stage 2: minimal runtime image:

     * Copy the `server` binary.
     * Create a non-root user.
   * Entrypoint runs:

     * `./server` with flags for:

       * Model path.
       * Port (e.g. 8080).
       * Any needed threading / GPU options.

2. **Local run script**
   `scripts/run_llamacpp_local.sh`:

   * Mount a model directory:

     * `-v /local/models:/models`
   * Set env vars like `MODEL_PATH=/models/llama-7b` and map to flags.
   * Run `docker run -p 8080:8080 llama:local`.

3. **Smoke test the raw API:**

   * Use `curl` or a tiny Python script in `src/serving/test_client.py` to:

     * Hit a health-like endpoint if available or send a small prompt.
     * Validate response format.

You now have one container that is your **unit of serving**, no Python app.

---

## Phase 2 · Local NGINX + multiple llama.cpp containers

**Goal:** Show that NGINX can load balance across multiple llama.cpp instances.

1. **docker-compose for local LB**
   `infrastructure/docker/docker-compose.llama-local.yml`:

   * Services:

     * `llama1`, `llama2`, `llama3`:

       * All use the same image, different container names.
       * Same model volume or separate volumes.
       * Expose ports 8081, 8082, 8083 internally.
     * `nginx`:

       * Uses `nginx:stable`.
       * Mounts a config from `infrastructure/docker/nginx.local.conf`.
       * Exposes port 8000.

2. **NGINX local config**
   `infrastructure/docker/nginx.local.conf`:

   * `upstream llama_backend` with:

     * `server llama1:8080;`
     * `server llama2:8080;`
     * `server llama3:8080;`
   * `server` section:

     * `listen 8000;`
     * `location /` proxy_pass to `http://llama_backend;`
     * Proper timeouts for long responses.

3. **Local metrics & testing:**

   * If you want, add NGINX status or expose metrics via `stub_status` or use `ingress-nginx` later.
   * Run:

     * `docker-compose -f infrastructure/docker/docker-compose.llama-local.yml up`
   * Use `k6` from your host or inside a container to hit `http://localhost:8000`:

     * Scripts in `scripts/k6/`:

       * `k6_smoke.js` (few fixed prompts).
       * `k6_rps_step.js` (ramp load).

This proves: “llama.cpp instances behind an NGINX LB” works locally.

---

## Phase 3 · Kubernetes base manifests (llama.cpp-only pods + ingress-nginx)

**Goal:** General K8s definitions (no environment specifics) to deploy llama.cpp pods behind NGINX ingress.

In `infrastructure/k8s/base/`:

1. **llama.cpp Deployment**
   `serving-deployment.yaml`:

   * `Deployment` with label `app: llama-serving`.
   * Container:

     * Image: `<registry>/llama-serving:<tag>` (built from `Dockerfile.llamacpp`).
     * Args / env to pass:

       * Model path.
       * Listening port.
       * GPU/CPU flags.
     * Resource requests/limits set reasonably.
   * Volumes:

     * `PersistentVolumeClaim` for model weights (`/models`).
   * Probes:

     * `livenessProbe` hitting a lightweight endpoint (if llama.cpp provides one) or simple TCP check.
     * `readinessProbe` to ensure server is ready to accept traffic.

2. **Service**
   `serving-service.yaml`:

   * `Service` `ClusterIP`:

     * Selects `app: llama-serving`.
     * Port 8080 (or whatever the server listens on).

3. **ingress-nginx controller**
   Depending on your cluster, but in base you can include:

   * `nginx-ingress-namespace.yaml`.
   * `nginx-ingress-controller.yaml` (or use official Helm chart as a separate step).
   * The controller will expose:

     * `/metrics` endpoint for Prometheus.
     * Custom annotations for timeouts.

4. **Ingress for llama backend**
   `serving-ingress.yaml`:

   * `Ingress` that routes `/<path>` to `serving-service:8080`.
   * `kubernetes.io/ingress.class: nginx`.
   * Timeouts configured for LLM latency (e.g. `proxy-read-timeout`).

5. **Prometheus & Grafana**
   In `infrastructure/k8s/base/monitoring/`:

   * Deploy a standard Prometheus stack (via manifests or Helm).
   * Configure scrape targets:

     * `ingress-nginx-controller` pod metrics endpoint (`/metrics`).
     * Node metrics (`node-exporter`).
     * DCGM exporter if you use GPUs.
     * Optional: your custom sidecar exporter if you implement it later.

6. **Prometheus adapter for HPA**
   `prometheus-adapter.yaml`:

   * Expose NGINX metrics as custom or external metrics:

     * Example: `nginx_ingress_controller_requests{...}` as RPS per backend.
     * Example: latency metrics as histogram quantiles.
   * These metrics will feed HPA.

7. **HPA for llama-serving**
   `serving-hpa.yaml`:

   * `minReplicas`: 1.
   * `maxReplicas`: X (based on cluster capacity).
   * `metrics`:

     * Resource metric:

       * `type: Resource`, `name: cpu`, `targetAverageUtilization: ~70`.
     * Custom metric via adapter:

       * `type: Pods` or `External`, e.g. `requests_per_second` or `latency_p95`.
   * You’re not implementing queue length inside the app; approximated by:

     * Concurrent connections per upstream from NGINX.
     * High latency & dropped requests.

8. **Optional sidecar exporter**
   If you want model-level metrics (tokens/sec, etc):

   * Add a sidecar container in `serving-deployment.yaml`:

     * Image: built from `src/serving/` (Python or Go), but:

       * Only reads logs / stats from llama.cpp.
       * Only exposes `/metrics` for Prometheus.
       * It is **not** on the request path.

At this point, base K8s config is generic and reusable.

---

## Phase 4 · Nautilus deployment (K8s)

**Goal:** Run the base stack on Nautilus with environment-specific overrides.

In `infrastructure/k8s/overlays/nautilus/`:

1. **Kustomize overlay:**

   * `kustomization.yaml`:

     * `resources`:

       * `../../base/serving-deployment.yaml`
       * `../../base/serving-service.yaml`
       * `../../base/serving-ingress.yaml`
       * Monitoring components as needed.
     * `patches`:

       * `deployment-patch.yaml`.
       * `ingress-patch.yaml`.
       * `pvc-patch.yaml`.
       * `hpa-patch.yaml`.

2. **Nautilus-specific patches:**

   * `deployment-patch.yaml`:

     * `image: <nautilus-registry>/llama-serving:<tag>`.
     * Node selectors for CPU or GPU pools.
     * Adjust resource requests for Nautilus node sizes.
   * `pvc-patch.yaml`:

     * Sets `storageClassName` to Nautilus storage.
   * `ingress-patch.yaml`:

     * Customize hostnames, TLS, and annotations required by Nautilus ingress.
   * `hpa-patch.yaml`:

     * Tune HPA max replicas and thresholds to fit Nautilus capacity.

3. **Model storage on Nautilus:**

   * Define `PersistentVolume` and `PersistentVolumeClaim` YAMLs here:

     * Backed by NFS / Ceph or whatever Nautilus provides.
   * Mount to `/models` on each llama.cpp pod.

4. **Monitoring integration (Nautilus):**

   * If Nautilus already has Prometheus:

     * Ensure the `ingress-nginx` and llama pods are scraped.
   * Otherwise:

     * Deploy your base monitoring manifests here.

5. **k6 jobs in Nautilus cluster:**

   * Add `k6-job.yaml` in overlay:

     * `Job` running `loadimpact/k6` or similar image.
     * Mount `scripts/k6/*` as ConfigMap.
     * Target the Nautilus ingress hostname (e.g. `https://llama-nautilus.example`).
   * Define scenarios:

     * Steady, ramp, spike, stress, soak.

6. **Deploy to Nautilus:**

   * Script `scripts/deploy_nautilus.sh`:

     * `kubectl apply -k infrastructure/k8s/overlays/nautilus/`.
   * Confirm:

     * Pods up and ready.
     * Ingress reachable.
     * HPA scaling under `k6` load.

7. **Collect Nautilus results:**

   * Use Grafana dashboards:

     * Latency vs replicas over time.
     * RPS vs CPU utilization.
   * Export these for your report.

---

## Phase 5 · GCP / GKE with Terraform + K8s overlay

**Goal:** Provision GKE via Terraform, then reuse K8s manifests with a GCP overlay.

### 5.1 Terraform for infrastructure

Under `infrastructure/terraform/`:

1. **Modules:**

   * `modules/network/`:

     * VPC, subnets, firewall rules.
   * `modules/gke_cluster/`:

     * GKE cluster (standard, not autopilot, if you want GPU).
   * `modules/node_pool/`:

     * CPU node pool.
     * GPU node pool (if needed) with labels/taints.

2. **Environment config**
   `envs/gcp-dev/main.tf`:

   * Instantiate:

     * Network.
     * GKE cluster.
     * Node pools.
   * Output:

     * Cluster name & location.

3. **Apply:**

   * `terraform init`
   * `terraform apply` in `envs/gcp-dev`.

4. **Kubeconfig:**

   * Use `gcloud container clusters get-credentials` or Terraform output.
   * Verify with `kubectl get nodes`.

### 5.2 GCP K8s overlay

In `infrastructure/k8s/overlays/gcp/`:

1. **Kustomize overlay:**

   * Similar structure to Nautilus overlay.
   * Patches for:

     * `serving-deployment`:

       * `image: gcr.io/<project>/llama-serving:<tag>`.
       * Node selectors for CPU/GPU pools.
     * `serving-ingress`:

       * Annotations for GCE ingress:

         * Timeouts.
         * Static IP.
         * TLS certs via cert-manager or managed certificates.
     * `serving-hpa`:

       * HPA limits based on GKE capacity.

2. **Model storage on GCP:**
   Options:

   * Bake models **into the llama.cpp image** for simplicity.
     Pros: simpler PV story. Cons: large image.
   * Or:

     * Use a `PersistentDisk` or `Filestore` mounted via PVC.
     * Or GCSFuse if you are comfortable with it.

   Create the relevant PVC manifests and patch them here.

3. **Monitoring on GKE:**

   * Deploy `ingress-nginx` with Prometheus metrics enabled.
   * Either:

     * Keep your in-cluster Prometheus + Grafana and scrape ingress & pods.
     * Or integrate with Cloud Monitoring (optional, more work).

4. **Deploy to GKE:**

   * `kubectl apply -k infrastructure/k8s/overlays/gcp/`.
   * Wait for:

     * `serving-deployment` to be ready.
     * Ingress to provision an external IP / domain.

5. **k6 on GKE:**

   * Reuse `k6-job.yaml` with target set to GCP ingress hostname.
   * Run:

     * Steady, ramp, spike, stress, soak.
   * Watch:

     * HPA reaction.
     * Latency and error rates.

6. **Cost measurements:**

   * Use GCP billing / price sheets to map:

     * Node hours (CPU/GPU) to dollars.
   * Combine with:

     * Tokens generated from your dataset features (if you track them).
     * RPS & duration from k6 / Prometheus.
   * Implement a small script in `scripts/cost_report.py` to compute cost per 1 000 tokens.

---

## Phase 6 · Configuration matrix: CPU/GPU, quantization, model size

**Goal:** Confirm the stack handles all combinations you care about without changing code.

1. **Parametrize llama.cpp via env/args:**

   * In `serving-deployment.yaml`, define env vars:

     * `MODEL_PATH`
     * `USE_GPU`
     * `N_THREADS`, etc.
   * Configure per-env values in overlay patches (Nautilus vs GCP).

2. **Variants:**

   Either:

   * One Deployment, different configs rolled out sequentially.
     Or:
   * Multiple deployments:

     * `llama-serving-7b-cpu`.
     * `llama-serving-7b-gpu`.
     * `llama-serving-13b-cpu`, etc.
   * Each with its own Service/Ingress if you want parallel experiments.

3. **Dashboards:**

   * Use labels (deployment name, namespace, etc) as a proxy for:

     * Model size.
     * Quantization.
     * Hardware type.
   * Build Grafana dashboards that let you:

     * Filter by deployment name.
     * Compare latency / RPS across configs.

---

## Phase 7 · CI integration

Even without a Python app server, you still need build and deploy automation.

1. **Docker image build in CI:**

   * Pipeline:

     * Build `llama-serving` image from `infrastructure/docker/Dockerfile.llamacpp`.
     * Push to:

       * Nautilus registry.
       * GCR for GCP.

2. **K8s deployment jobs:**

   * Optional: jobs that run:

     * `kubectl apply -k .../overlays/nautilus`.
     * `kubectl apply -k .../overlays/gcp`.
   * Typically manual / protected.

3. **Terraform plan/apply:**

   * CI pipeline steps for:

     * `terraform plan` for GCP.
     * Manual `terraform apply` gated behind approvals.

---

## Sanity check vs requirements

Using this design:

* **Serving is llama.cpp HTTP server**
  No Python API in the request path.

* **NGINX is the load balancer**
  Locally via docker-compose, in Kubernetes via ingress-nginx.

* **Autoscaling via HPA**
  Uses CPU + ingress metrics (RPS / latency) via Prometheus adapter.

* **Metrics & dashboards**
  Prometheus scrapes ingress-nginx, node metrics, and optional sidecar exporter.

* **Nautilus & GCP**
  Same base manifests, environment-specific overlays; GKE infra via Terraform.

If you want, next step is to zoom in on just *one* file (e.g. `serving-deployment.yaml` or the NGINX config) and I’ll spell out a concrete version that’s actually ready to commit.
