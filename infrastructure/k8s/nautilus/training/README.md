## Nautilus Deployment

1. **Build and push the image**

   ```bash
   # Target Nautilus (amd64). On Apple Silicon/arm, use buildx for the correct arch:
   # docker buildx build --platform linux/amd64 -t docker.io/esmaeilmirvakili/training_stack:latest -f infrastructure/docker/Dockerfile.training --push .
   docker build -t docker.io/esmaeilmirvakili/training_stack:latest -f infrastructure/docker/Dockerfile.training .
   docker push docker.io/esmaeilmirvakili/training_stack:latest

   # Lightweight preprocessing image (CPU-only, no CUDA deps); build for amd64 to avoid exec format errors.
   # docker buildx build --platform linux/amd64 -t docker.io/esmaeilmirvakili/training_preprocess:latest -f infrastructure/docker/Dockerfile.preprocessing --push .
   docker build -t docker.io/esmaeilmirvakili/training_preprocess:latest -f infrastructure/docker/Dockerfile.preprocessing .
   docker push docker.io/esmaeilmirvakili/training_preprocess:latest
   ```

   (If you haven't enabled buildx, run `docker buildx create --use` once. The preprocessing image omits torch/CUDA to stay small; still target amd64. Already reflected in the manifests.)

2. **Prepare data (PVCs + preprocessing job)**

   Make sure the `cse239fall2025` namespace already exists (course namespaces are pre-provisioned on Nautilus), then apply the data stack (PVCs + preprocessing job):

   ```bash
   kubectl apply -k infrastructure/k8s/nautilus/training_data

   kubectl -n cse239fall2025 wait --for=condition=complete job/smirvaki-preprocess-shards
   kubectl -n cse239fall2025 get pods -l app=smirvaki-preprocess-shards
   kubectl -n cse239fall2025 get job smirvaki-preprocess-shards
   kubectl -n cse239fall2025 logs job/smirvaki-preprocess-shards -f
   ```

   This uses the lightweight preprocessing image (`training_preprocess:latest`) and writes `shard_*` and `val_shard_*` directories into the `smirvaki-training-data` PVC. Preprocess shard counts come from `training/training.env` (`PREPROCESS_TRAIN_SHARDS`, `PREPROCESS_VAL_SHARDS`). Re-run the kustomize apply if you delete the job and want to regenerate shards.

3. **Set training params**

   Edit `infrastructure/k8s/nautilus/training/training.env` if you want to change preprocessing shard counts (`PREPROCESS_TRAIN_SHARDS`, `PREPROCESS_VAL_SHARDS`), the trainer replica count (also sets WORLD_SIZE), or override training hyperparameters (model name/max_length, LoRA target modules, batch size, epochs, lr, use_lora, max_steps). These env vars are injected into the pods and passed to `train-llm`. Tweak `deepspeed_zero2.json` as needed. (Apply any image pull secrets you need in `cse239fall2025` as well.)

4. **Launch training**

   ```bash
   # Default (1 trainer): uses infrastructure/k8s/nautilus/training
   kubectl apply -k infrastructure/k8s/nautilus/training  # reads training.env -> configmap + WORLD_SIZE; pods wait for Prometheus health before starting

   # Preset overlays (scale to 2/4/8 trainers with matching WORLD_SIZE):
   # kubectl apply -k infrastructure/k8s/nautilus/training_2_gpu
   # kubectl apply -k infrastructure/k8s/nautilus/training_4_gpu
   # kubectl apply -k infrastructure/k8s/nautilus/training_8_gpu

   kubectl -n cse239fall2025 describe statefulset smirvaki-trainer
   kubectl -n cse239fall2025 get pods -l app=smirvaki-trainer -o wide
   kubectl -n cse239fall2025 describe pod smirvaki-trainer-0
   kubectl -n cse239fall2025 get statefulset smirvaki-trainer
   kubectl -n cse239fall2025 logs statefulset/smirvaki-trainer smirvaki-trainer-0 -f
   kubectl -n cse239fall2025 exec -it smirvaki-trainer-0 -- bash

   ```

   Check logs per replica:

   ```bash
   kubectl -n cse239fall2025 logs statefulset/smirvaki-trainer smirvaki-trainer-0 -f
   ```

   Training shards are named `shard_<id>`; each rank trains on the subset where `id % WORLD_SIZE == rank`.

5. **Export checkpoints and cleanup**

   After training finishes (StatefulSet pods show `Completed`), upload the saved model from the shared PVC to the Nautilus object store and delete both training PVCs in one step:

   ```bash
   # Create/update the Nautilus RGW credential secret once.
   # Provide a .env file with ACCESS_KEY and SECRET_KEY (or S3_ACCESS_KEY/S3_SECRET_KEY).
   ./scripts/configure_nautilus_s3_secret.sh --namespace cse239fall2025 --env-file path/to/nautilus.env

   # S3_BUCKET_URI inside job-export-model.yaml defaults to s3://smirvaki-training-data; adjust if needed.
   # then run the export/cleanup helper.
   ./scripts/export_and_cleanup.sh
   ```

   The helper job in `job-export-model.yaml` mounts `smirvaki-training-outputs`, writes a `~/.s3cfg` using the secret, runs `s3cmd sync` against the Nautilus RGW endpoint (`rook-ceph-rgw-nautiluss3.rook`), and the script deletes both PVCs once the upload succeeds.

6. **Metrics stack (Prometheus + Grafana)**

   ```bash
   # Already part of the kustomization; reapply here only if you skipped earlier
   kubectl apply -k infrastructure/k8s/nautilus/training

   # Port-forward Prometheus
   kubectl -n cse239fall2025 port-forward svc/smirvaki-prometheus 9090:9090

   # Port-forward Grafana and log in with smirvaki-grafana-admin secret (change password in grafana-deployment.yaml)
   kubectl -n cse239fall2025 port-forward svc/smirvaki-grafana 3000:3000
   ```

   Prometheus scrapes the `metrics` port (8000) via static targets (currently only trainer-0 is listed); update `prometheus-deployment.yaml` if you change replica counts or service names. Grafana ships with a starter dashboard you can extend.

7. **Cleanup**

   ```bash
   kubectl delete -n cse239fall2025 statefulset smirvaki-trainer service smirvaki-trainer
   kubectl delete -n cse239fall2025 job smirvaki-preprocess-shards
   # delete PVCs if you no longer need the shards/outputs
   ```
Adjust resource requests, deepspeed overrides, and Hydra flags as needed for your Nautilus quota.
