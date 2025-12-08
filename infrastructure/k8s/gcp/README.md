# GCP Training Stack (GKE)

Kustomize manifests to run the training stack on GKE with GPU nodes and Filestore-backed RWX volumes. Prometheus/Grafana are included and scrape the trainer `/metrics` endpoint.

## Prerequisites
- GKE cluster with Workload Identity enabled.
- GPU node pool (update `cloud.google.com/gke-accelerator` values in `training/statefulset-training.yaml` if you are not using L4/T4).
- Filestore CSI `StorageClass` named `filestore-rwx` (or edit `pvc-data.yaml` to match yours).
- Training images pushed to a registry you can pull from (`training_stack`, `training_preprocess`).
- GCS bucket for exports; give the default SA or a bound KSA write access.

### Example cluster create (us-east1-c)
```bash
gcloud config set project llmtrainingserving
gcloud services enable container.googleapis.com compute.googleapis.com file.googleapis.com
gcloud container clusters create gke-llm-training-serving-stack \
  --project llmtrainingserving \
  --zone us-east1-c \
  --machine-type n1-standard-4 \
  --num-nodes 1 \
  --workload-pool=llmtrainingserving.svc.id.goog \
  --addons=GcsFuseCsiDriver,GcpFilestoreCsiDriver \
  --enable-ip-alias
gcloud container node-pools create gpu-pool \
  --cluster gke-llm-training-serving-stack \
  --project llmtrainingserving \
  --zone us-east1-c \
  --machine-type g2-standard-8 \
  --num-nodes 1 \
  --accelerator type=nvidia-l4,count=1 \
  --disk-type=pd-standard --disk-size=50 \
  --workload-metadata=GKE_METADATA
gcloud container clusters get-credentials gke-llm-training-serving-stack \
  --project llmtrainingserving \
  --zone us-east1-c
```

## Layout
- `training_data/`: RWX PVCs and preprocessing Job that writes `shard_*` and `val_shard_*` into `/data`.
- `training/`: headless service, GPU StatefulSet, DeepSpeed config, Prometheus/Grafana, export Job to GCS.
- `training_{2,4,8}_gpu/`: overlays that patch StatefulSet replicas and `TRAINER_REPLICAS`.
- Namespace and StorageClass are included in `training_data/` (and also in `training/` for convenience).

## Quickstart
1) Set images and bucket:
   - In `training/statefulset-training.yaml`, set your `training_stack` image (and `training_preprocess` in `training_data/job-preprocess.yaml`).
   - In `training/job-export-model.yaml`, set `GCS_BUCKET_URI` (or pass via env override).

2) Apply data prep (includes namespace + Filestore StorageClass):
```bash
kubectl apply -k infrastructure/k8s/gcp/training_data
kubectl -n llm-training wait --for=condition=complete job/smirvaki-preprocess-shards
kubectl -n llm-training logs job/smirvaki-preprocess-shards -f
```

3) Launch training (pick replica count):
```bash
# 1 GPU
kubectl apply -k infrastructure/k8s/gcp/training
# or 2/4/8 GPUs
kubectl apply -k infrastructure/k8s/gcp/training_2_gpu
# kubectl apply -k infrastructure/k8s/gcp/training_4_gpu
# kubectl apply -k infrastructure/k8s/gcp/training_8_gpu
```

4) Check pods/logs:
```bash
kubectl -n llm-training get pods -l app=smirvaki-trainer -o wide
kubectl -n llm-training logs statefulset/smirvaki-trainer smirvaki-trainer-0 -f
```

5) Metrics dashboards:
```bash
kubectl -n llm-training port-forward svc/smirvaki-prometheus 9090:9090
kubectl -n llm-training port-forward svc/smirvaki-grafana 3000:3000
# Login: admin / changeme (see grafana-deployment.yaml)
```

6) Export checkpoints to GCS:
```bash
kubectl -n llm-training create job --from=job/smirvaki-export-model smirvaki-export-model-$(date +%s)
kubectl -n llm-training logs job/smirvaki-export-model-<id> -f
```

7) Cleanup:
```bash
kubectl delete -k infrastructure/k8s/gcp/training
kubectl delete -k infrastructure/k8s/gcp/training_data
```

## Tweaks
- Edit `training/training.env` (and overlay versions) to change hyperparameters and `TRAINER_REPLICAS`.
- Adjust PVC sizes in `training_data/pvc-data.yaml`.
- If using a different StorageClass or zone, update `training_data/storageclass-filestore-rwx.yaml` (tier/network/location) and match the PVCs.
- To use Managed Service for Prometheus instead, remove the bundled Prometheus/Grafana and rely on pod scrape annotations already present.***
