## Training Image

Build the CUDA-enabled trainer image from the repo root:

```bash
docker build -t training-stack:local -f infrastructure/docker/training.Dockerfile .
```

Run a quick sanity check (mounting local shards + outputs and forwarding the metrics port):

```bash
docker run --rm --gpus all \
  -v $PWD/data:/data \
  -v $PWD/outputs:/outputs \
  -p 8000:8000 \
  training-stack:local \
  mode=train train.data_dir=/data train.output_dir=/outputs train.max_steps=10
```

Override the command (e.g., to open a shell) with `--entrypoint` or `command`:

```bash
docker run --rm -it --entrypoint bash training-stack:local
```

### Multi-node DeepSpeed smoke test

The repository includes `infrastructure/docker/docker-compose.deepspeed.yaml` to spin up a three-node DeepSpeed run (each node hosts one GPU process and consumes its own shard). Workflow:

1. Build the training image if you haven’t already:
   ```bash
   docker build -t training-stack:local -f infrastructure/docker/training.Dockerfile .
   ```
2. Ensure the shared `data/` and `outputs/` directories exist locally (the compose file mounts them).
3. Launch the compose stack:
   ```bash
   docker compose -f infrastructure/docker/docker-compose.deepspeed.yaml up --abort-on-container-exit
   ```
   - `preprocess` runs first and writes three shards into `./data`.
   - `trainer0`, `trainer1`, `trainer2` each launch `deepspeed` with `WORLD_SIZE=3` and consume shard `shard_{rank}.pt`.
   - Outputs land under `./outputs/rank{0,1,2}` and Prometheus metrics stay on port 8000 inside each container.
4. Tear everything down (and optionally remove containers/volumes) once you’re done testing:
   ```bash
   docker compose -f infrastructure/docker/docker-compose.deepspeed.yaml down
   ```

Feel free to tweak commands (dataset, `train.max_steps`, etc.) directly in the compose file for faster experiments.
