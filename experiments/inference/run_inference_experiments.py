#!/usr/bin/env python3
"""
Deploy the inference serving stack on Kubernetes, drive load, and pull metrics from Prometheus.

This follows the inference experiments in experiments.md:
- Quantization sweep on CPU (TinyLlama 1.1B, Phi-2).
- Repeat on GPU for CPU vs GPU comparison.

Workflow per experiment:
1) Apply a model env ConfigMap (HF/S3 source, threads/batch/ctx/gpu layers).
2) Recreate the model download job + serving deployment via kustomize overlay (cpu/gpu).
3) Wait for the download job to finish and serving deployment to become ready.
4) Port-forward serving + Prometheus, send a burst of chat requests, and read latency/tokens/memory from Prometheus.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml


# ----------------------------
# Helpers
# ----------------------------


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def read_env_file(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def kubectl_cmd(
    args: List[str],
    context: Optional[str],
    input_str: Optional[str] = None,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    cmd = ["kubectl"]
    if context:
        cmd += ["--context", context]
    cmd += args
    proc = subprocess.run(
        cmd,
        input=input_str if input_str is not None else None,
        capture_output=capture,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return proc


def apply_configmap(
    namespace: str,
    context: Optional[str],
    env_data: Dict[str, str],
) -> None:
    manifest = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "smirvaki-llama-model-config", "namespace": namespace},
        "data": env_data,
    }
    yaml_body = yaml.safe_dump(manifest)
    kubectl_cmd(["apply", "-f", "-"], context=context, input_str=yaml_body)


def apply_kustomize(path: Path, context: Optional[str]) -> None:
    kubectl_cmd(["apply", "-k", str(path)], context=context)


def delete_job(namespace: str, context: Optional[str], name: str) -> None:
    kubectl_cmd(
        ["-n", namespace, "delete", "job", name, "--ignore-not-found=true"],
        context=context,
    )


def wait_for_job(namespace: str, context: Optional[str], name: str, timeout: str = "900s") -> None:
    kubectl_cmd(
        [
            "-n",
            namespace,
            "wait",
            f"--for=condition=complete",
            f"--timeout={timeout}",
            f"job/{name}",
        ],
        context=context,
    )


def rollout_restart(namespace: str, context: Optional[str], deployment: str) -> None:
    kubectl_cmd(
        ["-n", namespace, "rollout", "restart", f"deploy/{deployment}"],
        context=context,
    )


def rollout_status(namespace: str, context: Optional[str], deployment: str, timeout: str = "600s") -> None:
    kubectl_cmd(
        [
            "-n",
            namespace,
            "rollout",
            "status",
            f"deploy/{deployment}",
            f"--timeout={timeout}",
        ],
        context=context,
    )


def delete_configmap(namespace: str, context: Optional[str], name: str) -> None:
    kubectl_cmd(
        ["-n", namespace, "delete", "configmap", name, "--ignore-not-found=true"],
        context=context,
    )


@contextmanager
def port_forward(namespace: str, context: Optional[str], resource: str, local: int, remote: int):
    cmd = ["kubectl"]
    if context:
        cmd += ["--context", context]
    cmd += ["-n", namespace, "port-forward", resource, f"{local}:{remote}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        start = time.time()
        ready = False
        while time.time() - start < 10:
            if proc.poll() is not None:
                raise RuntimeError(f"Port-forward {resource} exited early")
            line = proc.stdout.readline() if proc.stdout else ""
            if "Forwarding from" in line:
                ready = True
                break
        if not ready:
            raise RuntimeError(f"Port-forward {resource} not ready after 10s")
        yield
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


# ----------------------------
# Load generation
# ----------------------------


async def run_load(
    base_url: str,
    prompts: List[str],
    total_requests: int,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    route: str = "chat_completions",
    timeout: float = 60.0,
) -> Dict[str, Any]:
    path = "/v1/chat/completions" if route == "chat_completions" else "/completion"
    queue: asyncio.Queue[str] = asyncio.Queue()
    for i in range(total_requests):
        queue.put_nowait(prompts[i % len(prompts)])

    results: List[int] = []
    errors: List[str] = []

    async def worker(client: httpx.AsyncClient) -> None:
        while True:
            try:
                prompt = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            payload = {
                "model": "llama",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            try:
                resp = await client.post(f"{base_url}{path}", json=payload)
                results.append(resp.status_code)
            except Exception as exc:
                errors.append(str(exc))
            finally:
                queue.task_done()

    async with httpx.AsyncClient(timeout=timeout) as client:
        workers = [asyncio.create_task(worker(client)) for _ in range(concurrency)]
        await asyncio.gather(*workers)

    return {
        "status_counts": {code: results.count(code) for code in set(results)},
        "errors": errors,
    }


# ----------------------------
# Prometheus helpers
# ----------------------------


def prom_query(prom_url: str, query: str) -> Optional[float]:
    try:
        resp = httpx.get(f"{prom_url}/api/v1/query", params={"query": query}, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            return None
        result = data.get("data", {}).get("result")
        if not result:
            return None
        value = result[0].get("value")
        if not value or len(value) < 2:
            return None
        return float(value[1])
    except Exception:
        return None


def collect_prom_metrics(prom_url: str, route: str, window: str) -> Dict[str, Optional[float]]:
    return {
        "latency_p50": prom_query(
            prom_url,
            f'histogram_quantile(0.5, sum(rate(llama_request_latency_seconds_bucket{{route="{route}"}}[{window}])) by (le))',
        ),
        "latency_p95": prom_query(
            prom_url,
            f'histogram_quantile(0.95, sum(rate(llama_request_latency_seconds_bucket{{route="{route}"}}[{window}])) by (le))',
        ),
        "ttft_p95": prom_query(
            prom_url,
            f'histogram_quantile(0.95, sum(rate(llama_request_ttft_seconds_bucket{{route="{route}"}}[{window}])) by (le))',
        ),
        "tokens_per_second": prom_query(
            prom_url,
            f'sum(rate(llama_tokens_total{{route="{route}",type="completion"}}[{window}]))',
        ),
        "cpu_util_percent": prom_query(
            prom_url,
            f'avg_over_time(llama_cpu_utilization_percent{{pid="container"}}[{window}])',
        ),
        "rss_bytes": prom_query(
            prom_url,
            f'max_over_time(llama_memory_rss_bytes{{pid="container"}}[{window}])',
        ),
        "gpu_mem_bytes": prom_query(
            prom_url,
            f'max_over_time(llama_gpu_memory_used_bytes[{window}])',
        ),
        "gpu_util_percent": prom_query(
            prom_url,
            f'avg_over_time(llama_gpu_utilization_percent[{window}])',
        ),
    }


# ----------------------------
# Experiment runner
# ----------------------------


@dataclass
class ExperimentResult:
    name: str
    overlay: str
    env: Dict[str, str]
    load: Dict[str, Any]
    prom_metrics: Dict[str, Optional[float]]
    load_status: Dict[str, Any]


def merge_env(base_env: Dict[str, str], overrides: Dict[str, Any]) -> Dict[str, str]:
    merged = dict(base_env)
    merged.update({k: str(v) for k, v in overrides.items()})
    return merged


def run_experiment(
    exp_cfg: Dict[str, Any],
    cluster: Dict[str, Any],
    load_defaults: Dict[str, Any],
    base_env: Dict[str, str],
    prom_window: str,
) -> ExperimentResult:
    namespace = cluster["namespace"]
    context = cluster.get("context")
    overlay_key = exp_cfg["overlay"]
    overlays = cluster.get("overlays", {})
    if overlay_key not in overlays:
        raise ValueError(f"Overlay '{overlay_key}' not found in cluster.overlays")
    overlay_path = Path(overlays[overlay_key]).resolve()

    env_overrides = exp_cfg.get("env_overrides", {})
    env_data = merge_env(base_env, env_overrides)

    log(f"[{exp_cfg['name']}] Applying ConfigMap for model/env settings")
    apply_configmap(namespace, context, env_data)

    log(f"[{exp_cfg['name']}] Deleting previous model download job (if any)")
    delete_job(namespace, context, "smirvaki-llama-model-download")

    log(f"[{exp_cfg['name']}] Applying kustomize overlay: {overlay_path}")
    apply_kustomize(overlay_path, context)

    log(f"[{exp_cfg['name']}] Waiting for model download job to complete")
    wait_for_job(namespace, context, "smirvaki-llama-model-download")

    log(f"[{exp_cfg['name']}] Restarting serving deployment")
    rollout_restart(namespace, context, "smirvaki-llama-serving")
    rollout_status(namespace, context, "smirvaki-llama-serving")

    serving_resource = f"svc/{cluster['service_name']}"
    prom_resource = f"svc/{cluster['prometheus_service']}"
    serving_local = int(cluster["local_service_port"])
    prom_local = int(cluster["local_prometheus_port"])
    serving_port = int(cluster.get("service_port", 80))
    prom_port = int(cluster.get("prometheus_port", 9090))

    load_cfg = dict(load_defaults)
    load_cfg.update({k: v for k, v in exp_cfg.items() if k in {"prompts", "total_requests", "concurrency", "max_tokens", "temperature", "route"}})
    prompts = load_cfg.get("prompts") or ["Hello!"]

    log(f"[{exp_cfg['name']}] Port-forwarding services and sending load")
    with port_forward(namespace, context, serving_resource, serving_local, serving_port), port_forward(
        namespace, context, prom_resource, prom_local, prom_port
    ):
        base_url = f"http://127.0.0.1:{serving_local}"
        prom_url = f"http://127.0.0.1:{prom_local}"

        load_status = asyncio.run(
            run_load(
                base_url=base_url,
                prompts=prompts,
                total_requests=int(load_cfg.get("total_requests", 10)),
                concurrency=int(load_cfg.get("concurrency", 2)),
                max_tokens=int(load_cfg.get("max_tokens", 128)),
                temperature=float(load_cfg.get("temperature", 0.1)),
                route=load_cfg.get("route", "chat_completions"),
            )
        )

        # Give Prometheus a moment to scrape the recent requests.
        time.sleep(10)
        prom_metrics = collect_prom_metrics(
            prom_url=prom_url,
            route=load_cfg.get("route", "chat_completions"),
            window=prom_window,
        )

    return ExperimentResult(
        name=exp_cfg["name"],
        overlay=overlay_key,
        env=env_data,
        load=load_cfg,
        prom_metrics=prom_metrics,
        load_status=load_status,
    )


def cleanup_cluster(
    cluster: Dict[str, Any],
    overlays_used: List[str],
    delete_cm: bool = True,
) -> None:
    namespace = cluster["namespace"]
    context = cluster.get("context")
    overlays = cluster.get("overlays", {})

    for overlay_key in dict.fromkeys(overlays_used):
        if overlay_key not in overlays:
            continue
        overlay_path = Path(overlays[overlay_key]).resolve()
        log(f"[cleanup] Deleting overlay {overlay_key} ({overlay_path})")
        try:
            kubectl_cmd(["delete", "-k", str(overlay_path)], context=context)
        except Exception as exc:
            warn(f"[cleanup] Failed to delete overlay {overlay_key}: {exc}")

    log("[cleanup] Deleting model download job (best-effort)")
    try:
        delete_job(namespace, context, "smirvaki-llama-model-download")
    except Exception as exc:
        warn(f"[cleanup] Failed to delete job: {exc}")

    if delete_cm:
        log("[cleanup] Deleting model ConfigMap smirvaki-llama-model-config")
        try:
            delete_configmap(namespace, context, "smirvaki-llama-model-config")
        except Exception as exc:
            warn(f"[cleanup] Failed to delete ConfigMap: {exc}")


# ----------------------------
# CLI
# ----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference experiments via k8s + Prometheus.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs/inference_models.example.yaml",
        help="YAML config describing cluster settings and experiments.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/inference_perf.json"),
        help="Path to write JSON results.",
    )
    parser.add_argument(
        "--cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete applied overlays and ConfigMap after the run (default: true).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if not cfg or "cluster" not in cfg or "experiments" not in cfg:
        raise SystemExit("Config must include 'cluster' and 'experiments' sections.")

    cluster = cfg["cluster"]
    load_defaults = cfg.get("load", {})
    prom_window = cfg.get("prom_window", "5m")

    base_env_path = Path(cluster.get("base_env_file", "infrastructure/k8s/aws/serving/model.env"))
    base_env = read_env_file(base_env_path)

    results: List[Dict[str, Any]] = []
    overlays_used: List[str] = []
    for exp in cfg["experiments"]:
        overlays_used.append(exp["overlay"])
        result = run_experiment(
            exp_cfg=exp,
            cluster=cluster,
            load_defaults=load_defaults,
            base_env=base_env,
            prom_window=prom_window,
        )
        results.append(asdict(result))
        log(
            f"[{exp['name']}] lat_p50={result.prom_metrics.get('latency_p50')} "
            f"lat_p95={result.prom_metrics.get('latency_p95')} "
            f"tok_s={result.prom_metrics.get('tokens_per_second')}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, indent=2))
    log(f"Wrote results to {args.output}")

    if args.cleanup:
        cleanup_cluster(cluster, overlays_used)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
