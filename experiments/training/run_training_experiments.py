#!/usr/bin/env python3
"""
Run training performance experiments (single GPU vs distributed) via k8s overlays.

Workflow per experiment:
- Build temp kustomize bundle (overlay + per-run training.env overrides).
- kubectl apply -k bundle; wait for statefulset rollout.
- Capture logs tail from the trainer pod.
- kubectl delete -k bundle (unless --no-cleanup).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def read_env_file(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def write_env_file(path: Path, env_data: Dict[str, str]) -> None:
    lines = [f"{k}={v}" for k, v in env_data.items()]
    path.write_text("\n".join(lines) + "\n")


def kubectl_cmd(args: List[str], context: Optional[str], capture: bool = False) -> subprocess.CompletedProcess:
    cmd = ["kubectl"]
    if context:
        cmd += ["--context", context]
    cmd += args
    proc = subprocess.run(cmd, capture_output=capture, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nstdout: {proc.stdout}\nstderr: {proc.stderr}")
    return proc


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


def apply_kustomize(path: Path, context: Optional[str]) -> None:
    kubectl_cmd(["apply", "-k", str(path)], context=context)


def delete_kustomize(path: Path, context: Optional[str]) -> None:
    kubectl_cmd(["delete", "-k", str(path)], context=context)


def rollout_status(namespace: str, context: Optional[str], statefulset: str, timeout: str = "900s") -> None:
    kubectl_cmd(
        ["-n", namespace, "rollout", "status", f"statefulset/{statefulset}", f"--timeout={timeout}"],
        context=context,
    )


def wait_for_job(namespace: str, context: Optional[str], job_name: str, timeout: str = "900s") -> None:
    kubectl_cmd(
        [
            "-n",
            namespace,
            "wait",
            f"--for=condition=complete",
            f"--timeout={timeout}",
            f"job/{job_name}",
        ],
        context=context,
    )


def get_first_pod(namespace: str, context: Optional[str], selector: str) -> Optional[str]:
    proc = kubectl_cmd(
        ["-n", namespace, "get", "pods", "-l", selector, "-o", "jsonpath={.items[0].metadata.name}"],
        context=context,
        capture=True,
    )
    name = (proc.stdout or "").strip()
    return name or None


def get_pod_logs(namespace: str, context: Optional[str], pod: str, tail: int = 200) -> str:
    proc = kubectl_cmd(
        ["-n", namespace, "logs", pod, f"--tail={tail}"],
        context=context,
        capture=True,
    )
    return proc.stdout or ""


def wait_for_pods_terminated(namespace: str, context: Optional[str], selector: str, timeout_sec: int = 900) -> None:
    start = time.time()
    while True:
        proc = kubectl_cmd(
            ["-n", namespace, "get", "pods", "-l", selector, "-o", "jsonpath={.items[*].status.phase}"],
            context=context,
            capture=True,
        )
        phases = (proc.stdout or "").strip().split()
        if phases and all(p in {"Succeeded", "Failed"} for p in phases):
            return
        if time.time() - start > timeout_sec:
            raise TimeoutError(f"Timed out waiting for pods with selector {selector} to terminate")
        time.sleep(10)


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


def collect_prom_metrics(prom_url: str, window: str = "5m") -> Dict[str, Optional[float]]:
    return {
        "step_time_p50": prom_query(prom_url, 'step_time_seconds{quantile="0.5"}'),
        "step_time_p95": prom_query(prom_url, 'step_time_seconds{quantile="0.95"}'),
        "step_time_mean": prom_query(
            prom_url, "step_time_seconds_sum / step_time_seconds_count"
        ),
        "steps_per_second": prom_query(prom_url, "steps_per_second"),
        "tokens_per_second": prom_query(prom_url, "tokens_per_second"),
        "global_step": prom_query(prom_url, "training_global_step"),
        "training_loss": prom_query(prom_url, "training_loss"),
        "training_perplexity": prom_query(prom_url, "training_perplexity"),
        "cpu_util_percent": prom_query(prom_url, f'avg_over_time(cpu_utilization_percent[{window}])'),
        "gpu_util_percent": prom_query(prom_url, f'avg_over_time(gpu_utilization_percent[{window}])'),
        "gpu_mem_bytes": prom_query(prom_url, f'max_over_time(gpu_memory_used_bytes[{window}])'),
    }


def build_temp_bundle(namespace: str, overlay_path: Path, env_data: Dict[str, str]) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="training-bundle-"))
    overlay_parent_dest = tmpdir / "overlayroot"
    shutil.copytree(overlay_path.parent, overlay_parent_dest, dirs_exist_ok=True)
    training_env_path = overlay_parent_dest / "training.env"
    if not training_env_path.exists():
        training_env_path = overlay_parent_dest / "training" / "training.env"
    training_env_path.parent.mkdir(parents=True, exist_ok=True)
    write_env_file(training_env_path, env_data)
    overlay_rel = Path(shutil.os.path.relpath(overlay_parent_dest / overlay_path.name, tmpdir))
    kustomization = {
        "apiVersion": "kustomize.config.k8s.io/v1beta1",
        "kind": "Kustomization",
        "namespace": namespace,
        "resources": [overlay_rel.as_posix()],
    }
    (tmpdir / "kustomization.yaml").write_text(yaml.safe_dump(kustomization))
    return tmpdir


@dataclass
class ExperimentResult:
    name: str
    overlay: str
    bundle_path: str
    env: Dict[str, str]
    pod_name: Optional[str]
    logs_tail: str
    duration_seconds: float
    pod_phase: Optional[str]
    restart_count: Optional[int]
    prom_metrics: Dict[str, Optional[float]]


def merge_env(base_env: Dict[str, str], overrides: Dict[str, Any]) -> Dict[str, str]:
    merged = dict(base_env)
    merged.update({k: str(v) for k, v in overrides.items()})
    return merged


def run_experiment(
    exp_cfg: Dict[str, Any],
    cluster: Dict[str, Any],
    base_env: Dict[str, str],
) -> ExperimentResult:
    start_time = time.time()
    namespace = cluster["namespace"]
    context = cluster.get("context")
    overlay_key = exp_cfg["overlay"]
    overlays = cluster.get("overlays", {})
    if overlay_key not in overlays:
        raise ValueError(f"Overlay '{overlay_key}' not found in cluster.overlays")
    overlay_path = Path(overlays[overlay_key]).resolve()

    env_data = merge_env(base_env, exp_cfg.get("env_overrides", {}))
    bundle_path = build_temp_bundle(namespace, overlay_path, env_data)

    log(f"[{exp_cfg['name']}] Applying kustomize bundle: {bundle_path}")
    apply_kustomize(bundle_path, context)

    statefulset = cluster.get("statefulset_name", "smirvaki-trainer")
    log(f"[{exp_cfg['name']}] Waiting for statefulset rollout: {statefulset}")
    rollout_status(namespace, context, statefulset)

    selector = cluster.get("pod_selector", "app=smirvaki-trainer")
    pod_name = get_first_pod(namespace, context, selector)
    logs_tail = ""
    pod_phase = None
    restart_count = None
    if pod_name:
        try:
            logs_tail = get_pod_logs(namespace, context, pod_name, tail=200)
        except Exception as exc:
            warn(f"[{exp_cfg['name']}] Failed to get logs: {exc}")
        try:
            status = get_pod_status(namespace, context, pod_name)
            pod_phase = status.get("phase")
            restart_count = status.get("restart_count")
        except Exception as exc:
            warn(f"[{exp_cfg['name']}] Failed to get pod status: {exc}")

    # Wait a bounded time for trainer pods; log a warning if still running
    timeout_sec = cluster.get("trainer_timeout_sec", 200)
    try:
        wait_for_pods_terminated(namespace, context, selector, timeout_sec=timeout_sec)
    except Exception as exc:
        warn(f"[{exp_cfg['name']}] Trainer pods still running after {timeout_sec}s: {exc}")

    duration_seconds = time.time() - start_time

    prom_metrics: Dict[str, Optional[float]] = {}
    prom_svc = cluster.get("prometheus_service")
    if prom_svc:
        prom_local = int(cluster.get("local_prometheus_port", 19091))
        prom_port = int(cluster.get("prometheus_port", 9090))
        prom_resource = f"svc/{prom_svc}"
        prom_window = cluster.get("prom_window", "5m")
        try:
            with port_forward(namespace, context, prom_resource, prom_local, prom_port):
                prom_url = f"http://127.0.0.1:{prom_local}"
                prom_metrics = collect_prom_metrics(prom_url, window=prom_window)
        except Exception as exc:
            warn(f"[{exp_cfg['name']}] Failed to collect Prometheus metrics: {exc}")

    return ExperimentResult(
        name=exp_cfg["name"],
        overlay=overlay_key,
        bundle_path=str(bundle_path),
        env=env_data,
        pod_name=pod_name,
        logs_tail=logs_tail,
        duration_seconds=duration_seconds,
        pod_phase=pod_phase,
        restart_count=restart_count,
        prom_metrics=prom_metrics,
    )


def cleanup_bundles(context: Optional[str], bundle_paths: List[str]) -> None:
    for bundle in bundle_paths:
        path = Path(bundle)
        log(f"[cleanup] Deleting kustomize bundle {path}")
        try:
            delete_kustomize(path, context)
        except Exception as exc:
            warn(f"[cleanup] Failed to delete {path}: {exc}")
        shutil.rmtree(path, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training performance experiments via k8s overlays.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "training_config.example.yaml",
        help="YAML config path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/training_results.json"),
        help="Where to write results JSON.",
    )
    parser.add_argument(
        "--cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete applied bundles after each experiment (default: true).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if not cfg or "cluster" not in cfg or "experiments" not in cfg:
        raise SystemExit("Config must include 'cluster' and 'experiments'.")

    cluster = cfg["cluster"]
    base_env = read_env_file(Path(cluster.get("base_env_file", "infrastructure/k8s/aws/training/training.env")))

    # Pre-apply shared training data kustomization if specified
    training_data_kustomize = cluster.get("training_data_kustomize")
    if training_data_kustomize:
        log(f"Applying shared training data kustomize: {training_data_kustomize}")
        apply_kustomize(Path(training_data_kustomize), cluster.get("context"))
        # Wait for preprocess job to complete if specified
        training_data_job = cluster.get("training_data_job")
        if training_data_job:
            log(f"Waiting for training data job to complete: {training_data_job}")
            wait_for_job(cluster["namespace"], cluster.get("context"), training_data_job)

    results: List[Dict[str, Any]] = []
    bundles: List[str] = []
    for exp in cfg["experiments"]:
        res = run_experiment(
            exp_cfg=exp,
            cluster=cluster,
            base_env=base_env,
        )
        results.append(asdict(res))
        bundles.append(res.bundle_path)
        log(f"[{res.name}] pod={res.pod_name}")
        if args.cleanup:
            cleanup_bundles(cluster.get("context"), [res.bundle_path])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, indent=2))
    log(f"Wrote results to {args.output}")

    if not args.cleanup and bundles:
        log("[INFO] Cleanup disabled; applied bundles remain on the cluster:")
        for bp in bundles:
            log(f" - {bp}")

    # Do not delete training_data_kustomize; it is intended to persist across experiments


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
