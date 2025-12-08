#!/usr/bin/env python3
"""
Run elastic serving + autoscaling experiments via k8s deploy/teardown and k6 load.

Workflow per experiment:
1) Build temp kustomize bundle (overlay + per-run model.env overrides).
2) kubectl apply -k bundle; wait for download job + rollout.
3) Port-forward serving + Prometheus, run k6 load (script from config) against service URL.
4) Query Prometheus for latency/TTFT and utilization metrics.
5) kubectl delete -k bundle (unless --no-cleanup).
"""
from __future__ import annotations

import argparse
import json
import os
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


def kubectl_cmd(args: List[str], context: Optional[str]) -> None:
    cmd = ["kubectl"]
    if context:
        cmd += ["--context", context]
    cmd += args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nstdout: {proc.stdout}\nstderr: {proc.stderr}")


def apply_kustomize(path: Path, context: Optional[str]) -> None:
    kubectl_cmd(["apply", "-k", str(path)], context=context)


def delete_kustomize(path: Path, context: Optional[str]) -> None:
    kubectl_cmd(["delete", "-k", str(path)], context=context)


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


def build_temp_bundle(namespace: str, overlay_path: Path, env_data: Dict[str, str]) -> Path:
    """Copy overlay root, override serving/model.env, create kustomization."""
    tmpdir = Path(tempfile.mkdtemp(prefix="autoscale-bundle-"))
    overlay_parent_dest = tmpdir / "overlayroot"
    shutil.copytree(overlay_path.parent, overlay_parent_dest, dirs_exist_ok=True)

    model_env_path = overlay_parent_dest / "serving" / "model.env"
    model_env_path.parent.mkdir(parents=True, exist_ok=True)
    write_env_file(model_env_path, env_data)

    overlay_rel = Path(os.path.relpath(overlay_parent_dest / overlay_path.name, tmpdir))
    kustomization = {
        "apiVersion": "kustomize.config.k8s.io/v1beta1",
        "kind": "Kustomization",
        "namespace": namespace,
        "resources": [overlay_rel.as_posix()],
    }
    (tmpdir / "kustomization.yaml").write_text(yaml.safe_dump(kustomization))
    return tmpdir


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


def slo_percent(prom_url: str, histogram_metric: str, route: str, le: float, window: str) -> Optional[float]:
    le_re = f"{le}.0"  # match both "2" and "2.0" style bucket labels

    # Try a few likely label keys first (different instrumentations name this differently).
    for key in ("route", "endpoint", "handler", "path"):
        print(
            f'sum(rate({histogram_metric}_bucket{{{key}="{route}",le=~"{le_re}"}}[{window}]))'
        )
        num = prom_query(
            prom_url,
            f'sum(rate({histogram_metric}_bucket{{{key}="{route}",le=~"{le_re}"}}[{window}]))',
        )
        den = prom_query(
            prom_url,
            f'sum(rate({histogram_metric}_count{{{key}="{route}"}}[{window}]))',
        )
        if num is not None and den not in (None, 0):
            return (num / den) * 100.0

    # Fallback: compute across all routes (better than silently returning 0.0).
    num = prom_query(
        prom_url,
        f'sum(rate({histogram_metric}_bucket{{le=~"{le_re}"}}[{window}]))',
    )
    den = prom_query(
        prom_url,
        f'sum(rate({histogram_metric}_count[{window}]))',
    )
    if num is None or den in (None, 0):
        return None
    return (num / den) * 100.0


def collect_prom_metrics(prom_url: str, route: str, window: str) -> Dict[str, Optional[float]]:
    return {
        "ttft_p95": prom_query(
            prom_url,
            f'histogram_quantile(0.95, sum(rate(llama_request_ttft_seconds_bucket{{route="{route}"}}[{window}])) by (le))',
        ),
        "latency_p95": prom_query(
            prom_url,
            f'histogram_quantile(0.95, sum(rate(llama_request_total_seconds_bucket{{route="{route}"}}[{window}])) by (le))',
        ),
        "ttft_slo_lt_2s_pct": slo_percent(prom_url, "llama_request_ttft_seconds", route, 2, window),
        "ttft_slo_lt_5s_pct": slo_percent(prom_url, "llama_request_ttft_seconds", route, 5, window),
        "ttft_slo_lt_10s_pct": slo_percent(prom_url, "llama_request_ttft_seconds", route, 10, window),
        "ttft_slo_lt_20s_pct": slo_percent(prom_url, "llama_request_ttft_seconds", route, 20, window),
        "ttft_slo_lt_30s_pct": slo_percent(prom_url, "llama_request_ttft_seconds", route, 30, window),

        # Use the same histogram family as `latency_p95` (llama_request_total_seconds_*).
        "latency_lt_1s_pct": slo_percent(prom_url, "llama_request_latency_seconds", route, 1, window),
        "latency_lt_5s_pct": slo_percent(prom_url, "llama_request_latency_seconds", route, 5, window),
        "latency_lt_10s_pct": slo_percent(prom_url, "llama_request_latency_seconds", route, 10, window),
        "cpu_util_percent": prom_query(
            prom_url,
            f'avg_over_time(llama_cpu_utilization_percent{{pid="container"}}[{window}])',
        ),
        "gpu_util_percent": prom_query(
            prom_url,
            f'avg_over_time(llama_gpu_utilization_percent[{window}])',
        ),
        "gpu_mem_bytes": prom_query(
            prom_url,
            f'max_over_time(llama_gpu_memory_used_bytes[{window}])',
        ),
    }


def run_k6(script: Path, service_url: str, vus: Optional[int], duration: Optional[str], extra_args: List[str]) -> Dict[str, Any]:
    if shutil.which("k6") is None:
        raise RuntimeError("k6 binary not found in PATH")
    cmd = ["k6", "run", "--quiet", "--env", f"SERVICE_URL={service_url}"]
    if vus:
        cmd += ["--vus", str(vus)]
    if duration:
        cmd += ["--duration", duration]
    cmd += extra_args
    cmd.append(str(script))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        warn(f"k6 exited with code {proc.returncode}: {proc.stderr}")
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


@dataclass
class ExperimentResult:
    name: str
    overlay: str
    bundle_path: str
    env: Dict[str, str]
    k6: Dict[str, Any]
    prom_metrics: Dict[str, Optional[float]]


def merge_env(base_env: Dict[str, str], overrides: Dict[str, Any]) -> Dict[str, str]:
    merged = dict(base_env)
    merged.update({k: str(v) for k, v in overrides.items()})
    return merged


def run_experiment(
    exp_cfg: Dict[str, Any],
    cluster: Dict[str, Any],
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

    env_data = merge_env(base_env, exp_cfg.get("env_overrides", {}))
    bundle_path = build_temp_bundle(namespace, overlay_path, env_data)

    log(f"[{exp_cfg['name']}] Applying kustomize bundle: {bundle_path}")
    apply_kustomize(bundle_path, context)

    log(f"[{exp_cfg['name']}] Waiting for model download job to complete")
    wait_for_job(namespace, context, cluster.get("download_job", "smirvaki-llama-model-download"))
    rollout_status(namespace, context, cluster.get("deployment_name", "smirvaki-llama-serving"))

    service_resource = f"svc/{cluster['service_name']}"
    prom_resource = f"svc/{cluster['prometheus_service']}"
    serving_local = int(cluster["local_service_port"])
    prom_local = int(cluster["local_prometheus_port"])
    serving_port = int(cluster.get("service_port", 80))
    prom_port = int(cluster.get("prometheus_port", 9090))

    k6_script = Path(exp_cfg["k6_script"]).resolve()
    k6_vus = exp_cfg.get("k6_vus")
    k6_duration = exp_cfg.get("k6_duration")
    k6_extra = exp_cfg.get("k6_extra_args", []) or []
    route = exp_cfg.get("route", "chat_completions")

    log(f"[{exp_cfg['name']}] Port-forwarding services and running k6")
    with port_forward(namespace, context, service_resource, serving_local, serving_port), port_forward(
        namespace, context, prom_resource, prom_local, prom_port
    ):
        service_url = f"http://127.0.0.1:{serving_local}"
        prom_url = f"http://127.0.0.1:{prom_local}"

        k6_result = run_k6(
            script=k6_script,
            service_url=service_url,
            vus=None,
            duration=None,
            extra_args=k6_extra,
        )

        time.sleep(10)
        prom_metrics = collect_prom_metrics(prom_url, route=route, window=prom_window)

    return ExperimentResult(
        name=exp_cfg["name"],
        overlay=overlay_key,
        bundle_path=str(bundle_path),
        env=env_data,
        k6=k6_result,
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
    parser = argparse.ArgumentParser(description="Run autoscaling experiments via k8s + k6 + Prometheus.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "autoscaling_config.example.yaml",
        help="YAML config path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/autoscaling_results.json"),
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
    prom_window = cfg.get("prom_window", "5m")
    base_env = read_env_file(Path(cluster.get("base_env_file", "infrastructure/k8s/aws/serving/model.env")))

    results: List[Dict[str, Any]] = []
    applied_bundles: List[str] = []

    for exp in cfg["experiments"]:
        res = run_experiment(
            exp_cfg=exp,
            cluster=cluster,
            base_env=base_env,
            prom_window=prom_window,
        )
        results.append(asdict(res))
        applied_bundles.append(res.bundle_path)
        log(
            f"[{res.name}] ttft_p95={res.prom_metrics.get('ttft_p95')} "
            f"lat_p95={res.prom_metrics.get('latency_p95')} "
            f"ttft<2s%={res.prom_metrics.get('ttft_slo_lt_2s_pct')}"
        )
        if args.cleanup:
            cleanup_bundles(cluster.get("context"), [res.bundle_path])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, indent=2))
    log(f"Wrote results to {args.output}")

    if not args.cleanup and applied_bundles:
        log("[INFO] Cleanup disabled; applied bundles remain on the cluster:")
        for bp in applied_bundles:
            log(f" - {bp}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
