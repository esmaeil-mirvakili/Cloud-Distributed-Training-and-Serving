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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def apply_kustomize(path: Path, context: Optional[str]) -> None:
    kubectl_cmd(["apply", "-k", str(path)], context=context)


def delete_kustomize(path: Path, context: Optional[str]) -> None:
    kubectl_cmd(["delete", "-k", str(path)], context=context)


def rollout_status(namespace: str, context: Optional[str], statefulset: str, timeout: str = "900s") -> None:
    kubectl_cmd(
        ["-n", namespace, "rollout", "status", f"statefulset/{statefulset}", f"--timeout={timeout}"],
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


def merge_env(base_env: Dict[str, str], overrides: Dict[str, Any]) -> Dict[str, str]:
    merged = dict(base_env)
    merged.update({k: str(v) for k, v in overrides.items()})
    return merged


def run_experiment(
    exp_cfg: Dict[str, Any],
    cluster: Dict[str, Any],
    base_env: Dict[str, str],
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

    statefulset = cluster.get("statefulset_name", "smirvaki-trainer")
    log(f"[{exp_cfg['name']}] Waiting for statefulset rollout: {statefulset}")
    rollout_status(namespace, context, statefulset)

    selector = cluster.get("pod_selector", "app=smirvaki-trainer")
    pod_name = get_first_pod(namespace, context, selector)
    logs_tail = ""
    if pod_name:
        try:
            logs_tail = get_pod_logs(namespace, context, pod_name, tail=200)
        except Exception as exc:
            warn(f"[{exp_cfg['name']}] Failed to get logs: {exc}")

    return ExperimentResult(
        name=exp_cfg["name"],
        overlay=overlay_key,
        bundle_path=str(bundle_path),
        env=env_data,
        pod_name=pod_name,
        logs_tail=logs_tail,
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
