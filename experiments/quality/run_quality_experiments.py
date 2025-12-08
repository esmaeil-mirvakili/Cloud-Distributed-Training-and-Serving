#!/usr/bin/env python3
"""
Run the output quality experiments (WildChat subset) with k8s deploy/teardown per model.

Workflow per experiment (similar to inference_perf runner):
1) Build a temp kustomize bundle combining the selected overlay (cpu/gpu) and a per-run model.env.
2) kubectl apply -k bundle; wait for model download job + serving rollout.
3) Port-forward service, run the WildChat subset through the model, compute ROUGE-L/ BLEU / BERTScore.
4) kubectl delete -k bundle (unless --no-cleanup).
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
import sacrebleu
import yaml
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from tqdm import tqdm


# ------------- k8s helpers ----------------

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
    """Copy overlay root, override serving/model.env, create a kustomization."""
    tmpdir = Path(tempfile.mkdtemp(prefix="quality-bundle-"))

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


# ------------- quality evaluation ----------------


def load_dataset(path: Path, prompt_field: str, reference_field: str, limit: Optional[int]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with path.open() as f:
        for line in f:
            if limit and len(records) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get(prompt_field)
            ref = obj.get(reference_field)
            if prompt is None or ref is None:
                continue
            records.append({"prompt": prompt, "reference": ref})
    if not records:
        raise ValueError(f"No records loaded from {path}")
    return records


def fetch_response(base_url: str, route: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float) -> str:
    url = f"{base_url.rstrip('/')}{route}"
    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content")
        return "" if content is None else str(content)
    except Exception as exc:
        warn(f"Inference error: {exc}")
        return ""


def compute_metrics(references: List[str], predictions: List[str], skip_bertscore: bool = False) -> Dict[str, Any]:
    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_f = sum(rouge.score(r, p)["rougeL"].fmeasure for r, p in zip(references, predictions)) / len(references)
    metrics: Dict[str, Any] = {"bleu": bleu, "rougeL_f": rouge_l_f}
    if not skip_bertscore:
        _, _, F = bert_score(predictions, references, lang="en", verbose=False)
        metrics["bertscore_f1"] = float(F.mean())
    else:
        metrics["bertscore_f1"] = None
    return metrics


@dataclass
class ExperimentResult:
    name: str
    overlay: str
    bundle_path: str
    env: Dict[str, str]
    metrics: Dict[str, Any]
    errors: int


def merge_env(base_env: Dict[str, str], overrides: Dict[str, Any]) -> Dict[str, str]:
    merged = dict(base_env)
    merged.update({k: str(v) for k, v in overrides.items()})
    return merged


def run_experiment(
    exp_cfg: Dict[str, Any],
    cluster: Dict[str, Any],
    base_env: Dict[str, str],
    records: List[Dict[str, str]],
    skip_bertscore: bool,
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
    serving_local = int(cluster["local_service_port"])
    serving_port = int(cluster.get("service_port", 80))

    route = exp_cfg.get("route", "/v1/chat/completions")
    model_id = exp_cfg.get("model_id", "llama")
    temperature = float(exp_cfg.get("temperature", 0.1))
    max_tokens = int(exp_cfg.get("max_tokens", 256))
    headers = exp_cfg.get("headers", {}) or {}
    timeout = float(exp_cfg.get("timeout", 60.0))

    predictions: List[str] = []
    references: List[str] = []
    errors = 0

    log(f"[{exp_cfg['name']}] Port-forwarding service and running evaluation")
    with port_forward(namespace, context, service_resource, serving_local, serving_port):
        base_url = f"http://127.0.0.1:{serving_local}"
        for rec in tqdm(records, desc=f"Eval [{exp_cfg['name']}]"):
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": rec["prompt"]}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            pred = fetch_response(base_url, route, payload, headers, timeout=timeout)
            if pred == "":
                errors += 1
            predictions.append(pred)
            references.append(rec["reference"])

    metrics = compute_metrics(references, predictions, skip_bertscore=skip_bertscore)

    return ExperimentResult(
        name=exp_cfg["name"],
        overlay=overlay_key,
        bundle_path=str(bundle_path),
        env=env_data,
        metrics=metrics,
        errors=errors,
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


# ------------- CLI ----------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WildChat quality experiments via k8s-deployed models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "quality_config.example.yaml",
        help="YAML config path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/quality_results.json"),
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
    if not cfg or "cluster" not in cfg or "dataset" not in cfg or "experiments" not in cfg:
        raise SystemExit("Config must include 'cluster', 'dataset', and 'experiments'.")

    cluster = cfg["cluster"]
    ds_cfg = cfg["dataset"]
    base_env = read_env_file(Path(cluster.get("base_env_file", "infrastructure/k8s/aws/serving/model.env")))

    records = load_dataset(
        Path(ds_cfg["path"]),
        prompt_field=ds_cfg.get("prompt_field", "prompt"),
        reference_field=ds_cfg.get("reference_field", "reference"),
        limit=ds_cfg.get("limit"),
    )
    skip_bertscore = cfg.get("skip_bertscore", False)

    results: List[Dict[str, Any]] = []
    applied_bundles: List[str] = []

    for exp in cfg["experiments"]:
        res = run_experiment(
            exp_cfg=exp,
            cluster=cluster,
            base_env=base_env,
            records=records,
            skip_bertscore=skip_bertscore,
        )
        results.append(asdict(res))
        applied_bundles.append(res.bundle_path)
        log(
            f"[{res.name}] BLEU={res.metrics['bleu']:.2f}, "
            f"ROUGE-L_f={res.metrics['rougeL_f']:.4f}, "
            f"BERTScore_F1={res.metrics['bertscore_f1']} "
            f"errors={res.errors}"
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
