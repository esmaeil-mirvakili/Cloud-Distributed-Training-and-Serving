import asyncio
import json
import os
import threading
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import Response, StreamingResponse

# Optional local backend
try:
    from llama_cpp import Llama  # type: ignore
except Exception:
    Llama = None  # type: ignore

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

try:
    import pynvml  # type: ignore
except Exception:
    pynvml = None

app = FastAPI(title="llama-wrapper")

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
# Backends:
# - "python": use llama-cpp-python bindings in-process (your original behavior)
# - "server": proxy requests to standalone llama.cpp llama-server
BACKEND = os.getenv("LLAMA_BACKEND", "python").strip().lower()
UPSTREAM_BASE_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
UPSTREAM_TIMEOUT_S = float(os.getenv("LLAMA_SERVER_TIMEOUT", "600"))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "/models/model.gguf")
N_THREADS = _env_int("LLAMA_THREADS", 4)
N_CTX = _env_int("LLAMA_CONTEXT", 2048)
N_GPU_LAYERS = _env_int("LLAMA_GPU_LAYERS", 0)
N_BATCH = _env_int("LLAMA_BATCH", 128)
MODEL_NAME = os.getenv("HF_MODEL_REPO") or os.getenv("LLAMA_MODEL_NAME") or os.path.basename(MODEL_PATH) or "llama-model"

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
REQ_COUNTER = Counter("llama_requests_total", "Total llama requests", ["route", "status"])
LATENCY = Histogram(
    "llama_request_latency_seconds",
    "Request latency seconds",
    ["route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 250, 500, float("inf")),
)
BYTES_IN = Counter("llama_request_bytes_total", "Bytes received", ["route"])
BYTES_OUT = Counter("llama_response_bytes_total", "Bytes sent", ["route"])
TOKENS_PROMPT = Counter("llama_prompt_tokens_total", "Prompt tokens", ["route"])
TOKENS_GEN = Counter("llama_generated_tokens_total", "Generated tokens", ["route"])

# LLM-friendly metrics (TTFT vs total, plus unified token counter and inflight gauge)
TTFT = Histogram(
    "llama_request_ttft_seconds",
    "Time to first token/chunk (seconds). For non-streaming responses this equals total latency.",
    ["route"],
    buckets=(
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.35,
        0.5,
        0.75,
        1,
        1.5,
        2,
        3,
        5,
        7,
        10,
        15,
        20,
        30,
        60,
        100,
        200,
        500,
        float("inf"),
    ),
)
TOTAL_LATENCY = Histogram(
    "llama_request_total_seconds",
    "Total request latency until completion (seconds)",
    ["route"],
    buckets=(0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, float("inf")),
)
TOKENS_TOTAL = Counter(
    "llama_tokens_total",
    "Total tokens processed (prompt + completion)",
    ["route", "type"],  # type: prompt|completion
)
INFLIGHT = Gauge("llama_inflight_requests", "In-flight requests", ["route"])

# Host metrics (best-effort; for real node/GPU monitoring prefer node_exporter + dcgm_exporter)
GPU_MEM_TOTAL = Gauge("llama_gpu_memory_total_bytes", "GPU memory total (bytes)", ["index"])
GPU_MEM_USED = Gauge("llama_gpu_memory_used_bytes", "GPU memory used (bytes)", ["index"])
GPU_MEM_FREE = Gauge("llama_gpu_memory_free_bytes", "GPU memory free (bytes)", ["index"])
GPU_UTIL = Gauge("llama_gpu_utilization_percent", "GPU utilization percent", ["index"])
GPU_TEMP = Gauge("llama_gpu_temperature_celsius", "GPU temperature (C)", ["index"])
CPU_UTIL = Gauge("llama_cpu_utilization_percent", "Process CPU utilization percent", ["pid"])
MEM_RSS = Gauge("llama_memory_rss_bytes", "Process resident set size in bytes", ["pid"])

_gpu_handles: List[Any] = []


def _init_gpu_handles() -> None:
    if not pynvml:
        return
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            _gpu_handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
    except Exception:
        _gpu_handles.clear()


def _ensure_gpu_handles() -> None:
    # Try to lazily (re)initialize GPU handles in case NVML wasn't ready at import time.
    if _gpu_handles:
        return
    _init_gpu_handles()


def _refresh_gpu_metrics() -> None:
    _ensure_gpu_handles()
    if not _gpu_handles:
        return
    for idx, handle in enumerate(_gpu_handles):
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            GPU_MEM_TOTAL.labels(index=str(idx)).set(mem.total)
            GPU_MEM_USED.labels(index=str(idx)).set(mem.used)
            GPU_MEM_FREE.labels(index=str(idx)).set(mem.free)
            GPU_UTIL.labels(index=str(idx)).set(util.gpu)
            GPU_TEMP.labels(index=str(idx)).set(temp)
        except Exception:
            continue


def _refresh_process_metrics() -> None:
    if not psutil:
        return
    try:
        # System CPU percent (within container namespace) and total RSS across container processes.
        cpu_pct = psutil.cpu_percent(interval=0.05)
        total_rss = 0
        for p in psutil.process_iter(attrs=["memory_info"]):
            try:
                mi = p.info.get("memory_info")
                if mi:
                    total_rss += mi.rss
            except Exception:
                continue
        CPU_UTIL.labels(pid="container").set(cpu_pct)
        MEM_RSS.labels(pid="container").set(total_rss)
    except Exception:
        pass


def _touch_metrics() -> None:
    REQ_COUNTER.labels(route="completion", status="0").inc(0)
    REQ_COUNTER.labels(route="chat_completions", status="0").inc(0)
    LATENCY.labels(route="completion").observe(0)
    LATENCY.labels(route="chat_completions").observe(0)
    BYTES_IN.labels(route="completion").inc(0)
    BYTES_IN.labels(route="chat_completions").inc(0)
    BYTES_OUT.labels(route="completion").inc(0)
    BYTES_OUT.labels(route="chat_completions").inc(0)
    TOKENS_PROMPT.labels(route="completion").inc(0)
    TOKENS_PROMPT.labels(route="chat_completions").inc(0)
    TOKENS_GEN.labels(route="completion").inc(0)
    TOKENS_GEN.labels(route="chat_completions").inc(0)
    if pynvml:
        _refresh_gpu_metrics()
    # initialize CPU/mem gauges with the container label
    if psutil:
        CPU_UTIL.labels(pid="container").set(0)
        MEM_RSS.labels(pid="container").set(0)


_touch_metrics()

# ------------------------------------------------------------
# Token accounting helpers
# ------------------------------------------------------------

def _observe_usage(route: str, usage: Optional[Dict[str, Any]]) -> None:
    if not usage:
        return
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is not None:
        n = float(prompt_tokens)
        TOKENS_PROMPT.labels(route=route).inc(n)
        TOKENS_TOTAL.labels(route=route, type="prompt").inc(n)
    if completion_tokens is not None:
        n = float(completion_tokens)
        TOKENS_GEN.labels(route=route).inc(n)
        TOKENS_TOTAL.labels(route=route, type="completion").inc(n)


def _json_bytes(obj: Any) -> int:
    try:
        return len(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    except Exception:
        return len(str(obj).encode("utf-8"))


# ------------------------------------------------------------
# Backend: in-process llama-cpp-python
# ------------------------------------------------------------
_llm_lock = threading.Lock()
_llm: Optional[Any] = None


def _get_llm() -> Any:
    global _llm
    if Llama is None:
        raise RuntimeError("llama_cpp is unavailable; set LLAMA_BACKEND=server or install llama-cpp-python")
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                gpu_layers = N_GPU_LAYERS
                if gpu_layers < 0:
                    # Use a large number to offload as many layers as possible when "-1" is provided.
                    gpu_layers = 9999
                _llm = Llama(
                    model_path=MODEL_PATH,
                    n_threads=N_THREADS,
                    n_ctx=N_CTX,
                    n_gpu_layers=gpu_layers,
                    n_batch=N_BATCH,
                    verbose=False,
                )
    return _llm


def _build_chat_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]


async def _handle_completion_python(route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    # llama_cpp calls are CPU/GPU-bound; run them off the event loop.
    llm = _get_llm()
    start = time.perf_counter()

    def _call() -> Dict[str, Any]:
        return llm.create_completion(
            prompt=payload.get("prompt", ""),
            max_tokens=payload.get("n_predict", payload.get("max_tokens", 128)),
            temperature=payload.get("temperature", 0.7),
            top_p=payload.get("top_p", 0.95),
        )

    try:
        result = await asyncio.to_thread(_call)
        result["model"] = MODEL_NAME
        usage = result.get("usage") or {}
        _observe_usage(route, usage)
        return result
    finally:
        elapsed = time.perf_counter() - start
        LATENCY.labels(route=route).observe(elapsed)
        TOTAL_LATENCY.labels(route=route).observe(elapsed)


async def _handle_chat_python(route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    llm = _get_llm()
    start = time.perf_counter()

    def _call() -> Dict[str, Any]:
        messages = _build_chat_messages(payload.get("messages", []))
        return llm.create_chat_completion(
            messages=messages,
            max_tokens=payload.get("max_tokens", 128),
            temperature=payload.get("temperature", 0.7),
            top_p=payload.get("top_p", 0.95),
            stream=False,
        )

    try:
        result = await asyncio.to_thread(_call)
        result["model"] = MODEL_NAME
        usage = result.get("usage") or {}
        _observe_usage(route, usage)
        return result
    finally:
        elapsed = time.perf_counter() - start
        LATENCY.labels(route=route).observe(elapsed)
        TOTAL_LATENCY.labels(route=route).observe(elapsed)


# ------------------------------------------------------------
# Backend: proxy to llama.cpp llama-server
# ------------------------------------------------------------

async def _stream_upstream(route: str, method: str, url: str, payload: Dict[str, Any]) -> AsyncIterator[bytes]:
    start = time.perf_counter()
    first = True
    status_code: Optional[int] = None
    out_bytes = 0
    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_S) as client:
            async with client.stream(method, url, json=payload) as r:
                status_code = r.status_code
                async for chunk in r.aiter_bytes():
                    out_bytes += len(chunk)
                    if first:
                        first = False
                        TTFT.labels(route=route).observe(time.perf_counter() - start)
                    yield chunk
    finally:
        elapsed = time.perf_counter() - start
        LATENCY.labels(route=route).observe(elapsed)
        TOTAL_LATENCY.labels(route=route).observe(elapsed)
        BYTES_OUT.labels(route=route).inc(out_bytes)
        # For streaming, we don't reliably get OpenAI "usage" without buffering. Keep token counters for non-stream.
        if status_code is None:
            REQ_COUNTER.labels(route=route, status="0").inc()
        else:
            REQ_COUNTER.labels(route=route, status=str(status_code)).inc()


async def _proxy_json(route: str, method: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()
    url = f"{UPSTREAM_BASE_URL}{path}"
    status_code = "0"
    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_S) as client:
            r = await client.request(method, url, json=payload)
            status_code = str(r.status_code)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                data["model"] = MODEL_NAME
            usage = (data or {}).get("usage") or {}
            _observe_usage(route, usage)
            BYTES_OUT.labels(route=route).inc(len(r.content))
            return data
    finally:
        elapsed = time.perf_counter() - start
        LATENCY.labels(route=route).observe(elapsed)
        TOTAL_LATENCY.labels(route=route).observe(elapsed)
        REQ_COUNTER.labels(route=route, status=status_code).inc()


async def _proxy_json_streaming_ttft(route: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stream from upstream to capture TTFT, but buffer and return full JSON."""
    start = time.perf_counter()
    url = f"{UPSTREAM_BASE_URL}{path}"
    status_code = "0"
    chunks: List[bytes] = []
    first = True
    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_S) as client:
            async with client.stream("POST", url, json=payload) as r:
                status_code = str(r.status_code)
                async for chunk in r.aiter_bytes():
                    chunks.append(chunk)
                    if first:
                        first = False
                        TTFT.labels(route=route).observe(time.perf_counter() - start)
                r.raise_for_status()
        content = b"".join(chunks)
        events: List[Dict[str, Any]] = []
        if content:
            for raw in content.splitlines():
                line = raw.strip()
                if not line or line == b"data: [DONE]":
                    continue
                if line.startswith(b"data:"):
                    line = line[len(b"data:"):].strip()
                try:
                    evt = json.loads(line.decode("utf-8"))
                    events.append(evt)
                except Exception:
                    continue
        data: Dict[str, Any] = {}
        if events:
            last = events[-1]
            text_parts: List[str] = []
            finish_reason = last.get("choices", [{}])[0].get("finish_reason")
            for evt in events:
                delta = (evt.get("choices") or [{}])[0].get("delta") or {}
                part = delta.get("content")
                if part:
                    text_parts.append(str(part))
            data = {
                "id": last.get("id"),
                "object": last.get("object", "chat.completion"),
                "created": last.get("created"),
                "model": MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "".join(text_parts)},
                        "finish_reason": finish_reason,
                    }
                ],
            }
            usage = last.get("usage") or {}
            _observe_usage(route, usage)
        else:
            # Fallback to direct JSON if no events parsed
            try:
                data = json.loads(content.decode("utf-8")) if content else {}
                if isinstance(data, dict):
                    data["model"] = MODEL_NAME
                usage = (data or {}).get("usage") or {}
                _observe_usage(route, usage)
            except Exception:
                data = {}
        return data
    finally:
        elapsed = time.perf_counter() - start
        LATENCY.labels(route=route).observe(elapsed)
        TOTAL_LATENCY.labels(route=route).observe(elapsed)
        BYTES_OUT.labels(route=route).inc(sum(len(c) for c in chunks))
        REQ_COUNTER.labels(route=route, status=status_code).inc()


def _upstream_streaming_media_type(path: str) -> str:
    # llama-server uses SSE for OpenAI streaming; for /completion it may also stream JSON chunks.
    if path.startswith("/v1/"):
        return "text/event-stream"
    return "application/octet-stream"


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------


@app.post("/completion")
async def completion(request: Request):
    route = "completion"
    raw = await request.body()
    BYTES_IN.labels(route=route).inc(len(raw))
    body: Dict[str, Any] = json.loads(raw) if raw else {}

    INFLIGHT.labels(route=route).inc()
    try:
        if BACKEND == "server":
            # Stream upstream to capture TTFT, but return full JSON to the client.
            body["stream"] = True
            result = await _proxy_json_streaming_ttft(route, "/completion", body)
            return result

        # python backend
        result = await _handle_completion_python(route, body)
        REQ_COUNTER.labels(route=route, status="200").inc()
        BYTES_OUT.labels(route=route).inc(_json_bytes(result))
        return result
    except Exception:
        if BACKEND != "server":
            REQ_COUNTER.labels(route=route, status="500").inc()
        raise
    finally:
        INFLIGHT.labels(route=route).dec()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    route = "chat_completions"
    raw = await request.body()
    BYTES_IN.labels(route=route).inc(len(raw))
    body: Dict[str, Any] = json.loads(raw) if raw else {}

    INFLIGHT.labels(route=route).inc()
    try:
        if BACKEND == "server":
            body["stream"] = True
            result = await _proxy_json_streaming_ttft(route, "/v1/chat/completions", body)
            return result

        result = await _handle_chat_python(route, body)
        REQ_COUNTER.labels(route=route, status="200").inc()
        BYTES_OUT.labels(route=route).inc(_json_bytes(result))
        return result
    except Exception:
        if BACKEND != "server":
            REQ_COUNTER.labels(route=route, status="500").inc()
        raise
    finally:
        INFLIGHT.labels(route=route).dec()


@app.get("/metrics")
async def metrics():
    _refresh_gpu_metrics()
    _refresh_process_metrics()
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/healthz")
async def healthz():
    # Pod can be ready even if model is still loading or upstream isn't warmed up.
    return {"status": "ok", "backend": BACKEND}


@app.get("/upstream/metrics")
async def upstream_metrics():
    # Convenience endpoint: proxy standalone llama-server metrics if enabled there.
    # Scraping upstream directly is usually better.
    if BACKEND != "server":
        return Response(content="# upstream metrics only available when LLAMA_BACKEND=server\n", media_type="text/plain")
    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_S) as client:
        r = await client.get(f"{UPSTREAM_BASE_URL}/metrics")
        return Response(content=r.content, media_type=r.headers.get("content-type", "text/plain"))


_init_gpu_handles()
