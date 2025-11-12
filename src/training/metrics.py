import time
import threading
import psutil
from loguru import logger
from prometheus_client import start_http_server, Gauge, Summary

try:
    import pynvml

    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False

training_loss = Gauge("training_loss", "Current training loss")
training_perplexity = Gauge("training_perplexity", "Current training perplexity")
gradient_norm = Gauge("gradient_norm", "Global gradient L2 norm")
global_step = Gauge("training_global_step", "Current global training step")
step_time = Summary("step_time_seconds", "Time per training step (seconds)")
steps_per_second = Gauge("steps_per_second", "Training steps processed per second")
tokens_per_second = Gauge("tokens_per_second", "Tokens processed per second")
gpu_utilization = Gauge("gpu_utilization_percent", "GPU utilization percent")
gpu_memory_used = Gauge("gpu_memory_used_bytes", "GPU memory used in bytes")
cpu_utilization = Gauge("cpu_utilization_percent", "CPU utilization percent")
rss_memory = Gauge("process_rss_memory_bytes", "Process RSS memory in bytes")


def _monitor_resources(interval: float = 5.0):
    gpu_enabled = False
    handle = None

    if _HAS_NVML:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # assumes 1 GPU
            gpu_enabled = True
        except pynvml.NVMLError as e:
            logger.warning(
                "[metrics] NVML not available, disabling GPU metrics: {}", e
            )
            gpu_enabled = False

    proc = psutil.Process()

    while True:
        if gpu_enabled:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_utilization.set(util.gpu)
                gpu_memory_used.set(mem.used)
            except pynvml.NVMLError as e:
                # If something goes wrong later, just stop GPU metrics
                logger.warning("[metrics] NVML error, turning off GPU metrics: {}", e)
                gpu_enabled = False

        cpu_utilization.set(psutil.cpu_percent(interval=None))
        rss_memory.set(proc.memory_info().rss)

        time.sleep(interval)


def start_metrics_server(port: int = 8000):
    start_http_server(port)
    thread = threading.Thread(target=_monitor_resources, daemon=True)
    thread.start()
