from __future__ import annotations

import os
import sys
from typing import Optional

from loguru import logger

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level:<8}</level> | "
    "rank {extra[rank]:>2}/{extra[world_size]:<2} | "
    "{extra[process]} | "
    "<level>{message}</level>"
)
_CONFIGURED = False


def setup_logging(
    log_root: str,
    *,
    rank: int = 0,
    world_size: int = 1,
    process: str = "train",
    force: bool = False,
    console_level: str = "INFO",
) -> None:
    """
    Configure loguru with both stdout and per-rank file sinks.
    Calling this multiple times will reconfigure the sinks when force=True,
    otherwise it simply refreshes contextual metadata (rank/world/process).
    """
    global _CONFIGURED
    os.makedirs(log_root, exist_ok=True)
    log_dir = os.path.join(log_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger.configure(extra={"rank": rank, "world_size": world_size, "process": process})

    if _CONFIGURED and not force:
        return

    logger.remove()
    logger.configure(extra={"rank": rank, "world_size": world_size, "process": process})
    logger.add(sys.stdout, format=_LOG_FORMAT, level=console_level, enqueue=True)
    log_file = os.path.join(log_dir, f"{process}_rank_{rank}.log")
    logger.add(
        log_file,
        format=_LOG_FORMAT,
        level="DEBUG",
        enqueue=True,
        rotation="250 MB",
        retention=10,
    )
    _CONFIGURED = True


def detect_env_rank(default: int = 0) -> int:
    for key in ("RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return default


def detect_env_world_size(default: int = 1) -> int:
    value = os.environ.get("WORLD_SIZE")
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default
