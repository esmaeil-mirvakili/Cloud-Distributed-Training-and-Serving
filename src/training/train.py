from __future__ import annotations

from dataclasses import asdict
import json
import os
import time
from importlib import import_module
from typing import Type

from loguru import logger

from training.logging_utils import setup_logging
from training.metrics import start_metrics_server
from training.train.config import LoraSettings, TrainConfig

__all__ = ["TrainConfig", "LoraSettings", "run_training"]


def run_training(cfg: TrainConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    trainer_cls = _import_trainer(cfg.trainer_class)
    trainer = trainer_cls(cfg)

    setup_logging(
        cfg.output_dir,
        rank=getattr(trainer, "rank", 0),
        world_size=getattr(trainer, "world_size", 1),
        process="train",
    )
    logger.info("Resolved training config:\n{}", json.dumps(asdict(cfg), indent=2))

    if cfg.metrics_port:
        logger.info("Starting metrics server on port {}", cfg.metrics_port)
        start_metrics_server(cfg.metrics_port)

    t0 = time.time()
    trainer.train()
    total_time = time.time() - t0
    save_path = trainer.export_model()

    if trainer.is_main_process:
        summary = {
            "config": asdict(cfg),
            "global_step": trainer.global_step,
            "total_time_seconds": total_time,
        }
        with open(os.path.join(cfg.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            "Training done in {:.1f}s, steps={}", total_time, trainer.global_step
        )
        if save_path:
            logger.info("Model saved to {}", save_path)


def _import_trainer(path: str) -> Type:
    module_path, _, class_name = path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"trainer_class must be a full module path, got '{path}' instead"
        )
    module = import_module(module_path)
    if not hasattr(module, class_name):
        raise ImportError(f"Module '{module_path}' has no attribute '{class_name}'")
    trainer_cls = getattr(module, class_name)
    return trainer_cls
