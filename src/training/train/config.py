from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


MetricsLoggerFn = Callable[[Dict[str, float], int], None]
ModelBuilderFn = Callable[["TrainConfig"], Any]
OptimizerBuilderFn = Callable[[Any, "TrainConfig"], Any]
SchedulerBuilderFn = Callable[[Any, int, "TrainConfig"], Optional[Any]]


@dataclass
class LoraSettings:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: Optional[list[str]] = None


@dataclass
class TrainConfig:
    model_name: str
    data_dir: str
    trainer_class: str = "training.train.sft_trainer.SFTTrainer"
    batch_size: int = 4
    val_batch_size: Optional[int] = None
    num_epochs: int = 1
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    output_dir: str = "./outputs"
    use_lora: bool = False
    lora: Optional[LoraSettings] = None
    train_shard_template: str = "shard_{rank}"
    val_shard_template: Optional[str] = None
    deepspeed_config: Optional[Dict[str, Any]] = None
    metrics_port: int = 8000
    max_steps: Optional[int] = None  # for quick tests
    model_builder: Optional[ModelBuilderFn] = None
    optimizer_builder: Optional[OptimizerBuilderFn] = None
    scheduler_builder: Optional[SchedulerBuilderFn] = None
    metrics_logger: Optional[MetricsLoggerFn] = None
    log_every_n_steps: Optional[int] = None
