from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, Optional

import deepspeed
import torch
from loguru import logger
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from training.data import get_dataloader, load_sharded_dataset
from training.metrics import (
    gradient_norm as gradient_norm_metric,
    global_step as global_step_metric,
    step_time,
    steps_per_second,
    tokens_per_second,
    training_loss,
    training_perplexity,
)
from training.model import apply_lora, load_base_model
from training.train.base_trainer import BaseTrainer
from training.train.config import TrainConfig


class SFTTrainer(BaseTrainer):
    """
    Supervised fine-tuning trainer that can operate locally or with DeepSpeed ZeRO.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        trainer_config = {
            "max_epochs": cfg.num_epochs,
            "max_steps": cfg.max_steps,
            "log_every_n_steps": cfg.log_every_n_steps,
        }
        super().__init__(trainer_config)
        self.cfg = cfg
        self.should_use_deepspeed = self.world_size > 1
        self.ds_config = cfg.deepspeed_config
        self.ds_engine: Optional[deepspeed.DeepSpeedEngine] = None
        self._shard_name: Optional[str] = None
        self._step_start_time: Optional[float] = None
        self._last_step_duration: Optional[float] = None
        self._current_token_count: int = 0
        self._last_loss_value: Optional[float] = None
        self._last_grad_norm: Optional[float] = None

    # ------------------------------------------------------------------
    # BaseTrainer API
    # ------------------------------------------------------------------

    def setup(self) -> None:
        self.on_setup_start()

        model = self.build_model()
        self.train_dataloader = self.build_train_dataloader()
        self.val_dataloader = self.build_val_dataloader()

        if self.should_use_deepspeed:
            if self.ds_config is None:
                raise ValueError(
                    "Distributed training requires train.deepspeed_config_path to be set"
                )
            optimizer_params = self._build_optimizer_param_groups(model)
            engine, optimizer, _, scheduler = deepspeed.initialize(
                model=model,
                model_parameters=optimizer_params,
                config=self.ds_config,
            )
            self.model = engine
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.ds_engine = engine
        else:
            model.to(self.device)
            self.model = model
            self.optimizer = self.build_optimizer(model)
            self.scheduler = self.build_scheduler(self.optimizer)

        self.on_setup_end()

    def build_model(self) -> torch.nn.Module:
        if self.cfg.model_builder is not None:
            return self.cfg.model_builder(self.cfg)
        return self._default_build_model()

    def build_train_dataloader(self):
        shard_path, shard_name = self._resolve_shard(
            self.cfg.train_shard_template, "train"
        )
        if self.is_main_process:
            logger.info(
                "[rank {}/{}] Loading shard {} from {}",
                self.rank,
                self.world_size,
                shard_name,
                self.cfg.data_dir,
            )
        dataset = load_sharded_dataset(shard_path)
        return get_dataloader(dataset, batch_size=self.cfg.batch_size)

    def build_val_dataloader(self):
        template = self.cfg.val_shard_template
        if not template:
            return None

        single_shard = ("{rank" not in template) and ("{id" not in template)
        if single_shard and self.rank != 0:
            logger.info(
                "Skipping val dataloader on rank %s because template '%s' has no rank placeholder",
                self.rank,
                template,
            )
            return None

        shard_path, shard_name = self._resolve_shard(template, "val")
        logger.info(
            "[rank {}/{}] Loading val shard {} from {}",
            self.rank,
            self.world_size,
            shard_name,
            self.cfg.data_dir,
        )
        dataset = load_sharded_dataset(shard_path)
        batch_size = self.cfg.val_batch_size or self.cfg.batch_size
        return get_dataloader(dataset, batch_size=batch_size, shuffle=False)

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        if self.cfg.optimizer_builder is not None:
            return self.cfg.optimizer_builder(model, self.cfg)
        param_groups = self._build_optimizer_param_groups(model)
        return AdamW(param_groups, lr=self.cfg.lr)

    def build_scheduler(self, optimizer: torch.optim.Optimizer):
        if self.cfg.scheduler_builder is not None:
            total_steps = self._total_training_steps()
            return self.cfg.scheduler_builder(optimizer, total_steps, self.cfg)

        total_steps = self._total_training_steps()
        if total_steps == 0:
            return None
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.cfg.warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        device = self._target_device()
        self._step_start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        self._current_token_count = attention_mask.sum().item()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        self._last_loss_value = float(loss.detach().cpu())
        return loss

    def optimizer_step(self, loss: torch.Tensor) -> None:
        if self.should_use_deepspeed:
            assert self.ds_engine is not None
            self.ds_engine.backward(loss)
            self._last_grad_norm = _get_deepspeed_grad_norm(self.ds_engine)
            self.ds_engine.step()
        else:
            loss.backward()
            assert self.model is not None
            self._last_grad_norm = _compute_grad_norm(self.model.parameters())
            assert self.optimizer is not None
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()

        if self._step_start_time is not None:
            self._last_step_duration = time.time() - self._step_start_time
        else:
            self._last_step_duration = None

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        target = self.model
        if hasattr(target, "module"):
            target = target.module
        target.save_pretrained(path)

    def load_checkpoint(self, path: str) -> None:
        raise NotImplementedError("Checkpoint loading not implemented yet.")

    def _train_batch(self, batch: Any, batch_idx: int) -> float:
        loss_value = super()._train_batch(batch, batch_idx)
        self._publish_metrics()
        return loss_value

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def export_model(self) -> Optional[str]:
        if not self.is_main_process:
            return None
        if self.cfg.use_lora:
            save_path = os.path.join(self.cfg.output_dir, "lora_adapter")
        else:
            save_path = os.path.join(self.cfg.output_dir, "full_finetuned")
        self.save_checkpoint(save_path)
        return save_path

    def _default_build_model(self) -> torch.nn.Module:
        model = load_base_model(self.cfg.model_name, device_map=None)
        if self.cfg.use_lora:
            if self.cfg.lora is None:
                raise ValueError("cfg.lora must be provided when use_lora=True")
            lora_cfg = self.cfg.lora
            model = apply_lora(
                model,
                r=lora_cfg.r,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                target_modules=lora_cfg.target_modules,
            )
        return model

    def _publish_metrics(self) -> None:
        if not self.is_main_process or self._last_loss_value is None:
            return

        duration = self._last_step_duration or 0.0
        loss_value = self._last_loss_value
        perplexity_value = math.exp(min(loss_value, 20))
        grad_norm = self._last_grad_norm
        tokens = self._current_token_count
        steps_ps = 1.0 / duration if duration > 0 else 0.0
        tokens_ps = tokens / duration if duration > 0 else 0.0

        training_loss.set(loss_value)
        training_perplexity.set(perplexity_value)
        if grad_norm is not None:
            gradient_norm_metric.set(grad_norm)
        step_time.observe(duration)
        steps_per_second.set(steps_ps)
        tokens_per_second.set(tokens_ps)
        global_step_metric.set(self.global_step)

        if self.cfg.metrics_logger is not None:
            payload = {
                "loss": loss_value,
                "perplexity": perplexity_value,
                "grad_norm": grad_norm or 0.0,
                "step_duration": duration,
                "steps_per_second": steps_ps,
                "tokens_per_second": tokens_ps,
            }
            self.cfg.metrics_logger(payload, self.global_step)

    def _build_optimizer_param_groups(self, model: torch.nn.Module):
        no_decay = ["bias", "LayerNorm.weight"]
        return [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

    def _target_device(self) -> torch.device:
        if self.should_use_deepspeed and self.ds_engine is not None:
            return torch.device("cuda", self.ds_engine.local_rank)
        return self.device

    def _resolve_shard(self, template: str, purpose: str) -> tuple[str, str]:
        try:
            shard_name = template.format(
                id=self.rank, rank=self.rank, world_size=self.world_size
            )
        except KeyError as err:
            raise ValueError(
                f"{purpose} shard template must contain '{{id}}' or '{{rank}}' placeholder."
            ) from err

        shard_path = os.path.join(self.cfg.data_dir, shard_name)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(
                f"{purpose.capitalize()} shard '{shard_name}' not found in {self.cfg.data_dir}"
            )
        self._shard_name = shard_name
        return shard_path, shard_name

    def _total_training_steps(self) -> int:
        if self.train_dataloader is None:
            return 0
        return len(self.train_dataloader) * self.max_epochs


def _compute_grad_norm(parameters) -> Optional[float]:
    total_norm = 0.0
    has_grad = False
    for param in parameters:
        if param.grad is None:
            continue
        has_grad = True
        param_norm = param.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    if not has_grad:
        return None
    return math.sqrt(total_norm)


def _get_deepspeed_grad_norm(ds_engine) -> Optional[float]:
    if ds_engine is None or not hasattr(ds_engine, "get_global_grad_norm"):
        return None
    try:
        value = ds_engine.get_global_grad_norm()
    except TypeError:
        return None
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    return float(value)
