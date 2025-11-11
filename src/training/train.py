from dataclasses import dataclass, asdict
from typing import Optional
import math
import os
import json
import time

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

import deepspeed

from training.data.data import load_sharded_dataset, get_dataloader
from training.model import load_base_model, apply_lora, load_tokenizer
from training.metrics import (
    training_loss,
    training_perplexity,
    gradient_norm as gradient_norm_metric,
    global_step as global_step_metric,
    step_time,
    steps_per_second,
    tokens_per_second,
    start_metrics_server,
)


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
    shard_id: int = 0
    batch_size: int = 4
    num_epochs: int = 1
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    output_dir: str = "./outputs"
    use_lora: bool = False
    lora: Optional[LoraSettings] = None
    deepspeed_config: Optional[str] = None
    metrics_port: int = 8000
    max_steps: Optional[int] = None  # for quick tests


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


def run_training(cfg: TrainConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    start_metrics_server(cfg.metrics_port)

    ds = load_sharded_dataset(cfg.data_dir, cfg.shard_id)
    dataloader = get_dataloader(ds, batch_size=cfg.batch_size)

    model = load_base_model(cfg.model_name, device_map=None)  # let DeepSpeed handle
    if cfg.use_lora:
        lora_cfg = cfg.lora or LoraSettings()
        model = apply_lora(
            model,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=lora_cfg.target_modules,
        )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": cfg.weight_decay,
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

    num_update_steps_per_epoch = len(dataloader)
    t_total = num_update_steps_per_epoch * cfg.num_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.deepspeed_config:
        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=optimizer_grouped_parameters,
            config=cfg.deepspeed_config,
        )
        ds_engine = model
    else:
        model.to(device)
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(cfg.warmup_ratio * t_total),
            num_training_steps=t_total,
        )
        ds_engine = None

    global_step = 0
    model.train()
    t0 = time.time()

    for epoch in range(cfg.num_epochs):
        for batch in dataloader:
            step_start = time.time()
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            token_count = attention_mask.sum().item()
            grad_norm_value = None

            if ds_engine is None:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                grad_norm_value = _compute_grad_norm(model.parameters())
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            else:
                input_ids = input_ids.to(ds_engine.local_rank)
                attention_mask = attention_mask.to(ds_engine.local_rank)
                labels = labels.to(ds_engine.local_rank)

                outputs = ds_engine(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                ds_engine.backward(loss)
                grad_norm_value = _get_deepspeed_grad_norm(ds_engine)
                ds_engine.step()

            step_duration = time.time() - step_start
            step_time.observe(step_duration)
            if step_duration > 0:
                steps_per_second.set(1.0 / step_duration)
                tokens_per_second.set(token_count / step_duration)
            else:
                steps_per_second.set(0.0)
                tokens_per_second.set(0.0)
            loss_value = loss.item()
            perplexity_value = math.exp(min(loss_value, 20))
            training_loss.set(loss_value)
            training_perplexity.set(perplexity_value)
            if grad_norm_value is not None:
                gradient_norm_metric.set(grad_norm_value)
            global_step += 1
            global_step_metric.set(global_step)

            if cfg.max_steps and global_step >= cfg.max_steps:
                break

        if cfg.max_steps and global_step >= cfg.max_steps:
            break

    total_time = time.time() - t0

    # Save model / adapter
    if cfg.use_lora:
        save_path = os.path.join(cfg.output_dir, "lora_adapter")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
    else:
        save_path = os.path.join(cfg.output_dir, "full_finetuned")
        os.makedirs(save_path, exist_ok=True)
        # deepspeed engine requires .module
        target = model.module if hasattr(model, "module") else model
        target.save_pretrained(save_path)

    summary = {
        "config": asdict(cfg),
        "global_step": global_step,
        "total_time_seconds": total_time,
    }
    with open(os.path.join(cfg.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Training done in {total_time:.1f}s, steps={global_step}")
    print(f"Model saved to {save_path}")
