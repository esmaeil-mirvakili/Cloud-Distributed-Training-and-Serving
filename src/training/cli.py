from __future__ import annotations

import json
import os

import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from training.data import preprocess_and_save_shards
from training.formatting import ExampleFormatter
from training.logging_utils import (
    detect_env_rank,
    detect_env_world_size,
    setup_logging,
)
from training.train import LoraSettings, TrainConfig, run_training


@hydra.main(version_base="1.3", config_path="configs", config_name="training")
def main(cfg: DictConfig) -> None:
    if cfg.mode == "preprocess":
        setup_logging(cfg.preprocess.output_dir, process="preprocess", force=True)
        logger.info("\n{}", OmegaConf.to_yaml(cfg))
        formatter: ExampleFormatter = instantiate(cfg.formatter)
        _run_preprocess(cfg, formatter)
    elif cfg.mode == "train":
        rank = detect_env_rank()
        world_size = detect_env_world_size()
        setup_logging(
            cfg.train.output_dir,
            rank=rank,
            world_size=world_size,
            process="train",
            force=True,
        )
        logger.info("\n{}", OmegaConf.to_yaml(cfg))
        _run_train(cfg)
    else:
        raise ValueError(f"Unknown mode={cfg.mode}")


def _run_preprocess(cfg: DictConfig, formatter: ExampleFormatter) -> None:
    train_subset = cfg.preprocess.train.get("subset", None)
    if train_subset is None:
        train_subset = cfg.dataset.get("subset", None)

    val_subset = cfg.preprocess.val.get("subset", None)
    if val_subset is None:
        val_subset = cfg.dataset.get("val_subset", None)

    preprocess_and_save_shards(
        dataset_name=cfg.dataset.name,
        formatter=formatter,
        model_name=cfg.model.name,
        output_dir=cfg.preprocess.output_dir,
        split=cfg.dataset.split,
        subset=train_subset,
        val_subset=val_subset,
        train_max_examples=cfg.preprocess.train.get("max_examples", None),
        val_max_examples=cfg.preprocess.val.get("max_examples", None),
        max_length=cfg.model.max_length,
        num_train_shards=cfg.preprocess.train.num_shards,
        num_val_shards=cfg.preprocess.val.num_shards,
        val_split_ratio=cfg.preprocess.val_split_ratio,
    )


def _run_train(cfg: DictConfig) -> None:
    ds_config = None
    if cfg.train.deepspeed_config_path is not None:
        path = cfg.train.deepspeed_config_path
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path) as f:
            ds_config = json.load(f)

    lora_settings = _build_lora_settings(cfg)
    model_builder = _instantiate_optional(cfg.train.get("model_builder"))
    optimizer_builder = _instantiate_optional(cfg.train.get("optimizer_builder"))
    scheduler_builder = _instantiate_optional(cfg.train.get("scheduler_builder"))
    metrics_logger = _instantiate_optional(cfg.train.get("metrics_logger"))

    train_cfg = TrainConfig(
        model_name=cfg.model.name,
        data_dir=cfg.train.data_dir,
        batch_size=cfg.train.batch_size,
        val_batch_size=cfg.train.get("val_batch_size", None),
        num_epochs=cfg.train.num_epochs,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        output_dir=cfg.train.output_dir,
        use_lora=cfg.train.use_lora,
        lora=lora_settings,
        train_shard_template=cfg.train.train_shard_template,
        val_shard_template=cfg.train.get("val_shard_template", None),
        deepspeed_config=ds_config,
        metrics_port=cfg.train.metrics_port,
        max_steps=cfg.train.max_steps,
        trainer_class=cfg.train.trainer_class,
        model_builder=model_builder,
        optimizer_builder=optimizer_builder,
        scheduler_builder=scheduler_builder,
        metrics_logger=metrics_logger,
    )
    run_training(train_cfg)


def _build_lora_settings(cfg: DictConfig) -> LoraSettings | None:
    if not hasattr(cfg, "lora") or cfg.lora is None:
        if cfg.train.use_lora:
            return LoraSettings()
        return None
    lora_values = OmegaConf.to_container(cfg.lora, resolve=True) or {}
    target_modules = lora_values.get("target_modules")
    if target_modules is not None:
        target_modules = list(target_modules)
    return LoraSettings(
        r=lora_values.get("r", 8),
        alpha=lora_values.get("alpha", 16),
        dropout=lora_values.get("dropout", 0.05),
        target_modules=target_modules,
    )


def _instantiate_optional(node):
    if node is None:
        return None
    return instantiate(node)
