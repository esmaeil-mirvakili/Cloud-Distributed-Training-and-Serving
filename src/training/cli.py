from __future__ import annotations

import json
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from training.data.data import preprocess_and_save_shards
from training.data.formatting import ExampleFormatter
from training.train import (
    LoraSettings,
    TrainConfig,
    run_training,
)  # assuming you already have these


@hydra.main(version_base="1.3", config_path="configs", config_name="training")
def main(cfg: DictConfig) -> None:
    # Optional: quick dump to see what Hydra resolved
    print(OmegaConf.to_yaml(cfg))

    # Instantiate formatter from YAML (_target_ + params)
    formatter: ExampleFormatter = instantiate(cfg.formatter)

    if cfg.mode == "preprocess":
        _run_preprocess(cfg, formatter)
    elif cfg.mode == "train":
        _run_train(cfg)
    else:
        raise ValueError(f"Unknown mode={cfg.mode}")


def _run_preprocess(cfg: DictConfig, formatter: ExampleFormatter) -> None:
    preprocess_and_save_shards(
        dataset_name=cfg.dataset.name,
        formatter=formatter,
        model_name=cfg.model.name,
        output_dir=cfg.preprocess.output_dir,
        split=cfg.dataset.split,
        subset=cfg.dataset.subset,
        max_length=cfg.model.max_length,
        num_shards=cfg.preprocess.num_shards,
    )


def _run_train(cfg: DictConfig) -> None:
    # Load deepspeed config (if any)
    ds_config = None
    if cfg.train.deepspeed_config_path is not None:
        path = cfg.train.deepspeed_config_path
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path) as f:
            ds_config = json.load(f)

    lora_settings = _build_lora_settings(cfg)

    train_cfg = TrainConfig(
        model_name=cfg.model.name,
        data_dir=cfg.train.data_dir,
        shard_id=cfg.train.shard_id,
        batch_size=cfg.train.batch_size,
        num_epochs=cfg.train.num_epochs,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        output_dir=cfg.train.output_dir,
        use_lora=cfg.train.use_lora,
        lora=lora_settings,
        deepspeed_config=ds_config,
        metrics_port=cfg.train.metrics_port,
        max_steps=cfg.train.max_steps,
    )
    run_training(train_cfg)


def _build_lora_settings(cfg: DictConfig) -> LoraSettings | None:
    if not hasattr(cfg, "lora") or cfg.lora is None:
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
