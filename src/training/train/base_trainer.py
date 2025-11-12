from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Iterable, Mapping

import os
import time

import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    """
    Abstract trainer for maximum flexibility.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config: Dict[str, Any] = dict(config)

        # Basic training hyperparams (fall back to simple defaults)
        self.max_epochs: int = int(self.config.get("max_epochs", 1))
        self.max_steps: Optional[int] = self.config.get("max_steps", None)
        self.gradient_accumulation_steps: int = int(
            self.config.get("gradient_accumulation_steps", 1)
        )
        self.log_every_n_steps: int = int(self.config.get("log_every_n_steps", 50))
        self.val_every_n_steps: Optional[int] = self.config.get(
            "val_every_n_steps", None
        )

        # Distributed context
        self.rank, self.world_size = self._init_distributed()
        self.device = self._get_device()

        # Core objects (to be built in setup)
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataloader: Optional[DataLoader] = None

        # State
        self.global_step: int = 0
        self.epoch: int = 0
        self._should_stop: bool = False

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def train(self) -> None:
        """
        Main training entrypoint. This is what you call from CLI.
        """
        self.setup()  # builds model, loaders, optimizer, etc.

        self.on_train_start()
        t0 = time.time()

        for epoch in range(self.max_epochs):
            self.epoch = epoch
            self.on_epoch_start(epoch)

            for batch_idx, batch in enumerate(self._iterate(self.train_dataloader)):
                if self._should_stop:
                    break

                self.model.train()
                loss = self._train_batch(batch, batch_idx)

                if self.global_step % self.log_every_n_steps == 0:
                    self.log_metrics({"train_loss": float(loss)}, step=self.global_step)

                if (
                    self.val_dataloader is not None
                    and self.val_every_n_steps is not None
                    and self.global_step > 0
                    and self.global_step % self.val_every_n_steps == 0
                ):
                    self.validate()

                if self.max_steps is not None and self.global_step >= self.max_steps:
                    self._should_stop = True
                    break

            self.on_epoch_end(epoch)

            if self._should_stop:
                break

        total_time = time.time() - t0
        if self.is_main_process:
            logger.info(
                "[trainer] Finished training in {:.1f}s, steps={}",
                total_time,
                self.global_step,
            )

        self.on_train_end()

    # ------------------------------------------------------------------
    # ABSTRACT METHODS: must be implemented in subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def build_model(self) -> nn.Module:
        """
        Construct and return the model (can be already wrapped in DDP/FSDP/DeepSpeed etc.).
        """
        raise NotImplementedError

    @abstractmethod
    def build_train_dataloader(self) -> DataLoader:
        """
        Return DataLoader for training.
        """
        raise NotImplementedError

    @abstractmethod
    def build_val_dataloader(self) -> Optional[DataLoader]:
        """
        Return DataLoader for validation (or None if you don't use validation).
        """
        raise NotImplementedError

    @abstractmethod
    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Construct optimizer for the given model (may be DeepSpeed- or FSDP-aware).
        """
        raise NotImplementedError

    @abstractmethod
    def build_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[Any]:
        """
        Construct LR scheduler (or return None).
        """
        raise NotImplementedError

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Compute the training loss for a single batch.
        Should return a scalar loss tensor on the correct device.

        For DeepSpeed/FSDP/DDP, implement any model-specific logic here.
        """
        raise NotImplementedError

    @abstractmethod
    def optimizer_step(self, loss: torch.Tensor) -> None:
        """
        Perform backward + optimizer step (+ scheduler step, gradient clipping, etc.).

        For plain PyTorch:
            loss.backward()
            optimizer.step()
            scheduler.step()

        For DeepSpeed:
            engine.backward(loss)
            engine.step()

        You get full control here.
        """
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """
        Save model/optimizer/scheduler (and any additional state) to disk.
        """
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """
        Load model/optimizer/scheduler (and any additional state) from disk.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # OPTIONAL HOOKS: override as needed
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """
        Build model, dataloaders, optimizer, scheduler.
        Override if you want full control over init, but for most cases
        the default is enough.
        """
        self.on_setup_start()

        model = self.build_model()
        model.to(self.device)
        self.model = model

        self.train_dataloader = self.build_train_dataloader()
        self.val_dataloader = self.build_val_dataloader()

        self.optimizer = self.build_optimizer(self.model)
        self.scheduler = self.build_scheduler(self.optimizer)

        self.on_setup_end()

    def on_setup_start(self) -> None:
        pass

    def on_setup_end(self) -> None:
        pass

    def on_train_start(self) -> None:
        pass

    def on_train_end(self) -> None:
        pass

    def on_epoch_start(self, epoch: int) -> None:
        pass

    def on_epoch_end(self, epoch: int) -> None:
        pass

    def on_validation_start(self) -> None:
        pass

    def on_validation_end(self) -> None:
        pass

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, float]:
        """
        Optional per-batch validation logic.
        Return a dict of metrics (e.g. {'val_loss': 1.23}).
        """
        return {}

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Override to integrate with Prometheus, WandB, TensorBoard, etc.

        Default: print on main process.
        """
        if not self.is_main_process:
            return
        msg = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info("[step {}] {}", step, msg)

    # ------------------------------------------------------------------
    # INTERNALS
    # ------------------------------------------------------------------

    def _train_batch(self, batch: Any, batch_idx: int) -> float:
        """
        Internal: handles gradient accumulation and calls training_step/optimizer_step.
        """
        loss_accum = 0.0

        # Gradient accumulation: user may also choose to override optimizer_step completely.
        for micro_idx in range(self.gradient_accumulation_steps):
            loss = self.training_step(batch, batch_idx)
            loss = loss / self.gradient_accumulation_steps
            self.optimizer_step(loss)
            loss_accum += float(loss.detach().cpu())

        self.global_step += 1
        return loss_accum

    def validate(self) -> None:
        if self.val_dataloader is None:
            return

        self.on_validation_start()
        self.model.eval()

        aggregated: Dict[str, float] = {}
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self._iterate(self.val_dataloader)):
                metrics = self.validation_step(batch, batch_idx)
                if not metrics:
                    continue
                for k, v in metrics.items():
                    aggregated[k] = aggregated.get(k, 0.0) + float(v)
                num_batches += 1

        if num_batches > 0:
            for k in list(aggregated.keys()):
                aggregated[k] /= num_batches

            self.log_metrics(aggregated, step=self.global_step)

        self.on_validation_end()

    # ------------------------------------------------------------------
    # DISTRIBUTED HELPERS
    # ------------------------------------------------------------------

    def _init_distributed(self) -> tuple[int, int]:
        """
        Default distributed init: read env vars set by torchrun/DeepSpeed.
        Subclass/override if you want to call dist.init_process_group here.
        """
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return rank, world_size

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        return torch.device("cpu")

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @staticmethod
    def _iterate(loader: Optional[Iterable]) -> Iterable:
        if loader is None:
            return []
        return loader
