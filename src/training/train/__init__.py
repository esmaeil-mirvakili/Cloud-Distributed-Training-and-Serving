from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType
from typing import Any

_TRAIN_IMPL: ModuleType | None = None


def _load_train_module() -> ModuleType:
    global _TRAIN_IMPL
    if _TRAIN_IMPL is not None:
        return _TRAIN_IMPL
    """
    Dynamically load src/training/train.py so that package imports
    (training.train) expose the dataclasses and run_training helper.
    """
    module_name = "training._train_impl"
    root = pathlib.Path(__file__).resolve().parent.parent
    train_path = root / "train.py"
    spec = importlib.util.spec_from_file_location(module_name, train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load training module from {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    _TRAIN_IMPL = module
    return module


def __getattr__(name: str) -> Any:
    module = _load_train_module()
    if hasattr(module, name):
        return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["LoraSettings", "TrainConfig", "run_training"]
