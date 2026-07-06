"""Reproducibility helpers shared by every training and scoring entrypoint."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from the torch initial seed (use as ``worker_init_fn``)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch (CPU and CUDA) and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_global_determinism(seed: int) -> None:
    """Seed all RNGs via :func:`set_seed` and additionally request deterministic torch kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(seed)
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
