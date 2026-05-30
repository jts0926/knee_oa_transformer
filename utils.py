import random
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None


def require_torch():
    if torch is None:
        raise ImportError("PyTorch is required for training/evaluation. Install project requirements first.")
    return torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str):
    torch = require_torch()
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def move_batch_to_device(batch: Dict[str, object], device) -> Dict[str, object]:
    non_blocking = device.type == "cuda"
    return {
        **batch,
        "images": {key: value.to(device, non_blocking=non_blocking) for key, value in batch["images"].items()},
        "incidence_label": batch["incidence_label"].to(device, non_blocking=non_blocking),
        "incidence_mask": batch["incidence_mask"].to(device, non_blocking=non_blocking),
        "progression_label": batch["progression_label"].to(device, non_blocking=non_blocking),
        "progression_mask": batch["progression_mask"].to(device, non_blocking=non_blocking),
    }


def safe_path_exists(value: object) -> bool:
    if pd.isna(value):
        return False
    path = str(value).strip()
    return bool(path) and Path(path).expanduser().exists()


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max"):
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.patience = patience
        self.mode = mode
        self.best: Optional[float] = None
        self.num_bad_epochs = 0

    def step(self, value: float) -> bool:
        improved = False
        if self.best is None:
            improved = True
        elif self.mode == "max" and value > self.best:
            improved = True
        elif self.mode == "min" and value < self.best:
            improved = True

        if improved:
            self.best = value
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience


def save_metrics_csv(metrics: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(path, index=False)
