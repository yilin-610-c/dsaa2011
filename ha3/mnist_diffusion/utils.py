import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torchvision.utils import save_image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def append_csv_row(path: str | Path, row: Dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: str | Path, rows: List[Dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cycle(loader: Iterable):
    while True:
        for batch in loader:
            yield batch


def save_grid(images: torch.Tensor, path: str | Path, nrow: int = 8) -> None:
    """Save images expected in [-1, 1]."""
    path = Path(path)
    ensure_dir(path.parent)
    images = images.detach().cpu().clamp(-1, 1)
    images = (images + 1.0) / 2.0
    save_image(images, path, nrow=nrow, padding=2)


class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start


def set_matplotlib_cache() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

