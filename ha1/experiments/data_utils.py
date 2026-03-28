from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


FASHION_MEAN = 0.2860
FASHION_STD = 0.3530


class FashionTensorDataset(Dataset):
    def __init__(self, images_uint8: torch.Tensor, labels: torch.Tensor, augment: bool = False) -> None:
        self.images_uint8 = images_uint8
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def _image_to_tensor(self, image_uint8: torch.Tensor) -> torch.Tensor:
        image_array = image_uint8.float() / 255.0
        if self.augment:
            if np.random.rand() < 0.5:
                image_array = torch.flip(image_array, dims=[1])
            if np.random.rand() < 0.2:
                shift = np.random.randint(-2, 3)
                image_array = torch.roll(image_array, shifts=shift, dims=1)
        tensor = image_array.unsqueeze(0)
        tensor = (tensor - FASHION_MEAN) / FASHION_STD
        return tensor

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        x = self._image_to_tensor(self.images_uint8[index])
        y = int(self.labels[index].item())
        return x, y


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_split_indices(labels: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    indices_train: List[int] = []
    indices_val: List[int] = []
    rng = np.random.default_rng(seed)
    for cls in sorted(np.unique(labels).tolist()):
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        cutoff = int(len(cls_indices) * (1.0 - val_ratio))
        indices_train.extend(cls_indices[:cutoff].tolist())
        indices_val.extend(cls_indices[cutoff:].tolist())
    train_idx = np.array(indices_train, dtype=np.int64)
    val_idx = np.array(indices_val, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _decode_png_bytes(png_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(png_bytes)).convert("L")
    image_array = np.asarray(image, dtype=np.uint8)
    return image_array


def build_or_load_tensor_cache(parquet_path: str, cache_path: str) -> Dict[str, torch.Tensor]:
    cache_file = Path(cache_path)
    if cache_file.exists():
        return torch.load(cache_file, map_location="cpu")

    frame = pd.read_parquet(parquet_path)
    images: List[np.ndarray] = []
    labels: List[int] = []
    split_flags: List[int] = []
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="Building tensor cache"):
        images.append(_decode_png_bytes(row["image"]["bytes"]))
        labels.append(int(row["label"]))
        split_flags.append(0 if row["split"] == "train" else 1)

    cache = {
        "images_uint8": torch.from_numpy(np.stack(images, axis=0)),
        "labels": torch.tensor(labels, dtype=torch.long),
        "split_flags": torch.tensor(split_flags, dtype=torch.int8),
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_file)
    return cache


def build_dataloaders(
    parquet_path: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    use_augmentation: bool,
    cache_path: str,
) -> Dict[str, DataLoader]:
    cache = build_or_load_tensor_cache(parquet_path=parquet_path, cache_path=cache_path)
    images_uint8 = cache["images_uint8"]
    labels = cache["labels"]
    split_flags = cache["split_flags"]

    train_pool_idx = torch.nonzero(split_flags == 0, as_tuple=False).squeeze(1).numpy()
    test_idx = torch.nonzero(split_flags == 1, as_tuple=False).squeeze(1)
    train_pool_labels = labels[torch.from_numpy(train_pool_idx)].numpy()
    train_sub_idx_local, val_sub_idx_local = stratified_split_indices(train_pool_labels, val_ratio=val_ratio, seed=seed)
    train_idx = torch.from_numpy(train_pool_idx[train_sub_idx_local])
    val_idx = torch.from_numpy(train_pool_idx[val_sub_idx_local])

    train_dataset = FashionTensorDataset(images_uint8[train_idx], labels[train_idx], augment=use_augmentation)
    val_dataset = FashionTensorDataset(images_uint8[val_idx], labels[val_idx], augment=False)
    test_dataset = FashionTensorDataset(images_uint8[test_idx], labels[test_idx], augment=False)

    pin_memory = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        ),
    }
    return loaders

