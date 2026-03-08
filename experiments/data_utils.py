from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


FASHION_MEAN = 0.2860
FASHION_STD = 0.3530


class FashionParquetDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, augment: bool = False) -> None:
        self.frame = frame.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def _bytes_to_tensor(self, png_bytes: bytes) -> torch.Tensor:
        image = Image.open(BytesIO(png_bytes)).convert("L")
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        if self.augment:
            if np.random.rand() < 0.5:
                image_array = np.fliplr(image_array).copy()
            if np.random.rand() < 0.2:
                shift = np.random.randint(-2, 3)
                image_array = np.roll(image_array, shift=shift, axis=1)
        tensor = torch.from_numpy(image_array).unsqueeze(0)
        tensor = (tensor - FASHION_MEAN) / FASHION_STD
        return tensor

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        image_data = row["image"]
        image_bytes = image_data["bytes"]
        x = self._bytes_to_tensor(image_bytes)
        y = int(row["label"])
        return x, y


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_split(frame: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    indices_train: List[int] = []
    indices_val: List[int] = []
    rng = np.random.default_rng(seed)
    for cls in sorted(frame["label"].unique().tolist()):
        cls_indices = frame.index[frame["label"] == cls].to_numpy()
        rng.shuffle(cls_indices)
        cutoff = int(len(cls_indices) * (1.0 - val_ratio))
        indices_train.extend(cls_indices[:cutoff].tolist())
        indices_val.extend(cls_indices[cutoff:].tolist())
    train_frame = frame.loc[indices_train].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_frame = frame.loc[indices_val].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_frame, val_frame


def load_dataframes(parquet_path: str, val_ratio: float, seed: int) -> Dict[str, pd.DataFrame]:
    frame = pd.read_parquet(parquet_path)
    train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    test_frame = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    train_subframe, val_subframe = stratified_split(train_frame, val_ratio=val_ratio, seed=seed)
    return {"train": train_subframe, "val": val_subframe, "test": test_frame}


def build_dataloaders(
    parquet_path: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    use_augmentation: bool,
) -> Dict[str, DataLoader]:
    frames = load_dataframes(parquet_path, val_ratio=val_ratio, seed=seed)
    train_dataset = FashionParquetDataset(frames["train"], augment=use_augmentation)
    val_dataset = FashionParquetDataset(frames["val"], augment=False)
    test_dataset = FashionParquetDataset(frames["test"], augment=False)

    pin_memory = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    return loaders

