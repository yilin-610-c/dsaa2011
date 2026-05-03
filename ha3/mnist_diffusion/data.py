from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def mnist_transform():
    return transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def get_mnist_datasets(data_dir: str | Path = "./data", download: bool = True):
    transform = mnist_transform()
    train = datasets.MNIST(
        root=str(data_dir), train=True, transform=transform, download=download
    )
    test = datasets.MNIST(
        root=str(data_dir), train=False, transform=transform, download=download
    )
    return train, test


def get_mnist_loaders(
    data_dir: str | Path = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    download: bool = True,
):
    train, test = get_mnist_datasets(data_dir=data_dir, download=download)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader

