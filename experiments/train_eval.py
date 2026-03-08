import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from experiments.data_utils import build_dataloaders, set_seed
from models import build_model, count_parameters


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return correct / labels.size(0)


def run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_train: bool,
    epoch_idx: int,
    total_epochs: int,
    model_name: str,
    seed: int,
    show_batch_progress: bool,
) -> Dict[str, float]:
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    desc = f"{model_name} seed={seed} epoch {epoch_idx}/{total_epochs} {'train' if is_train else 'eval'}"
    iterator = tqdm(dataloader, desc=desc, leave=False, disable=not show_batch_progress)
    for inputs, labels in iterator:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(inputs)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += batch_size

    return {
        "loss": total_loss / total_count,
        "acc": total_correct / total_count,
    }


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch_idx: int,
    total_epochs: int,
    model_name: str,
    seed: int,
    show_batch_progress: bool,
) -> Dict[str, float]:
    return run_epoch(
        model,
        dataloader,
        criterion,
        optimizer=None,
        device=device,
        is_train=False,
        epoch_idx=epoch_idx,
        total_epochs=total_epochs,
        model_name=model_name,
        seed=seed,
        show_batch_progress=show_batch_progress,
    )


def train_one_run(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_dataloaders(
        parquet_path=args.parquet_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        use_augmentation=args.use_augmentation,
        cache_path=args.cache_path,
    )

    model, spec = build_model(args.model_name, num_classes=10)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_state = None
    history: List[Dict[str, float]] = []

    start_time = time.time()
    epoch_iterator = tqdm(
        range(1, args.epochs + 1),
        desc=f"Run {args.run_id} | {args.model_name} | seed={args.seed}",
        disable=not args.show_epoch_progress,
    )
    for epoch in epoch_iterator:
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            is_train=True,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            model_name=args.model_name,
            seed=args.seed,
            show_batch_progress=args.show_batch_progress,
        )
        val_metrics = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            model_name=args.model_name,
            seed=args.seed,
            show_batch_progress=args.show_batch_progress,
        )
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "lr": scheduler.get_last_lr()[0],
            }
        )
        if val_metrics["acc"] >= best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        epoch_iterator.set_postfix(
            train_acc=f"{train_metrics['acc']:.4f}",
            val_acc=f"{val_metrics['acc']:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.5f}",
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        epoch_idx=args.epochs,
        total_epochs=args.epochs,
        model_name=args.model_name,
        seed=args.seed,
        show_batch_progress=args.show_batch_progress,
    )
    elapsed_min = (time.time() - start_time) / 60.0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{args.model_name}_seed{args.seed}"

    history_path = out_dir / f"{run_tag}_history.csv"
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)

    checkpoint_path = out_dir / f"{run_tag}.pt"
    torch.save(model.state_dict(), checkpoint_path)

    result = {
        "run_id": args.run_id,
        "model_name": args.model_name,
        "model_id": spec.model_id,
        "family": spec.family,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": "SGD",
        "lr": args.lr,
        "scheduler": "CosineAnnealingLR",
        "weight_decay": args.weight_decay,
        "augmentation": args.use_augmentation,
        "best_val_acc": best_val_acc,
        "test_acc": test_metrics["acc"],
        "train_time_min": elapsed_min,
        "params": count_parameters(model),
        "history_csv": str(history_path),
        "checkpoint": str(checkpoint_path),
    }
    result_path = out_dir / f"{run_tag}_metrics.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    result["metrics_json"] = str(result_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--parquet-path", type=str, default="data.parquet")
    parser.add_argument("--cache-path", type=str, default="results/cache/fashion_tensor_cache.pt")
    parser.add_argument("--output-dir", type=str, default="results/runs")
    parser.add_argument("--use-augmentation", action="store_true")
    parser.add_argument("--show-epoch-progress", action="store_true")
    parser.add_argument("--show-batch-progress", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_result = train_one_run(args)
    print(json.dumps(run_result))

