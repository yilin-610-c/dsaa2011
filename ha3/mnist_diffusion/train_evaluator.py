import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import get_mnist_loaders
from .metrics import classifier_accuracy
from .models import MNISTClassifier
from .utils import ensure_dir, get_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train the MNIST evaluator classifier.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out", default="./results/evaluator/mnist_classifier.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--min-accuracy", type=float, default=0.98)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    train_loader, test_loader = get_mnist_loaders(
        args.data_dir, args.batch_size, args.num_workers, download=True
    )
    model = MNISTClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * y.numel()
            total += y.numel()
        acc = classifier_accuracy(model, test_loader, device)
        print(
            f"epoch {epoch:02d} | train_loss {total_loss / total:.4f} | test_acc {acc:.4f}"
        )

    test_acc = classifier_accuracy(model, test_loader, device)
    if test_acc < args.min_accuracy:
        raise RuntimeError(
            f"Evaluator test accuracy {test_acc:.4f} is below required {args.min_accuracy:.4f}."
        )

    out = Path(args.out)
    ensure_dir(out.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_dim": 256,
            "test_accuracy": test_acc,
            "args": vars(args),
        },
        out,
    )
    save_json({"test_accuracy": test_acc, "args": vars(args)}, out.with_suffix(".json"))
    print(f"saved evaluator to {out} with test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()

