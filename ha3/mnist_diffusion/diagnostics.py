import argparse
from pathlib import Path

import numpy as np
import torch

from .data import get_mnist_loaders
from .diffusion import build_schedule, q_sample
from .utils import ensure_dir, get_device, save_grid, set_matplotlib_cache, set_seed, write_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Save schedule and terminal-noise diagnostics.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out-dir", default="./results/diagnostics")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    set_matplotlib_cache()
    import matplotlib.pyplot as plt

    device = get_device(args.device)
    out_dir = ensure_dir(args.out_dir)
    train_loader, _ = get_mnist_loaders(
        args.data_dir, args.batch_size, args.num_workers, download=True
    )
    x0, _ = next(iter(train_loader))
    x0 = x0.to(device)

    rows = []
    plt.figure(figsize=(8, 5))
    for schedule_name in ["scaled_linear", "cosine"]:
        for training_t in [200, 1000]:
            sched = build_schedule(schedule_name, training_t, device=device)
            label = f"{schedule_name}, T={training_t}"
            alpha = sched.alpha_bars.detach().cpu().numpy()
            plt.plot(np.arange(1, training_t + 1), alpha, label=label)

            t = torch.full((x0.shape[0],), training_t - 1, device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            xt = q_sample(x0, t, noise, sched)
            save_grid(xt[:64], out_dir / f"terminal_xt_{schedule_name}_T{training_t}.png")
            rows.append(
                {
                    "schedule": schedule_name,
                    "training_T": training_t,
                    "alpha_bar_T": sched.alpha_bar_T,
                    "terminal_mean": float(xt.mean().detach().cpu()),
                    "terminal_std": float(xt.std().detach().cpu()),
                    "terminal_min": float(xt.min().detach().cpu()),
                    "terminal_max": float(xt.max().detach().cpu()),
                }
            )

    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel("alpha_bar_t")
    plt.title("alpha_bar_t curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "alpha_bar_curves.png", dpi=200)
    plt.close()
    write_csv(out_dir / "schedule_terminal_summary.csv", rows)
    print(f"saved diagnostics to {out_dir}")


if __name__ == "__main__":
    main()

