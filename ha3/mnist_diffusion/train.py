import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import get_mnist_loaders
from .diffusion import build_schedule, q_sample
from .models import MNISTUNet
from .utils import append_csv_row, cycle, ensure_dir, get_device, save_grid, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train one MNIST diffusion checkpoint.")
    parser.add_argument("--schedule", choices=["scaled_linear", "cosine"], required=True)
    parser.add_argument("--training-t", type=int, required=True)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out-dir", default="./results/checkpoints")
    parser.add_argument("--max-train-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--sample-interval", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    run_name = f"{args.schedule}_T{args.training_t}"
    run_dir = ensure_dir(Path(args.out_dir) / run_name)
    save_json(vars(args), run_dir / "config.json")

    train_loader, _ = get_mnist_loaders(
        args.data_dir, args.batch_size, args.num_workers, download=True
    )
    batches = cycle(train_loader)
    model = MNISTUNet(base_channels=args.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    schedule = build_schedule(args.schedule, args.training_t, device=device)

    schedule_summary = {
        "schedule": args.schedule,
        "training_T": args.training_t,
        "alpha_bar_T": schedule.alpha_bar_T,
        "max_train_steps": args.max_train_steps,
    }
    save_json(schedule_summary, run_dir / "schedule_summary.json")

    pbar = tqdm(range(1, args.max_train_steps + 1), desc=run_name)
    loss_ema = None
    for step in pbar:
        x0, _ = next(batches)
        x0 = x0.to(device)
        noise = torch.randn_like(x0)
        t = torch.randint(0, args.training_t, (x0.shape[0],), device=device).long()
        xt = q_sample(x0, t, noise, schedule)
        pred = model(xt, t)
        loss = F.mse_loss(pred, noise)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        value = loss.item()
        loss_ema = value if loss_ema is None else 0.98 * loss_ema + 0.02 * value
        pbar.set_postfix(loss=f"{value:.4f}", ema=f"{loss_ema:.4f}")

        if step % args.log_interval == 0 or step == 1:
            append_csv_row(
                run_dir / "train_log.csv",
                {"global_step": step, "loss": value, "loss_ema": loss_ema},
            )
        if step % args.sample_interval == 0 or step == args.max_train_steps:
            with torch.no_grad():
                # This is a quick noisy x_t diagnostic during training, not final sampling.
                n_vis = min(16, x0.shape[0])
                idx = torch.full(
                    (n_vis,), args.training_t - 1, device=device, dtype=torch.long
                )
                terminal = q_sample(x0[:n_vis], idx, torch.randn_like(x0[:n_vis]), schedule)
                save_grid(terminal, run_dir / f"terminal_xt_step{step}.png", nrow=4)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "schedule": schedule_summary,
            "global_step": args.max_train_steps,
        },
        run_dir / "model.pt",
    )
    print(f"saved checkpoint to {run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
