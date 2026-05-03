import argparse
import csv
from pathlib import Path

import torch

from .diffusion import build_schedule, ddim_sample, ddpm_sample
from .models import MNISTUNet
from .utils import ensure_dir, get_device, save_grid, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Far-from-origin terminal latent analysis.")
    parser.add_argument("--results-csv", default="./results/eval/results.csv")
    parser.add_argument("--checkpoints-dir", default="./results/checkpoints")
    parser.add_argument("--out-dir", default="./results/far_from_origin")
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def best_checkpoint_from_results(results_csv: str | Path):
    with Path(results_csv).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {results_csv}")
    best = min(rows, key=lambda r: float(r["Classifier_Feature_FID"]))
    return best["schedule"], int(best["training_T"]), best


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = ensure_dir(args.out_dir)
    schedule_name, training_t, best_row = best_checkpoint_from_results(args.results_csv)
    ckpt_path = Path(args.checkpoints_dir) / f"{schedule_name}_T{training_t}" / "model.pt"
    payload = torch.load(ckpt_path, map_location=device)
    model = MNISTUNet(base_channels=payload["config"].get("base_channels", 64)).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    sched = build_schedule(schedule_name, training_t, device=device)

    base_z = torch.randn(args.grid_size, 1, 32, 32, device=device)
    scales = [1, 2, 4, 8]
    rows = []
    for scale in scales:
        initial = scale * base_z
        if args.sampler == "ddim":
            images = ddim_sample(
                model,
                sched,
                initial.shape,
                device,
                sampling_steps=args.ddim_steps,
                initial_noise=initial,
            )
            sampling_steps = args.ddim_steps
        else:
            images = ddpm_sample(model, sched, initial.shape, device, initial_noise=initial)
            sampling_steps = training_t
        save_grid(images, out_dir / f"far_origin_scale_{scale}.png")
        rows.append(
            {
                "scale": scale,
                "x_T_definition": "x_T = scale * z, z ~ N(0,I)",
                "schedule": schedule_name,
                "training_T": training_t,
                "sampler": args.sampler.upper(),
                "sampling_steps": sampling_steps,
                "base_noise_shared_across_scales": True,
                "selected_from_results_row": best_row,
            }
        )

    with (out_dir / "far_from_origin_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Far-from-origin analysis\n")
        f.write("x_T = scale * z, z ~ N(0,I); scales = [1, 2, 4, 8]\n")
        f.write("The same base noise z is reused across scales for the comparison grid.\n")
        f.write(
            "Large scales move x_T outside the typical region of the standard Gaussian prior.\n"
        )
        f.write(f"Selected checkpoint: {schedule_name}_T{training_t}\n")
        f.write(f"Sampler: {args.sampler.upper()}\n")
    print(f"saved far-from-origin assets to {out_dir}")


if __name__ == "__main__":
    main()

