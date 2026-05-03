import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .data import get_mnist_loaders
from .diffusion import build_schedule, ddim_sample, ddpm_sample
from .metrics import (
    classifier_feature_fid,
    collect_real_features,
    load_classifier,
    mnist_is,
)
from .models import MNISTUNet
from .utils import Timer, ensure_dir, get_device, save_grid, set_seed, write_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Sample and evaluate all 8 HA3 configs.")
    parser.add_argument("--checkpoints-dir", default="./results/checkpoints")
    parser.add_argument("--evaluator", default="./results/evaluator/mnist_classifier.pt")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out-dir", default="./results/eval")
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_diffusion_checkpoint(path: Path, device):
    payload = torch.load(path, map_location=device)
    cfg = payload["config"]
    model = MNISTUNet(base_channels=cfg.get("base_channels", 64)).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    sched = build_schedule(cfg["schedule"], int(cfg["training_t"]), device=device)
    return model, sched, payload


@torch.no_grad()
def generate_batches(model, sched, sampler, batch_size, num_samples, device, ddim_steps):
    left = num_samples
    first = None
    probs = []
    feats = []
    while left > 0:
        bsz = min(batch_size, left)
        shape = (bsz, 1, 32, 32)
        if sampler == "ddpm":
            images = ddpm_sample(model, sched, shape, device)
        elif sampler == "ddim":
            images = ddim_sample(model, sched, shape, device, sampling_steps=ddim_steps)
        else:
            raise ValueError(sampler)
        if first is None:
            first = images[:64].detach().cpu()
        yield images
        left -= bsz


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = ensure_dir(args.out_dir)

    evaluator, evaluator_payload = load_classifier(args.evaluator, device)
    test_acc = float(evaluator_payload.get("test_accuracy", 0.0))
    if test_acc < 0.98:
        raise RuntimeError(
            f"Evaluator accuracy {test_acc:.4f} is below 0.98; retrain it before metrics."
        )
    _, test_loader = get_mnist_loaders(
        args.data_dir, args.batch_size, args.num_workers, download=True
    )
    _, real_features = collect_real_features(evaluator, test_loader, device)

    rows = []
    for schedule_name in ["scaled_linear", "cosine"]:
        for training_t in [200, 1000]:
            ckpt = Path(args.checkpoints_dir) / f"{schedule_name}_T{training_t}" / "model.pt"
            model, sched, payload = load_diffusion_checkpoint(ckpt, device)
            for sampler in ["ddpm", "ddim"]:
                sampling_steps = training_t if sampler == "ddpm" else args.ddim_steps
                probs_all = []
                feats_all = []
                first_images = None
                with Timer() as timer:
                    for images in generate_batches(
                        model,
                        sched,
                        sampler,
                        args.batch_size,
                        args.num_samples,
                        device,
                        args.ddim_steps,
                    ):
                        if first_images is None:
                            first_images = images[:64].detach().cpu()
                        with torch.no_grad():
                            p, f = evaluator(images.to(device), return_features=True)
                        probs_all.append(torch.softmax(p, dim=1).detach().cpu().numpy())
                        feats_all.append(f.detach().cpu().numpy())
                probs = np.concatenate(probs_all, axis=0)
                feats = np.concatenate(feats_all, axis=0)
                is_mean, is_std = mnist_is(probs)
                fid = classifier_feature_fid(real_features, feats)

                name = f"{schedule_name}_T{training_t}_{sampler}"
                save_grid(first_images, out_dir / f"samples_{name}.png")
                row = {
                    "schedule": schedule_name,
                    "training_T": training_t,
                    "sampler": sampler.upper(),
                    "sampling_steps": sampling_steps,
                    "sample_count": args.num_samples,
                    "runtime_seconds": timer.elapsed,
                    "seconds_per_sample": timer.elapsed / args.num_samples,
                    "MNIST_IS_mean": is_mean,
                    "MNIST_IS_std": is_std,
                    "Classifier_Feature_FID": fid,
                    "alpha_bar_T": sched.alpha_bar_T,
                    "checkpoint_global_step": payload.get("global_step", ""),
                    "evaluator_test_accuracy": test_acc,
                }
                rows.append(row)
                print(row)

    write_csv(out_dir / "results.csv", rows)
    print(f"saved results to {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
