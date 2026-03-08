import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm.auto import tqdm


CORE_MODELS = ["plain_cnn", "resnet18_optb", "resnet18_opta", "preact_resnet18"]
WIDE_MODEL = "wide_resnet14"
SUPPLEMENT_MODELS = ["resnet34_optb", "resnet50_bottleneck"]


def build_jobs(seeds: List[int], include_wide: bool, include_supplement: bool, supplement_only: bool) -> List[Dict[str, object]]:
    jobs: List[Dict[str, object]] = []
    run_counter = 1
    model_names: List[str] = []
    if supplement_only:
        model_names.extend(SUPPLEMENT_MODELS)
    else:
        model_names.extend(CORE_MODELS)
        if include_wide:
            model_names.append(WIDE_MODEL)
        if include_supplement:
            model_names.extend(SUPPLEMENT_MODELS)
    for seed in seeds:
        for model_name in model_names:
            jobs.append(
                {
                    "run_id": f"R{run_counter:03d}",
                    "model_name": model_name,
                    "seed": seed,
                }
            )
            run_counter += 1
    return jobs


def run_jobs(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    include_supplement = args.include_supplement or args.supplement_only
    jobs = build_jobs(
        args.seeds,
        include_wide=args.include_wide,
        include_supplement=include_supplement,
        supplement_only=args.supplement_only,
    )
    rows: List[Dict[str, object]] = []

    jobs_progress = tqdm(jobs, desc="All runs", unit="run")
    for job in jobs_progress:
        run_start = time.time()
        run_tag = f"{job['model_name']}_seed{job['seed']}"
        run_output_dir = output_dir / "runs"
        metrics_path = run_output_dir / f"{run_tag}_metrics.json"
        cmd = [
            "python",
            "-m",
            "experiments.train_eval",
            "--run-id",
            job["run_id"],
            "--model-name",
            job["model_name"],
            "--seed",
            str(job["seed"]),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--momentum",
            str(args.momentum),
            "--weight-decay",
            str(args.weight_decay),
            "--val-ratio",
            str(args.val_ratio),
            "--num-workers",
            str(args.num_workers),
            "--parquet-path",
            args.parquet_path,
            "--cache-path",
            args.cache_path,
            "--output-dir",
            str(run_output_dir),
        ]
        if args.use_augmentation:
            cmd.append("--use-augmentation")
        if args.show_epoch_progress:
            cmd.append("--show-epoch-progress")
        if args.show_batch_progress:
            cmd.append("--show-batch-progress")
        subprocess.run(cmd, check=True)
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(result)
        elapsed_min = (time.time() - run_start) / 60.0
        jobs_progress.set_postfix(
            run=job["run_id"],
            model=job["model_name"],
            seed=job["seed"],
            acc=f"{result['test_acc']:.4f}",
            min=f"{elapsed_min:.1f}",
        )
        print(
            f"Completed {job['run_id']} | {job['model_name']} | seed={job['seed']} | "
            f"test_acc={result['test_acc']:.4f} | elapsed={elapsed_min:.1f} min"
        )

    run_df = pd.DataFrame(rows)
    run_df.to_csv(output_dir / "run_matrix.csv", index=False)
    return run_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 3407, 2025])
    parser.add_argument("--include-wide", action="store_true")
    parser.add_argument("--include-supplement", action="store_true")
    parser.add_argument("--supplement-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--parquet-path", type=str, default="data.parquet")
    parser.add_argument("--cache-path", type=str, default="results/cache/fashion_tensor_cache.pt")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--use-augmentation", action="store_true")
    parser.add_argument("--show-epoch-progress", action="store_true")
    parser.add_argument("--show-batch-progress", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_jobs(args)

