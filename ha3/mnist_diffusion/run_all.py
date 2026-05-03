import argparse
import subprocess
import sys


def run(cmd):
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Convenience launcher for HA3 experiments.")
    parser.add_argument(
        "--stage",
        choices=["diagnostics", "evaluator", "train", "eval", "far", "all"],
        default="all",
    )
    parser.add_argument("--max-train-steps", type=int, default=10000)
    parser.add_argument("--num-samples", type=int, default=2000)
    return parser.parse_args()


def main():
    args = parse_args()
    py = sys.executable
    if args.stage in {"diagnostics", "all"}:
        run([py, "-m", "mnist_diffusion.diagnostics"])
    if args.stage in {"evaluator", "all"}:
        run([py, "-m", "mnist_diffusion.train_evaluator"])
    if args.stage in {"train", "all"}:
        for schedule in ["scaled_linear", "cosine"]:
            for training_t in ["200", "1000"]:
                run(
                    [
                        py,
                        "-m",
                        "mnist_diffusion.train",
                        "--schedule",
                        schedule,
                        "--training-t",
                        training_t,
                        "--max-train-steps",
                        str(args.max_train_steps),
                    ]
                )
    if args.stage in {"eval", "all"}:
        run([py, "-m", "mnist_diffusion.sample_eval", "--num-samples", str(args.num_samples)])
    if args.stage in {"far", "all"}:
        run([py, "-m", "mnist_diffusion.far_from_origin"])


if __name__ == "__main__":
    main()

