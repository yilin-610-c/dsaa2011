import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from models import build_model, count_parameters


MODEL_NAMES = [
    "plain_cnn",
    "resnet10_optb",
    "resnet18_optb",
    "resnet18_opta",
    "preact_resnet18",
    "wide_resnet14",
    "resnet34_optb",
    "resnet50_bottleneck",
]


def build_model_definition_matrix(output_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model_name in MODEL_NAMES:
        model, spec = build_model(model_name)
        rows.append(
            {
                "ModelID": spec.model_id,
                "Family": spec.family,
                "Residual": "Yes" if spec.residual else "No",
                "ShortcutType": spec.shortcut_type,
                "BlockType": spec.block_type,
                "Depth": spec.depth if spec.depth > 0 else "-",
                "WidthMultiplier": spec.width_multiplier,
                "Stem": spec.stem,
                "DownsampleStrategy": spec.downsample_strategy,
                "Params(M)": round(count_parameters(model) / 1e6, 3),
                "model_name": model_name,
            }
        )
    matrix = pd.DataFrame(rows)
    matrix.to_csv(output_dir / "model_definition_matrix.csv", index=False)
    return matrix


def summarize_runs(run_matrix: pd.DataFrame, model_def: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    agg = (
        run_matrix.groupby(["model_id", "model_name", "family"], as_index=False)
        .agg(test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"), best_val_acc_mean=("best_val_acc", "mean"))
        .fillna(0.0)
    )
    param_map = model_def.set_index("model_name")["Params(M)"].to_dict()
    agg["Params(M)"] = agg["model_name"].map(param_map)
    baseline = float(agg.loc[agg["model_id"] == "M1", "test_acc_mean"].iloc[0])
    agg["RelativeGain vs M1"] = agg["test_acc_mean"] - baseline
    agg["AccuracyPerMParam"] = agg["test_acc_mean"] / agg["Params(M)"]
    agg = agg.rename(
        columns={
            "model_id": "ModelID",
            "family": "Family",
            "test_acc_mean": "TestAcc Mean(%)",
            "test_acc_std": "TestAcc Std",
            "best_val_acc_mean": "BestValAcc Mean(%)",
        }
    )
    agg["TestAcc Mean(%)"] = (agg["TestAcc Mean(%)"] * 100.0).round(3)
    agg["TestAcc Std"] = (agg["TestAcc Std"] * 100.0).round(3)
    agg["BestValAcc Mean(%)"] = (agg["BestValAcc Mean(%)"] * 100.0).round(3)
    agg["RelativeGain vs M1"] = (agg["RelativeGain vs M1"] * 100.0).round(3)
    agg["AccuracyPerMParam"] = agg["AccuracyPerMParam"].round(4)
    summary_cols = [
        "ModelID",
        "Family",
        "Params(M)",
        "TestAcc Mean(%)",
        "TestAcc Std",
        "BestValAcc Mean(%)",
        "RelativeGain vs M1",
        "AccuracyPerMParam",
    ]
    summary = agg[summary_cols].sort_values("ModelID").reset_index(drop=True)
    summary.to_csv(output_dir / "summary_matrix.csv", index=False)
    return summary


def plot_training_curves(run_matrix: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for model_name, sub_df in run_matrix.groupby("model_name"):
        sample_history_path = sub_df.iloc[0]["history_csv"]
        history_df = pd.read_csv(sample_history_path)
        axes[0].plot(history_df["epoch"], history_df["train_acc"], label=model_name)
        axes[1].plot(history_df["epoch"], history_df["val_acc"], label=model_name)
    axes[0].set_title("Train Accuracy Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "curve_train_val.png", dpi=180)
    plt.close(fig)


def plot_capacity_vs_accuracy(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = summary["Params(M)"]
    y = summary["TestAcc Mean(%)"]
    ax.scatter(x, y)
    for _, row in summary.iterrows():
        ax.annotate(row["ModelID"], (row["Params(M)"], row["TestAcc Mean(%)"]), fontsize=8)
    ax.set_xlabel("Params (Millions)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Capacity vs Accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "capacity_vs_accuracy.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-matrix", type=str, default="results/run_matrix.csv")
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_matrix = pd.read_csv(args.run_matrix)
    model_def = build_model_definition_matrix(out_dir)
    summary = summarize_runs(run_matrix, model_def, out_dir)
    plot_training_curves(run_matrix, out_dir)
    plot_capacity_vs_accuracy(summary, out_dir)
