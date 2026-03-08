import argparse
from pathlib import Path

import pandas as pd


def pick(summary: pd.DataFrame, model_id: str) -> pd.Series:
    return summary.loc[summary["ModelID"] == model_id].iloc[0]


def main(summary_csv: str, output_txt: str) -> None:
    summary = pd.read_csv(summary_csv)
    m1 = pick(summary, "M1")
    m2 = pick(summary, "M2")
    m3 = pick(summary, "M3")
    m4 = pick(summary, "M4")
    m5_exists = (summary["ModelID"] == "M5").any()
    if m5_exists:
        m5 = pick(summary, "M5")

    lines = []
    lines.append("FashionMNIST-Resplit Technical Report Draft")
    lines.append("")
    lines.append("(a) Implementation")
    lines.append("Implemented Plain-CNN baseline and ResNet variants with controlled protocol.")
    lines.append("Core modules include convolution stem, residual blocks, skip connections, and BatchNorm.")
    lines.append("")
    lines.append("(b) Experimental results")
    lines.append("Please cite results/summary_matrix.csv and result figures in results/*.png.")
    lines.append("")
    lines.append("(c) Discussion and conclusions")
    lines.append(
        f"Q1 M2 vs M1: residual gain = {m2['TestAcc Mean(%)'] - m1['TestAcc Mean(%)']:.3f} percentage points."
    )
    lines.append(
        f"Q2 M2 vs M3: OptionB - OptionA = {m2['TestAcc Mean(%)'] - m3['TestAcc Mean(%)']:.3f} percentage points."
    )
    lines.append(
        f"Q3 M4 vs M2: PreAct - ResNet18 = {m4['TestAcc Mean(%)'] - m2['TestAcc Mean(%)']:.3f} percentage points."
    )
    if m5_exists:
        lines.append(
            f"Q4 M5 vs M2: Wide14 - ResNet18 = {m5['TestAcc Mean(%)'] - m2['TestAcc Mean(%)']:.3f} percentage points."
        )
    lines.append("Include analysis of parameter efficiency using AccuracyPerMParam.")
    lines.append("")
    lines.append("(d) Additional notes")
    lines.append("Add training stability remarks from curve_train_val.png.")
    lines.append("Document external references if any non-standard code is used.")

    out_path = Path(output_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=str, default="results/summary_matrix.csv")
    parser.add_argument("--output-txt", type=str, default="report/report_outline.txt")
    args = parser.parse_args()
    main(args.summary_csv, args.output_txt)
