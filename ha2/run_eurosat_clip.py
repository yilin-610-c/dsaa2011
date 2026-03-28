#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import EuroSAT


SEED = 2026
CLASS_NAME_MAP = {
    "AnnualCrop": "annual crop land",
    "Forest": "forest",
    "HerbaceousVegetation": "herbaceous vegetation",
    "Highway": "highway or road",
    "Industrial": "industrial area",
    "Pasture": "pasture land",
    "PermanentCrop": "permanent crop land",
    "Residential": "residential area",
    "River": "river",
    "SeaLake": "sea or lake",
}
SIMPLE_TEMPLATE = "a photo of a {class_name}."
OFFICIAL_ENSEMBLE_TEMPLATES = [
    "a centered satellite photo of {class_name}.",
    "a centered satellite photo of a {class_name}.",
    "a centered satellite photo of the {class_name}.",
]
CASE_TYPES = [
    "simple_wrong_ensemble_right",
    "both_wrong",
    "both_correct_confidence_shift",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class ClipBackend:
    name: str
    model_name: str
    pretrained_name: str
    model: torch.nn.Module
    preprocess: object
    tokenizer: object
    device: str

    @property
    def logit_scale(self) -> torch.Tensor:
        if hasattr(self.model, "logit_scale"):
            return self.model.logit_scale.exp()
        return torch.tensor(100.0, device=self.device)

    def tokenize(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(prompts)
        if isinstance(tokens, dict):
            return tokens["input_ids"]
        return tokens

    @torch.no_grad()
    def encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenize(prompts).to(self.device)
        if self.name == "open_clip":
            features = self.model.encode_text(tokens)
        else:
            features = self.model.encode_text(tokens)
        return features

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(images.to(self.device))


def load_clip_backend(model_name: str, pretrained_name: str, device: str) -> ClipBackend:
    try:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained_name,
            device=device,
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()
        return ClipBackend(
            name="open_clip",
            model_name=model_name,
            pretrained_name=pretrained_name,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            device=device,
        )
    except ImportError:
        import clip

        model, preprocess = clip.load(model_name, device=device)
        model.eval()
        return ClipBackend(
            name="clip",
            model_name=model_name,
            pretrained_name="openai",
            model=model,
            preprocess=preprocess,
            tokenizer=clip.tokenize,
            device=device,
        )


def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": output_dir,
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
        "cases": output_dir / "figures" / "cases",
        "predictions": output_dir / "predictions",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def build_dataset(root: Path, transform, download: bool) -> EuroSAT:
    return EuroSAT(root=str(root), transform=transform, download=download)


def build_test_subset(dataset: EuroSAT, test_size: float, seed: int) -> Tuple[Subset, np.ndarray]:
    targets = np.array(dataset.targets)
    indices = np.arange(len(dataset))
    _, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=targets,
    )
    test_indices = np.sort(test_indices)
    return Subset(dataset, test_indices.tolist()), test_indices


def build_prompt_sets(
    class_names: Sequence[str],
    ensemble_templates: Sequence[str],
) -> Dict[str, List[List[str]]]:
    prompt_sets = {"simple": [], "ensemble": []}
    for class_name in class_names:
        prompt_sets["simple"].append([SIMPLE_TEMPLATE.format(class_name=class_name)])
        prompt_sets["ensemble"].append(
            [template.format(class_name=class_name) for template in ensemble_templates]
        )
    return prompt_sets


@torch.no_grad()
def build_zero_shot_weights(backend: ClipBackend, prompt_groups: Sequence[Sequence[str]]) -> torch.Tensor:
    class_embeddings = []
    for prompts in prompt_groups:
        text_features = backend.encode_text(prompts)
        text_features = F.normalize(text_features, dim=-1)
        class_feature = text_features.mean(dim=0, keepdim=True)
        class_feature = F.normalize(class_feature, dim=-1)
        class_embeddings.append(class_feature)
    return torch.cat(class_embeddings, dim=0)


def extract_sample_path(dataset: EuroSAT, absolute_index: int) -> str:
    sample_path, _ = dataset.samples[absolute_index]
    return sample_path


@torch.no_grad()
def predict_dataset(
    backend: ClipBackend,
    loader: DataLoader,
    classifier: torch.Tensor,
    class_labels: Sequence[str],
    absolute_indices: Sequence[int],
    dataset: EuroSAT,
) -> pd.DataFrame:
    rows = []
    classifier = classifier.to(backend.device)
    scale = backend.logit_scale
    current = 0
    for images, targets in loader:
        image_features = backend.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        logits = scale * image_features @ classifier.T
        probs = logits.softmax(dim=-1)
        topk_probs, topk_indices = probs.topk(k=min(3, len(class_labels)), dim=-1)
        for offset in range(images.size(0)):
            abs_idx = int(absolute_indices[current + offset])
            target_idx = int(targets[offset].item())
            pred_idx = int(topk_indices[offset, 0].item())
            top1_prob = float(topk_probs[offset, 0].item())
            top2_prob = float(topk_probs[offset, 1].item()) if topk_probs.size(1) > 1 else 0.0
            rows.append(
                {
                    "absolute_index": abs_idx,
                    "sample_path": extract_sample_path(dataset, abs_idx),
                    "true_index": target_idx,
                    "true_label": class_labels[target_idx],
                    "pred_index": pred_idx,
                    "pred_label": class_labels[pred_idx],
                    "correct": int(pred_idx == target_idx),
                    "top1_prob": top1_prob,
                    "top2_prob": top2_prob,
                    "margin_top1_top2": top1_prob - top2_prob,
                    "top1_index": int(topk_indices[offset, 0].item()),
                    "top1_label": class_labels[int(topk_indices[offset, 0].item())],
                    "top1_prob_label": float(topk_probs[offset, 0].item()),
                    "top2_index": int(topk_indices[offset, 1].item()) if topk_probs.size(1) > 1 else -1,
                    "top2_label": class_labels[int(topk_indices[offset, 1].item())]
                    if topk_probs.size(1) > 1
                    else "",
                    "top2_prob_label": float(topk_probs[offset, 1].item()) if topk_probs.size(1) > 1 else 0.0,
                    "top3_index": int(topk_indices[offset, 2].item()) if topk_probs.size(1) > 2 else -1,
                    "top3_label": class_labels[int(topk_indices[offset, 2].item())]
                    if topk_probs.size(1) > 2
                    else "",
                    "top3_prob_label": float(topk_probs[offset, 2].item()) if topk_probs.size(1) > 2 else 0.0,
                }
            )
        current += images.size(0)
    return pd.DataFrame(rows)


def bootstrap_accuracy(y_true: np.ndarray, y_pred: np.ndarray, seed: int, rounds: int = 1000) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = []
    for _ in range(rounds):
        sample_idx = rng.integers(0, n, size=n)
        samples.append(float(np.mean(y_true[sample_idx] == y_pred[sample_idx])))
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples, ddof=1)),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def build_confusion_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Sequence[str],
    title: str,
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_labels))), normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels(class_labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_case_figure(
    case_df: pd.DataFrame,
    title: str,
    output_path: Path,
    limit: int = 6,
) -> None:
    if case_df.empty:
        return
    limit = min(limit, len(case_df))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for ax in axes:
        ax.axis("off")
    for idx, (_, row) in enumerate(case_df.head(limit).iterrows()):
        image = Image.open(row["sample_path"]).convert("RGB")
        axes[idx].imshow(image)
        axes[idx].set_title(
            "\n".join(
                [
                    f"True: {row['true_label']}",
                    f"Simple: {row['simple_pred_label']} ({row['simple_top1_prob']:.3f})",
                    f"Ensemble: {row['ensemble_pred_label']} ({row['ensemble_top1_prob']:.3f})",
                ]
            ),
            fontsize=9,
        )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_pairwise_confusions(df: pd.DataFrame, class_labels: Sequence[str]) -> pd.DataFrame:
    rows = []
    for prompt_name in ["simple", "ensemble"]:
        wrong = df[df[f"{prompt_name}_correct"] == 0]
        counts = (
            wrong.groupby(["true_label", f"{prompt_name}_pred_label"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        counts["prompt"] = prompt_name
        counts = counts.rename(columns={f"{prompt_name}_pred_label": "pred_label"})
        rows.append(counts)
    if not rows:
        return pd.DataFrame(columns=["prompt", "true_label", "pred_label", "count"])
    return pd.concat(rows, ignore_index=True)


def build_classwise_metrics(df: pd.DataFrame, class_labels: Sequence[str]) -> pd.DataFrame:
    rows = []
    for class_name in class_labels:
        class_slice = df[df["true_label"] == class_name]
        rows.append(
            {
                "class_name": class_name,
                "support": int(len(class_slice)),
                "simple_accuracy": float(class_slice["simple_correct"].mean()),
                "ensemble_accuracy": float(class_slice["ensemble_correct"].mean()),
                "delta_accuracy": float(
                    class_slice["ensemble_correct"].mean() - class_slice["simple_correct"].mean()
                ),
                "simple_avg_confidence": float(class_slice["simple_top1_prob"].mean()),
                "ensemble_avg_confidence": float(class_slice["ensemble_top1_prob"].mean()),
                "delta_confidence": float(
                    class_slice["ensemble_top1_prob"].mean() - class_slice["simple_top1_prob"].mean()
                ),
                "simple_avg_margin": float(class_slice["simple_margin_top1_top2"].mean()),
                "ensemble_avg_margin": float(class_slice["ensemble_margin_top1_top2"].mean()),
                "delta_margin": float(
                    class_slice["ensemble_margin_top1_top2"].mean()
                    - class_slice["simple_margin_top1_top2"].mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("delta_accuracy", ascending=False)


def merge_predictions(simple_df: pd.DataFrame, ensemble_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "absolute_index",
        "sample_path",
        "true_index",
        "true_label",
        "pred_index",
        "pred_label",
        "correct",
        "top1_prob",
        "top2_prob",
        "margin_top1_top2",
        "top2_label",
    ]
    simple_df = simple_df[keep_cols].rename(
        columns={
            "pred_index": "simple_pred_index",
            "pred_label": "simple_pred_label",
            "correct": "simple_correct",
            "top1_prob": "simple_top1_prob",
            "top2_prob": "simple_top2_prob",
            "margin_top1_top2": "simple_margin_top1_top2",
            "top2_label": "simple_top2_label",
        }
    )
    ensemble_df = ensemble_df[keep_cols].rename(
        columns={
            "pred_index": "ensemble_pred_index",
            "pred_label": "ensemble_pred_label",
            "correct": "ensemble_correct",
            "top1_prob": "ensemble_top1_prob",
            "top2_prob": "ensemble_top2_prob",
            "margin_top1_top2": "ensemble_margin_top1_top2",
            "top2_label": "ensemble_top2_label",
        }
    )
    merged = simple_df.merge(
        ensemble_df,
        on=["absolute_index", "sample_path", "true_index", "true_label"],
        how="inner",
    )
    merged["case_type"] = "other"
    merged.loc[
        (merged["simple_correct"] == 0) & (merged["ensemble_correct"] == 1),
        "case_type",
    ] = "simple_wrong_ensemble_right"
    merged.loc[
        (merged["simple_correct"] == 0) & (merged["ensemble_correct"] == 0),
        "case_type",
    ] = "both_wrong"
    confidence_shift = (merged["ensemble_top1_prob"] - merged["simple_top1_prob"]).abs()
    merged.loc[
        (merged["simple_correct"] == 1) & (merged["ensemble_correct"] == 1) & (confidence_shift >= 0.15),
        "case_type",
    ] = "both_correct_confidence_shift"
    return merged


def select_case_examples(merged_df: pd.DataFrame, limit_per_case: int = 6) -> Dict[str, pd.DataFrame]:
    examples = {}
    improved = merged_df[merged_df["case_type"] == "simple_wrong_ensemble_right"].sort_values(
        "ensemble_top1_prob", ascending=False
    )
    examples["simple_wrong_ensemble_right"] = improved.head(limit_per_case)
    both_wrong = merged_df[merged_df["case_type"] == "both_wrong"].copy()
    both_wrong["combined_conf"] = both_wrong["simple_top1_prob"] + both_wrong["ensemble_top1_prob"]
    examples["both_wrong"] = both_wrong.sort_values("combined_conf", ascending=False).head(limit_per_case)
    shifted = merged_df[merged_df["case_type"] == "both_correct_confidence_shift"].copy()
    shifted["confidence_delta_abs"] = (
        shifted["ensemble_top1_prob"] - shifted["simple_top1_prob"]
    ).abs()
    examples["both_correct_confidence_shift"] = shifted.sort_values(
        "confidence_delta_abs", ascending=False
    ).head(limit_per_case)
    return examples


def save_prompt_details(prompt_sets: Dict[str, List[List[str]]], class_labels: Sequence[str], output_path: Path) -> None:
    rows = []
    for prompt_name, class_prompts in prompt_sets.items():
        for class_label, prompts in zip(class_labels, class_prompts):
            rows.append(
                {
                    "prompt_setting": prompt_name,
                    "class_name": class_label,
                    "num_templates": len(prompts),
                    "templates": " || ".join(prompts),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def compute_summary(
    merged_df: pd.DataFrame,
    classwise_df: pd.DataFrame,
    boot_simple: Dict[str, float],
    boot_ensemble: Dict[str, float],
    backend: ClipBackend,
    test_size: float,
    output_dir: Path,
) -> Dict[str, object]:
    improved = int(
        ((merged_df["simple_correct"] == 0) & (merged_df["ensemble_correct"] == 1)).sum()
    )
    regressed = int(
        ((merged_df["simple_correct"] == 1) & (merged_df["ensemble_correct"] == 0)).sum()
    )
    unchanged_correct = int(
        ((merged_df["simple_correct"] == 1) & (merged_df["ensemble_correct"] == 1)).sum()
    )
    unchanged_wrong = int(
        ((merged_df["simple_correct"] == 0) & (merged_df["ensemble_correct"] == 0)).sum()
    )
    summary = {
        "dataset": "EuroSAT",
        "backend": backend.name,
        "model_name": backend.model_name,
        "pretrained_name": backend.pretrained_name,
        "seed": SEED,
        "test_fraction": test_size,
        "num_test_samples": int(len(merged_df)),
        "simple_accuracy": float(merged_df["simple_correct"].mean()),
        "ensemble_accuracy": float(merged_df["ensemble_correct"].mean()),
        "accuracy_delta": float(merged_df["ensemble_correct"].mean() - merged_df["simple_correct"].mean()),
        "simple_bootstrap": boot_simple,
        "ensemble_bootstrap": boot_ensemble,
        "improved_count": improved,
        "regressed_count": regressed,
        "unchanged_correct_count": unchanged_correct,
        "unchanged_wrong_count": unchanged_wrong,
        "most_improved_classes": classwise_df.head(3)["class_name"].tolist(),
        "most_degraded_classes": classwise_df.tail(3)["class_name"].tolist(),
        "output_dir": str(output_dir),
    }
    return summary


def save_summary_markdown(summary: Dict[str, object], output_path: Path) -> None:
    lines = [
        "# EuroSAT CLIP Experiment Summary",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Backend: `{summary['backend']}`",
        f"- Model: `{summary['model_name']}`",
        f"- Pretrained weights: `{summary['pretrained_name']}`",
        f"- Test samples: `{summary['num_test_samples']}`",
        f"- Simple accuracy: `{summary['simple_accuracy']:.4f}`",
        f"- Ensemble accuracy: `{summary['ensemble_accuracy']:.4f}`",
        f"- Accuracy delta: `{summary['accuracy_delta']:.4f}`",
        f"- Improved samples: `{summary['improved_count']}`",
        f"- Regressed samples: `{summary['regressed_count']}`",
        "",
        "## Stability",
        "",
        f"- Simple bootstrap mean/std: `{summary['simple_bootstrap']['mean']:.4f}` / `{summary['simple_bootstrap']['std']:.4f}`",
        f"- Simple 95% CI: `[{summary['simple_bootstrap']['ci_low']:.4f}, {summary['simple_bootstrap']['ci_high']:.4f}]`",
        f"- Ensemble bootstrap mean/std: `{summary['ensemble_bootstrap']['mean']:.4f}` / `{summary['ensemble_bootstrap']['std']:.4f}`",
        f"- Ensemble 95% CI: `[{summary['ensemble_bootstrap']['ci_low']:.4f}, {summary['ensemble_bootstrap']['ci_high']:.4f}]`",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data", type=str)
    parser.add_argument("--output-dir", default="outputs/eurosat_vitb32", type=str)
    parser.add_argument("--model-name", default="ViT-B/32", type=str)
    parser.add_argument("--pretrained", default="openai", type=str)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument(
        "--ensemble-preset",
        default="official",
        choices=["official"],
        type=str,
    )
    args = parser.parse_args()

    set_seed(SEED)
    output_paths = ensure_dirs(Path(args.output_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(output_paths["root"] / ".mplconfig"))

    backend = load_clip_backend(args.model_name, args.pretrained, args.device)
    dataset = build_dataset(Path(args.dataset_root), backend.preprocess, args.download)
    normalized_class_labels = [CLASS_NAME_MAP.get(label, label.lower()) for label in dataset.classes]
    if args.ensemble_preset == "official":
        ensemble_templates = OFFICIAL_ENSEMBLE_TEMPLATES
    else:
        raise ValueError(f"Unsupported ensemble preset: {args.ensemble_preset}")

    prompt_sets = build_prompt_sets(normalized_class_labels, ensemble_templates)
    save_prompt_details(prompt_sets, normalized_class_labels, output_paths["tables"] / "prompt_details.csv")

    test_subset, absolute_indices = build_test_subset(dataset, args.test_size, SEED)
    loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    simple_classifier = build_zero_shot_weights(backend, prompt_sets["simple"])
    ensemble_classifier = build_zero_shot_weights(backend, prompt_sets["ensemble"])
    classifier_delta = float(torch.mean(torch.abs(simple_classifier - ensemble_classifier)).item())

    simple_df = predict_dataset(
        backend,
        loader,
        simple_classifier,
        normalized_class_labels,
        absolute_indices,
        dataset,
    )
    ensemble_df = predict_dataset(
        backend,
        loader,
        ensemble_classifier,
        normalized_class_labels,
        absolute_indices,
        dataset,
    )
    simple_df.to_csv(output_paths["predictions"] / "simple_predictions.csv", index=False)
    ensemble_df.to_csv(output_paths["predictions"] / "ensemble_predictions.csv", index=False)

    merged_df = merge_predictions(simple_df, ensemble_df)
    merged_df["confidence_delta"] = merged_df["ensemble_top1_prob"] - merged_df["simple_top1_prob"]
    merged_df["margin_delta"] = merged_df["ensemble_margin_top1_top2"] - merged_df["simple_margin_top1_top2"]
    merged_df.to_csv(output_paths["predictions"] / "combined_predictions.csv", index=False)

    y_true = merged_df["true_index"].to_numpy()
    simple_pred = merged_df["simple_pred_index"].to_numpy()
    ensemble_pred = merged_df["ensemble_pred_index"].to_numpy()
    boot_simple = bootstrap_accuracy(y_true, simple_pred, seed=SEED)
    boot_ensemble = bootstrap_accuracy(y_true, ensemble_pred, seed=SEED + 1)

    classwise_df = build_classwise_metrics(merged_df, normalized_class_labels)
    classwise_df.to_csv(output_paths["tables"] / "classwise_metrics.csv", index=False)
    confusion_pairs_df = build_pairwise_confusions(merged_df, normalized_class_labels)
    confusion_pairs_df.to_csv(output_paths["tables"] / "confusion_pairs.csv", index=False)

    overall_df = pd.DataFrame(
        [
            {
                "prompt_setting": "simple",
                "accuracy": float(merged_df["simple_correct"].mean()),
                "bootstrap_mean": boot_simple["mean"],
                "bootstrap_std": boot_simple["std"],
                "bootstrap_ci_low": boot_simple["ci_low"],
                "bootstrap_ci_high": boot_simple["ci_high"],
            },
            {
                "prompt_setting": "ensemble",
                "accuracy": float(merged_df["ensemble_correct"].mean()),
                "bootstrap_mean": boot_ensemble["mean"],
                "bootstrap_std": boot_ensemble["std"],
                "bootstrap_ci_low": boot_ensemble["ci_low"],
                "bootstrap_ci_high": boot_ensemble["ci_high"],
            },
        ]
    )
    overall_df.to_csv(output_paths["tables"] / "overall_metrics.csv", index=False)

    build_confusion_figure(
        y_true,
        simple_pred,
        normalized_class_labels,
        "EuroSAT zero-shot confusion matrix: simple prompt",
        output_paths["figures"] / "confusion_simple.png",
    )
    build_confusion_figure(
        y_true,
        ensemble_pred,
        normalized_class_labels,
        "EuroSAT zero-shot confusion matrix: ensemble prompts",
        output_paths["figures"] / "confusion_ensemble.png",
    )

    case_examples = select_case_examples(merged_df, limit_per_case=6)
    for case_type, case_df in case_examples.items():
        case_df.to_csv(output_paths["tables"] / f"{case_type}.csv", index=False)
        build_case_figure(
            case_df,
            case_type.replace("_", " "),
            output_paths["cases"] / f"{case_type}.png",
        )

    summary = compute_summary(
        merged_df=merged_df,
        classwise_df=classwise_df,
        boot_simple=boot_simple,
        boot_ensemble=boot_ensemble,
        backend=backend,
        test_size=args.test_size,
        output_dir=output_paths["root"],
    )
    summary["classifier_embedding_mean_abs_delta"] = classifier_delta

    with (output_paths["root"] / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    save_summary_markdown(summary, output_paths["root"] / "summary.md")

    quick_check = {
        "dataset_size": len(dataset),
        "test_size": len(test_subset),
        "num_classes": len(normalized_class_labels),
        "backend": backend.name,
        "classifier_embedding_mean_abs_delta": classifier_delta,
        "simple_classifier_shape": list(simple_classifier.shape),
        "ensemble_classifier_shape": list(ensemble_classifier.shape),
    }
    with (output_paths["root"] / "quick_check.json").open("w", encoding="utf-8") as handle:
        json.dump(quick_check, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
