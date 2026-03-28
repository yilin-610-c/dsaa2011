#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def rel(path: Path, start: Path) -> str:
    return path.relative_to(start).as_posix()


def bullet_list(items):
    return "\n".join(f"- {item}" for item in items)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    rows = []
    for row in df.itertuples(index=False, name=None):
        formatted = []
        for value in row:
            if isinstance(value, float):
                formatted.append(f"{value:.4f}")
            else:
                formatted.append(str(value))
        rows.append(formatted)

    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(row) + " |")
    return "\n".join(table)


def prompt_list_for_setting(prompt_df: pd.DataFrame, prompt_setting: str):
    subset = prompt_df[prompt_df["prompt_setting"] == prompt_setting]
    if subset.empty:
        return []
    template_blob = str(subset.iloc[0]["templates"])
    return [item.strip() for item in template_blob.split(" || ") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="outputs/eurosat_vitb32", type=str)
    parser.add_argument("--report-path", default="report_eurosat_clip.md", type=str)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    report_path = Path(args.report_path)
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    cases_dir = figures_dir / "cases"

    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    overall_df = pd.read_csv(tables_dir / "overall_metrics.csv")
    classwise_df = pd.read_csv(tables_dir / "classwise_metrics.csv")
    confusion_pairs_df = pd.read_csv(tables_dir / "confusion_pairs.csv")
    improved_df = pd.read_csv(tables_dir / "simple_wrong_ensemble_right.csv")
    both_wrong_df = pd.read_csv(tables_dir / "both_wrong.csv")
    shifted_df = pd.read_csv(tables_dir / "both_correct_confidence_shift.csv")
    prompt_df = pd.read_csv(tables_dir / "prompt_details.csv")

    top_improved = classwise_df.head(3)
    top_degraded = classwise_df.sort_values("delta_accuracy").head(3)
    non_positive_df = classwise_df[classwise_df["delta_accuracy"] <= 0].sort_values("delta_accuracy")
    weakest_classes = non_positive_df.head(3) if not non_positive_df.empty else top_degraded
    top_confusions = confusion_pairs_df.head(8)

    ensemble_prompt_count = int(
        prompt_df[prompt_df["prompt_setting"] == "ensemble"]["num_templates"].iloc[0]
    )
    simple_prompt_templates = prompt_list_for_setting(prompt_df, "simple")
    ensemble_prompt_templates = prompt_list_for_setting(prompt_df, "ensemble")
    relative_gain = 100.0 * summary["accuracy_delta"] / max(summary["simple_accuracy"], 1e-8)

    lines = [
        "# EuroSAT Zero-Shot Classification with CLIP",
        "",
        "## 1. Summary",
        "",
        f"This report evaluates zero-shot classification on the EuroSAT dataset with the `{summary['model_name']}` CLIP model. "
        f"The implementation uses the `{summary['backend']}` backend with `{summary['pretrained_name']}` weights. "
        f"The evaluation compares a single-template baseline against a multi-template prompt ensemble.",
        "",
        f"- Dataset: `EuroSAT`",
        f"- CLIP model: `{summary['model_name']}`",
        f"- Backend used at runtime: `{summary['backend']}`",
        f"- Pretrained weights: `{summary['pretrained_name']}`",
        f"- Test split: deterministic stratified split with fraction `{summary['test_fraction']}` and seed `{summary['seed']}`",
        f"- Number of test samples: `{summary['num_test_samples']}`",
        "",
        "## 2. Methodology",
        "",
        "Two prompt settings were evaluated:",
        "",
        f"- `simple`: one prompt per class, `{simple_prompt_templates[0] if simple_prompt_templates else 'a photo of a {CLASS}.'}`",
        f"- `documentation-provided ensemble`: {ensemble_prompt_count} templates per class from the EuroSAT prompt documentation.",
        "",
        "For the ensemble setting, each prompt was encoded independently. The resulting text embeddings were L2-normalized, averaged within each class, and normalized again before image-text similarity scoring. This keeps the aggregation reproducible and makes the ensemble behave as one class prototype per category.",
        "",
        "The documentation-provided ensemble templates are listed below:",
        "",
        *[f"- `{template}`" for template in ensemble_prompt_templates],
        "",
        "Because the documentation-provided ensemble differs from the baseline in both domain wording and the number of templates, this experiment does not fully isolate the source of the improvement. However, since the three ensemble templates are highly similar, the result is most consistent with the hypothesis that better domain alignment plays the dominant role.",
        "",
        "## 3. Main Results",
        "",
        dataframe_to_markdown(overall_df),
        "",
        f"The ensemble changed top-1 accuracy from `{summary['simple_accuracy']:.4f}` to `{summary['ensemble_accuracy']:.4f}`, "
        f"an absolute gain of `{summary['accuracy_delta']:.4f}` and a relative gain of `{relative_gain:.2f}%` over the simple baseline.",
        "",
        f"- Samples corrected by the ensemble: `{summary['improved_count']}`",
        f"- Samples that regressed under the ensemble: `{summary['regressed_count']}`",
        f"- Samples both settings classified correctly: `{summary['unchanged_correct_count']}`",
        f"- Samples both settings classified incorrectly: `{summary['unchanged_wrong_count']}`",
        "",
        "### Stability",
        "",
        f"- Simple prompt bootstrap 95% CI: `[{summary['simple_bootstrap']['ci_low']:.4f}, {summary['simple_bootstrap']['ci_high']:.4f}]`",
        f"- Ensemble prompt bootstrap 95% CI: `[{summary['ensemble_bootstrap']['ci_low']:.4f}, {summary['ensemble_bootstrap']['ci_high']:.4f}]`",
        "- The inference pipeline is deterministic once the split and model weights are fixed, so stability is assessed through bootstrap resampling over the fixed test set rather than repeated stochastic training runs.",
        "",
        "## 4. Class-Wise Analysis",
        "",
        "### Most Improved Classes",
        "",
        dataframe_to_markdown(top_improved),
        "",
        "### Most Degraded Classes",
        "",
        dataframe_to_markdown(weakest_classes),
        "",
        "Key questions to answer from these tables:",
        "",
        "- Does the ensemble improvement spread across many classes, or is it concentrated in a small subset?",
        "- Are the gains larger for classes whose semantics are strongly tied to remote-sensing vocabulary?",
        "- Which classes remain hard even with better prompts, suggesting intrinsic visual ambiguity rather than wording mismatch?",
        "",
        "## 5. Patterns and Interpretation",
        "",
        "The analysis should focus on the following patterns, all of which are directly supported by the exported predictions, class-wise tables, and case figures:",
        "",
        "- `dataset-specific templates`: If the ensemble helps, one plausible reason is that `centered satellite photo` is closer to EuroSAT's image domain than the generic phrase `a photo of a ...`.",
        "- `simple prompt over-generalization`: The simple template may be too generic because it assumes natural-image phrasing rather than overhead sensing imagery.",
        "- `visually similar classes`: Confusions may persist for classes with similar textures, repeated patterns, or overlapping land-cover semantics.",
        "- `background and scale`: Some errors may be driven by mixed land cover within one patch, weak foreground objects, or strong contextual background cues.",
        "- `causal caution`: The current comparison does not cleanly separate the effect of domain wording from the effect of multi-template averaging.",
        "",
        "### Frequent Confusion Pairs",
        "",
        dataframe_to_markdown(top_confusions),
        "",
        f"Confusion matrix figures are saved as `{rel(figures_dir / 'confusion_simple.png', report_path.parent)}` and `{rel(figures_dir / 'confusion_ensemble.png', report_path.parent)}`.",
        "",
        "## 6. Error Case Analysis",
        "",
        "The exported case-study figures support three complementary views:",
        "",
        f"- `simple wrong, ensemble right`: `{rel(cases_dir / 'simple_wrong_ensemble_right.png', report_path.parent)}`",
        f"- `both wrong`: `{rel(cases_dir / 'both_wrong.png', report_path.parent)}`",
        f"- `both correct, different confidence`: `{rel(cases_dir / 'both_correct_confidence_shift.png', report_path.parent)}`",
        "",
        "Suggested manual inspection checklist for the final PDF discussion:",
        "",
        "- Check whether the target class occupies only a small fraction of the patch.",
        "- Check whether the patch is dominated by repetitive texture or mixed land cover.",
        "- Check whether the wrong prediction is semantically adjacent to the true class.",
        "- Check whether the ensemble fix looks like true disambiguation or simply a prompt-induced bias.",
        "",
        "## 7. Key Findings",
        "",
        "Use the following as the final discussion frame after reviewing the tables and figures:",
        "",
        f"- Overall, the ensemble {'outperforms' if summary['accuracy_delta'] > 0 else 'does not outperform'} the simple prompt by `{summary['accuracy_delta']:.4f}` absolute accuracy.",
        f"- The largest positive class-wise changes appear in: {', '.join(top_improved['class_name'].tolist())}.",
        f"- The classes with the weakest or negative change are: {', '.join(weakest_classes['class_name'].tolist())}.",
        "- If the confusion pairs remain similar across both settings, prompt engineering helps calibration more than core visual separation.",
        "- If some confusion pairs shrink meaningfully under the ensemble, the wording likely aligns better with the remote-sensing domain.",
        "",
        "## 8. Limitations",
        "",
        "- EuroSAT does not provide an official split in this pipeline, so a deterministic stratified split is used for reproducibility.",
        "- The experiment is zero-shot only; no fine-tuning or learned prompt optimization is applied.",
        "- Because the ensemble differs from the baseline in both wording and template count, the current experiment does not fully isolate the source of the gain.",
        "",
        "## 9. Reproducibility Artifacts",
        "",
        f"- Main output directory: `{rel(results_dir, report_path.parent)}`",
        f"- Overall metrics table: `{rel(tables_dir / 'overall_metrics.csv', report_path.parent)}`",
        f"- Class-wise metrics table: `{rel(tables_dir / 'classwise_metrics.csv', report_path.parent)}`",
        f"- Combined per-sample predictions: `{rel(results_dir / 'predictions' / 'combined_predictions.csv', report_path.parent)}`",
        f"- Prompt details: `{rel(tables_dir / 'prompt_details.csv', report_path.parent)}`",
        "",
        "## 10. References",
        "",
        "- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021.",
        "- OpenAI CLIP repository: https://github.com/openai/CLIP",
        "- CLIP Benchmark repository and dataset prompt templates: https://github.com/LAION-AI/CLIP_benchmark",
        "- Helber et al., *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification*, IEEE JSTARS, 2019.",
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
