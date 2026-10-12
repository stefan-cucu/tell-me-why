import argparse
import json
import os
from pathlib import Path


CHAPTER_DIR = Path(__file__).resolve().parent
CLEAN_ROOT = CHAPTER_DIR.parents[1]
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "chapter4.json"

cache_dir = CLEAN_ROOT / ".cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch


POSITIVE_COLOR = "#E45756"
NEGATIVE_COLOR = "#4C78A8"


def load_settings(config_path=None):
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.is_absolute():
        path = CLEAN_ROOT / path
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def configure_plots():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )


def save_figure(figure, figure_dir, name):
    figure.tight_layout()
    for suffix in ["pdf", "png"]:
        figure.savefig(
            figure_dir / f"{name}.{suffix}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(figure)


def short_entry(category, value, limit=52):
    value = str(value)
    if len(value) > limit:
        value = "..." + value[-(limit - 3):]
    return f"{category.replace('_', ' ')}: {value}"


def plot_pipeline(figure_dir):
    figure, axis = plt.subplots(figsize=(10.0, 3.7))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 4)
    axis.axis("off")
    boxes = [
        (0.25, "JSON behavior\nreport", "#E8F1F8"),
        (2.75, "Qwen embedding\nmodel", "#F3E9D7"),
        (5.25, "1024-dimensional\nembedding", "#E7F2E5"),
        (7.75, "MLP malware\nprobability", "#F6E4E3"),
    ]
    for x_position, label, color in boxes:
        box = FancyBboxPatch(
            (x_position, 2.25),
            1.8,
            0.85,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            facecolor=color,
            edgecolor="#444444",
            linewidth=1.0,
        )
        axis.add_patch(box)
        axis.text(
            x_position + 0.9,
            2.675,
            label,
            ha="center",
            va="center",
        )
    for x_position in [2.1, 4.6, 7.1]:
        axis.annotate(
            "",
            xy=(x_position + 0.58, 2.68),
            xytext=(x_position, 2.68),
            arrowprops={
                "arrowstyle": "-|>",
                "linewidth": 1.5,
                "color": "#222222",
                "mutation_scale": 13,
            },
        )

    methods = [
        (1.15, "Category\nablation"),
        (3.65, "Entry-level\nKernel SHAP"),
        (6.15, "Entry-level\nIntegrated Gradients"),
    ]
    for x_position, label in methods:
        axis.text(
            x_position,
            0.72,
            label,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#777777",
            },
        )
    for start, end in [(1.15, 1.15), (3.65, 3.65), (6.15, 5.95)]:
        axis.annotate(
            "",
            xy=(end, 2.2),
            xytext=(start, 1.18),
            arrowprops={"arrowstyle": "->", "linewidth": 1.0},
        )
    save_figure(figure, figure_dir, "explanation-pipeline")


def plot_category_ablation(results_dir, figure_dir):
    frame = pd.read_csv(results_dir / "category_ablation.csv")
    statistics = (
        frame.groupby("category")["absolute_change"]
        .agg(
            reports="count",
            mean="mean",
            median="median",
            maximum="max",
        )
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    statistics.to_csv(
        results_dir / "category_ablation_statistics.csv",
        index=False,
    )

    plotted = statistics.sort_values("mean")
    labels = (
        plotted["category"]
        .str.replace("_", " ")
        .str.capitalize()
    )
    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    axis.barh(labels, plotted["mean"], color=NEGATIVE_COLOR)
    axis.set_xlabel("Mean absolute change in malware probability")
    axis.set_ylabel("Behavior category")
    axis.set_title("Category ablation")
    save_figure(figure, figure_dir, "importance")


def plot_shap(results_dir, figure_dir, top_entries):
    frame = pd.read_csv(results_dir / "shap_entries.csv")
    selected = frame.loc[
        frame["shap_value"].abs().nlargest(top_entries).index
    ].sort_values("shap_value")
    labels = [
        short_entry(category, value)
        for category, value in zip(
            selected["category"],
            selected["value"],
        )
    ]
    colors = [
        POSITIVE_COLOR if value > 0 else NEGATIVE_COLOR
        for value in selected["shap_value"]
    ]

    figure, axis = plt.subplots(figsize=(9.2, 5.8))
    axis.barh(labels, selected["shap_value"], color=colors)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_xlabel("SHAP value for malware probability")
    axis.set_title("Entry-level SHAP")
    axis.margins(x=0.08)
    save_figure(figure, figure_dir, "shap")


def plot_integrated_gradients(results_dir, figure_dir, top_entries):
    frame = pd.read_csv(
        results_dir / "integrated_gradients_entries.csv"
    )
    selected = frame.loc[
        frame["signed_sum"].abs().nlargest(top_entries).index
    ].sort_values("signed_sum")
    labels = [
        short_entry(category, value)
        for category, value in zip(
            selected["category"],
            selected["value"],
        )
    ]
    colors = [
        POSITIVE_COLOR if value > 0 else NEGATIVE_COLOR
        for value in selected["signed_sum"]
    ]

    figure, axis = plt.subplots(figsize=(9.2, 5.8))
    axis.barh(labels, selected["signed_sum"], color=colors)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_xlabel("Entry attribution (sum of token IG scores)")
    axis.set_title("Entry-level Integrated Gradients")
    axis.margins(x=0.08)
    save_figure(figure, figure_dir, "gradients")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the Chapter 4 explainability figures."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    settings = load_settings(args.config)
    if args.output_dir:
        settings["output_dir"] = args.output_dir

    output_dir = Path(settings["output_dir"])
    results_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    configure_plots()
    plot_pipeline(figure_dir)
    plot_category_ablation(results_dir, figure_dir)
    plot_shap(
        results_dir,
        figure_dir,
        int(settings["plot_top_entries"]),
    )
    plot_integrated_gradients(
        results_dir,
        figure_dir,
        int(settings["plot_top_entries"]),
    )
    print(f"Figures written to {figure_dir}")


if __name__ == "__main__":
    main()
