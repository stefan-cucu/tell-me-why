import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"success": "#2A9D8F", "failure": "#7A7A7A", "accent": "#C43C39"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=5)
    return parser.parse_args()


def load_results(directory):
    rows = []
    for path in sorted(Path(directory).glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            rows.append(json.load(handle))
    if not rows:
        raise ValueError(f"No JSON results found in {directory}")
    return rows


def save(figure, output, name):
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def operation_count(diff):
    if not isinstance(diff, dict):
        return 0
    return sum(
        len(values)
        for operation in ("add", "edit")
        for values in diff.get(operation, {}).values()
        if isinstance(values, list)
    )


def trajectory(row, max_steps):
    values = [float(value) for value in row.get("score_progression", [])]
    if not values:
        values = [float(row.get("initial_score", 0))]
    values = values[: max_steps + 1]
    return values + [values[-1]] * (max_steps + 1 - len(values))


def plot_loop(output):
    figure, axis = plt.subplots(figsize=(8.2, 3.6))
    axis.axis("off")
    boxes = [
        (0.08, 0.65, "Current\nreport"),
        (0.31, 0.65, "Adversarial\nLLM"),
        (0.54, 0.65, "Candidate\ndiff"),
        (0.77, 0.65, "Classifier\nscore"),
        (0.54, 0.20, "Accept or\nrevert"),
    ]
    for x, y, label in boxes:
        axis.text(x, y, label, ha="center", va="center", transform=axis.transAxes, bbox={"boxstyle": "round,pad=0.5", "fc": "#F1F1F1", "ec": "#555555"})
    arrows = [
        ((0.15, 0.65), (0.24, 0.65)),
        ((0.38, 0.65), (0.47, 0.65)),
        ((0.61, 0.65), (0.70, 0.65)),
        ((0.77, 0.55), (0.61, 0.25)),
        ((0.47, 0.20), (0.14, 0.55)),
    ]
    for start, end in arrows:
        axis.annotate("", xy=end, xytext=start, xycoords=axis.transAxes, arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.5})
    axis.text(0.83, 0.20, "stop above 0.5", color=COLORS["success"], transform=axis.transAxes, ha="center")
    save(figure, output, "ch6-loop")


def plot_turns(rows, output, max_steps):
    counts = Counter()
    for row in rows:
        if row.get("evaded"):
            step = int(row.get("steps_taken", 0))
            if 1 <= step <= max_steps:
                counts[step] += 1
    new_successes = np.asarray([counts[step] for step in range(1, max_steps + 1)])
    cumulative = np.cumsum(new_successes) / len(rows) * 100
    turns = np.arange(1, max_steps + 1)
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))
    bars = axes[0].bar(turns, new_successes, color=COLORS["success"], width=0.62)
    axes[0].set(xlabel="Turn", ylabel="New evasions", xticks=turns)
    axes[0].set_title("Successful turn")
    axes[0].bar_label(bars, padding=2)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].plot(turns, cumulative, marker="o", color=COLORS["accent"], lw=2)
    axes[1].set(xlabel="Maximum turn", ylabel="Cumulative evasion rate (%)", xticks=turns, ylim=(0, 100))
    axes[1].set_title("Cumulative success")
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    save(figure, output, "ch6-turns")


def plot_trajectories(rows, output, max_steps):
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    turns = np.arange(max_steps + 1)
    for label, selected, color in [
        ("Successful", [row for row in rows if row.get("evaded")], COLORS["success"]),
        ("Unsuccessful", [row for row in rows if not row.get("evaded")], COLORS["failure"]),
    ]:
        if not selected:
            continue
        matrix = np.asarray([trajectory(row, max_steps) for row in selected])
        median = np.median(matrix, axis=0)
        lower = np.quantile(matrix, 0.25, axis=0)
        upper = np.quantile(matrix, 0.75, axis=0)
        axis.plot(turns, median, marker="o", color=color, lw=2, label=label)
        axis.fill_between(turns, lower, upper, color=color, alpha=0.16)
    axis.axhline(0.5, color=COLORS["accent"], ls="--", lw=1.2, label="Evasion threshold")
    axis.set(xlabel="Turn", ylabel="Goodware probability", xticks=turns, ylim=(0, 1))
    axis.set_title("Score progression")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    save(figure, output, "ch6-trajectories")


def plot_operations(rows, output):
    groups = [
        [operation_count(row.get("accumulated_diff")) for row in rows if row.get("evaded")],
        [operation_count(row.get("accumulated_diff")) for row in rows if not row.get("evaded")],
    ]
    figure, axis = plt.subplots(figsize=(6.8, 3.9))
    box = axis.boxplot(groups, tick_labels=["Successful", "Unsuccessful"], patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], [COLORS["success"], COLORS["failure"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axis.set_ylabel("Operations in final diff")
    axis.set_title("Perturbation size")
    axis.grid(axis="y", alpha=0.2)
    save(figure, output, "ch6-operations")


def main():
    args = parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    rows = load_results(args.results)
    output = Path(args.output)
    plot_loop(output)
    plot_turns(rows, output, args.max_steps)
    plot_trajectories(rows, output, args.max_steps)
    plot_operations(rows, output)
    successes = sum(bool(row.get("evaded")) for row in rows)
    print(f"Multi-turn evasion: {successes}/{len(rows)} ({100 * successes / len(rows):.2f}%)")


if __name__ == "__main__":
    main()
