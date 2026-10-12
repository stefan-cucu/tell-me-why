import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "sft": "#2878B5",
    "grpo": "#D95F02",
    "success": "#2A9D8F",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-results", required=True)
    parser.add_argument("--grpo-results", required=True)
    parser.add_argument("--trainer-state", required=True)
    parser.add_argument("--output", required=True)
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


def smooth(values, window=75):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="same")


def additions(row):
    if not row.get("evaded") or not isinstance(row.get("evaded_data"), dict):
        return Counter()
    original = row.get("original", {})
    changed = row["evaded_data"]
    original = original.get("summary", original)
    changed = changed.get("summary", changed)
    counts = Counter()
    for category in set(original) | set(changed):
        before = original.get(category, [])
        after = changed.get(category, [])
        if isinstance(before, list) and isinstance(after, list):
            counts[category] += sum((Counter(map(str, after)) - Counter(map(str, before))).values())
    return counts


def plot_comparison(sft, grpo, output):
    rates = [
        100 * sum(bool(row.get("evaded")) for row in rows) / len(rows)
        for rows in (sft, grpo)
    ]
    figure, axis = plt.subplots(figsize=(6.5, 3.7))
    bars = axis.bar(["SFT", "SFT + GRPO"], rates, color=[COLORS["sft"], COLORS["grpo"]], width=0.55)
    axis.set_ylabel("Evasion rate (%)")
    axis.set_ylim(0, max(rates) * 1.2)
    axis.set_title("Single-turn evasion")
    axis.grid(axis="y", alpha=0.2)
    axis.bar_label(bars, labels=[f"{value:.2f}%" for value in rates], padding=3)
    save(figure, output, "ch5-comparison")


def plot_scores(sft, grpo, output):
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    for label, rows, color in [
        ("SFT", sft, COLORS["sft"]),
        ("SFT + GRPO", grpo, COLORS["grpo"]),
    ]:
        values = np.sort([float(row["best_goodware_prob"]) for row in rows])
        axis.plot(values, np.arange(1, len(values) + 1) / len(values), color=color, lw=2, label=label)
    axis.axvline(0.5, color="#333333", ls="--", lw=1.2, label="Evasion threshold")
    axis.set(xlabel="Best goodware probability", ylabel="Cumulative share", xlim=(0, 1), ylim=(0, 1))
    axis.set_title("Classifier scores")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    save(figure, output, "ch5-scores")


def plot_categories(sft, grpo, output):
    normalized = []
    for rows in (sft, grpo):
        counts = Counter()
        for row in rows:
            counts.update(additions(row))
        successes = max(1, sum(bool(row.get("evaded")) for row in rows))
        normalized.append({name: count / successes for name, count in counts.items()})
    combined = Counter()
    for counts in normalized:
        combined.update(counts)
    categories = [name for name, _ in combined.most_common(8)][::-1]
    positions = np.arange(len(categories))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.barh(positions - width / 2, [normalized[0].get(name, 0) for name in categories], height=width, color=COLORS["sft"], label="SFT")
    axis.barh(positions + width / 2, [normalized[1].get(name, 0) for name in categories], height=width, color=COLORS["grpo"], label="SFT + GRPO")
    axis.set_yticks(positions, [name.replace("_", " ") for name in categories])
    axis.set_xlabel("Additions per successful report")
    axis.set_title("Changed categories")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False)
    save(figure, output, "ch5-categories")


def plot_training(state_path, output):
    with Path(state_path).open(encoding="utf-8") as handle:
        history = json.load(handle).get("log_history", [])
    rows = [row for row in history if "reward" in row and "kl" in row]
    if not rows:
        raise ValueError("The trainer state contains no reward and KL entries")
    steps = np.asarray([row["step"] for row in rows])
    rewards = np.asarray([row["reward"] for row in rows])
    kls = np.asarray([row["kl"] for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    for axis, values, title, ylabel, color in [
        (axes[0], rewards, "GRPO reward", "Reward", COLORS["grpo"]),
        (axes[1], kls, "KL divergence", "KL", COLORS["sft"]),
    ]:
        axis.plot(steps, values, color=color, alpha=0.15, lw=0.7)
        axis.plot(steps, smooth(values), color=color, lw=1.8)
        axis.set(xlabel="Training step", ylabel=ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    save(figure, output, "ch5-grpo")


def main():
    args = parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    output = Path(args.output)
    sft = load_results(args.sft_results)
    grpo = load_results(args.grpo_results)
    plot_comparison(sft, grpo, output)
    plot_scores(sft, grpo, output)
    plot_categories(sft, grpo, output)
    plot_training(args.trainer_state, output)
    print(f"SFT: {sum(bool(row.get('evaded')) for row in sft)}/{len(sft)}")
    print(f"SFT + GRPO: {sum(bool(row.get('evaded')) for row in grpo)}/{len(grpo)}")


if __name__ == "__main__":
    main()
