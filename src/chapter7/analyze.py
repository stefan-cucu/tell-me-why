import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


COLORS = {
    "total": "#2878B5",
    "sft": "#2A9D8F",
    "aux": "#D95F02",
    "accuracy": "#4F759B",
    "frozen": "#7A7A7A",
    "gradient": "#7A5195",
    "gate": "#C43C39",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True)
    parser.add_argument("--trainer-state", required=True)
    parser.add_argument("--fusion-history", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smooth-window", type=int, default=75)
    return parser.parse_args()


def load_validation(path):
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("Validation file contains no records")
    return rows


def validation_summary(rows):
    total = len(rows)
    labels = ("benign", "malware")
    confusion = Counter((row["target_label"], row["predicted_label"]) for row in rows)
    count = lambda predicate: sum(bool(predicate(row)) for row in rows)
    correct = count(lambda row: row["is_correct"])
    raw_parse = count(lambda row: row.get("raw_parse_error") is None)
    reasoning_closed = count(lambda row: row.get("reasoning_closed_by_model"))
    nonempty_reasoning = count(lambda row: str(row.get("generated_reasoning", "")).strip())
    label_agreement = count(lambda row: row["predicted_label"] == row.get("classifier_label"))
    confidence_agreement = count(
        lambda row: abs(
            float(row.get("json_output", {}).get("confidence", -1))
            - float(row.get("classifier_confidence", -2))
        )
        <= 5e-4
    )
    reasoning_tokens = [int(row.get("reasoning_tokens", 0)) for row in rows]
    return {
        "evaluated_reports": total,
        "labels": list(labels),
        "confusion_matrix": [
            [confusion[(true, predicted)] for predicted in labels]
            for true in labels
        ],
        "accuracy": correct / total,
        "raw_parse_rate": raw_parse / total,
        "reasoning_closure_rate": reasoning_closed / total,
        "nonempty_reasoning_rate": nonempty_reasoning / total,
        "classifier_label_agreement_rate": label_agreement / total,
        "classifier_confidence_agreement_rate": confidence_agreement / total,
        "mean_reasoning_tokens": sum(reasoning_tokens) / total,
        "max_reasoning_tokens": max(reasoning_tokens),
    }


def load_training_history(trainer_state, fusion_history):
    with Path(trainer_state).open(encoding="utf-8") as handle:
        state = json.load(handle)
    trainer_rows = []
    for item in state.get("log_history", []):
        if "step" not in item or "loss" not in item:
            continue
        trainer_rows.append({
            "step": int(item["step"]),
            "total_loss": float(item["loss"]),
            "gradient_norm": float(item.get("grad_norm", 0)),
            "learning_rate": float(item.get("learning_rate", 0)),
        })
    trainer = pd.DataFrame(trainer_rows).drop_duplicates("step", keep="last")
    fusion = pd.read_csv(fusion_history)
    required = {
        "step",
        "sft_loss",
        "aux_loss",
        "aux_accuracy",
        "frozen_classifier_accuracy",
        "gate_layer_28",
        "gate_layer_35",
    }
    missing = required - set(fusion.columns)
    if missing:
        raise ValueError(f"Fusion history is missing columns: {sorted(missing)}")
    history = trainer.merge(fusion, on="step", how="left").sort_values("step")
    if history.empty:
        raise ValueError("No training history could be loaded")
    return history


def smooth(values, window):
    return values.rolling(window, center=True, min_periods=1).mean()


def plot_training(history, output, window):
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axis = axes[0, 0]
    axis.plot(history["step"], history["total_loss"], color=COLORS["total"], alpha=0.09, lw=0.5)
    axis.plot(history["step"], smooth(history["total_loss"], window), color=COLORS["total"], lw=1.8)
    axis.set(title="Total training loss", ylabel="Loss")

    fusion = history.dropna(subset=["sft_loss", "aux_loss"])
    axis = axes[0, 1]
    axis.plot(fusion["step"], smooth(fusion["sft_loss"], window), color=COLORS["sft"], lw=1.8, label="SFT")
    axis.plot(fusion["step"], smooth(fusion["aux_loss"].clip(lower=1e-5), window), color=COLORS["aux"], lw=1.8, label="Auxiliary")
    axis.set_yscale("log")
    axis.set(title="Training losses", ylabel="Loss")
    axis.legend(frameon=False)

    axis = axes[1, 0]
    axis.plot(fusion["step"], smooth(fusion["aux_accuracy"], window), color=COLORS["accuracy"], lw=1.8, label="Auxiliary head")
    axis.plot(fusion["step"], smooth(fusion["frozen_classifier_accuracy"], window), color=COLORS["frozen"], lw=1.4, ls="--", label="Frozen classifier")
    axis.set(title="Batch accuracy", xlabel="Training step", ylabel="Accuracy", ylim=(-0.02, 1.03))
    axis.legend(frameon=False, loc="lower right")

    axis = axes[1, 1]
    axis.plot(history["step"], smooth(history["gradient_norm"], window), color=COLORS["gradient"], lw=1.8, label="Gradient norm")
    axis.set(title="Gradient and gates", xlabel="Training step", ylabel="Gradient norm")
    gate_axis = axis.twinx()
    gate_axis.plot(fusion["step"], fusion["gate_layer_28"], color=COLORS["gate"], lw=1.5, label="Gate, layer 28")
    gate_axis.plot(fusion["step"], fusion["gate_layer_35"], color="#7A7A7A", lw=1.2, label="Gate, layer 35")
    gate_axis.set_ylabel("Sigmoid gate")
    lines = axis.get_lines() + gate_axis.get_lines()
    axis.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper right")

    figure.tight_layout()
    figure.savefig(output / "fusion-training.pdf", bbox_inches="tight")
    figure.savefig(output / "fusion-training.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    rows = load_validation(args.validation)
    summary = validation_summary(rows)
    with (output / "validation-summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    history = load_training_history(args.trainer_state, args.fusion_history)
    history.to_csv(output / "training-history.csv", index=False)
    plot_training(history, output, args.smooth_window)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
