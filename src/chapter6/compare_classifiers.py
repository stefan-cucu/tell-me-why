import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

MODELS = ["baseline", "robust"]
LABELS = {"baseline": "Baseline", "robust": "Robustified"}
COLORS = {"baseline": "#4C78A8", "robust": "#E45756"}
METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "malware_precision", "malware_recall", "roc_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced acc.",
    "macro_f1": "Macro-F1",
    "malware_precision": "Malware precision",
    "malware_recall": "Malware recall",
    "roc_auc": "ROC-AUC",
}


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        layers = []
        size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(size, hidden_size), nn.ReLU(), nn.Dropout(0.3)])
            size = hidden_size
        layers.append(nn.Linear(size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, embeddings):
        return self.network(embeddings)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--robust", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def natural_key(value):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state = {key.removeprefix("module."): value for key, value in state.items()}
    weights = sorted(
        [(key, value) for key, value in state.items() if key.endswith(".weight") and value.ndim == 2],
        key=lambda item: natural_key(item[0]),
    )
    if len(weights) < 2:
        raise ValueError(f"Cannot infer MLP architecture from {path}")
    model = Classifier(weights[0][1].shape[1], [value.shape[0] for _, value in weights[:-1]])
    model.load_state_dict(state)
    return model.to(device).eval()


def predict(model, vectors, device, batch_size=1024):
    scores = []
    with torch.no_grad():
        for start in range(0, len(vectors), batch_size):
            batch = torch.as_tensor(np.asarray(vectors[start:start + batch_size]), dtype=torch.float32, device=device)
            scores.append(torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(scores)


def choose_threshold(labels, scores):
    candidates = []
    for threshold in np.linspace(0.01, 0.99, 199):
        predictions = (scores >= threshold).astype(np.int64)
        candidates.append((f1_score(labels, predictions, average="macro"), accuracy_score(labels, predictions), threshold))
    return float(max(candidates)[2])


def metric_values(labels, scores, threshold):
    predictions = (scores >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "malware_precision": float(precision_score(labels, predictions, zero_division=0)),
        "malware_recall": float(recall_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def split_data(index, matrix, split):
    names = [split]
    if split == "validation":
        names.append("val")
    selected = index["split"].isin(names).to_numpy()
    metadata = index.loc[selected].reset_index(drop=True)
    vectors = np.asarray(matrix[selected], dtype=np.float32)
    labels = metadata["label"].map({"goodware": 0, "malware": 1}).to_numpy(dtype=np.int64)
    if not len(labels):
        raise ValueError(f"No {split} rows found in embedding index")
    return metadata, vectors, labels


def bootstrap(labels, scores, thresholds, families, samples, seed):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(families == family) for family in np.unique(families)]
    values = {model: {metric: [] for metric in METRICS} for model in MODELS}
    differences = {metric: [] for metric in METRICS}
    for _ in range(samples):
        selected = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        current = {}
        for model in MODELS:
            current[model] = metric_values(labels[selected], scores[model][selected], thresholds[model])
            for metric in METRICS:
                values[model][metric].append(current[model][metric])
        for metric in METRICS:
            differences[metric].append(current["robust"][metric] - current["baseline"][metric])
    return {
        "models": {
            model: {
                metric: {
                    "lower": float(np.quantile(values[model][metric], 0.025)),
                    "upper": float(np.quantile(values[model][metric], 0.975)),
                }
                for metric in METRICS
            }
            for model in MODELS
        },
        "difference": {
            metric: {
                "lower": float(np.quantile(differences[metric], 0.025)),
                "upper": float(np.quantile(differences[metric], 0.975)),
            }
            for metric in METRICS
        },
    }


def save_figure(figure, output, name):
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(output / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_metrics(metrics, output):
    positions = np.arange(len(METRICS))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9.0, 4.4))
    all_values = []
    for number, model in enumerate(MODELS):
        values = [metrics[model][metric] for metric in METRICS]
        all_values.extend(values)
        bars = axis.bar(positions + (number - 0.5) * width, values, width, label=LABELS[model], color=COLORS[model])
        axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    axis.set_xticks(positions, [METRIC_LABELS[name] for name in METRICS], rotation=18)
    axis.set_ylim(max(0, min(all_values) - 0.05), min(1.01, max(all_values) + 0.04))
    axis.set_ylabel("Score")
    axis.set_title("Classifier comparison")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    save_figure(figure, output, "classifier-metrics")


def plot_families(rows, output):
    frame = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(7.6, 4.2))
    sns.barplot(data=frame, x="family", y="accuracy", hue="model", hue_order=MODELS, palette=COLORS, ax=axis)
    minimum = max(0, frame["accuracy"].min() - 0.05)
    axis.set(ylim=(minimum, 1.01), xlabel="", ylabel="Accuracy")
    axis.set_title("Accuracy by family")
    handles, _ = axis.get_legend_handles_labels()
    axis.legend(handles, [LABELS[name] for name in MODELS], frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.18), loc="upper center")
    figure.tight_layout()
    save_figure(figure, output, "classifier-families")


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    embedding_dir = Path(args.embeddings)
    index = pd.read_csv(embedding_dir / "index.csv")
    matrix = np.load(embedding_dir / "embeddings.npy", mmap_mode="r")
    if len(index) != len(matrix):
        raise ValueError("Embedding index and matrix have different lengths")
    validation_meta, validation_vectors, validation_labels = split_data(index, matrix, "validation")
    test_meta, test_vectors, test_labels = split_data(index, matrix, "test")
    del validation_meta
    device = torch.device(args.device)
    models = {
        "baseline": load_model(args.baseline, device),
        "robust": load_model(args.robust, device),
    }
    scores = {"validation": {}, "test": {}}
    thresholds = {}
    for name, model in models.items():
        scores["validation"][name] = predict(model, validation_vectors, device)
        scores["test"][name] = predict(model, test_vectors, device)
        thresholds[name] = choose_threshold(validation_labels, scores["validation"][name])
    metrics = {}
    matrices = {}
    family_rows = []
    predictions = test_meta.copy()
    predictions["true_label"] = test_labels
    for name in MODELS:
        predicted = (scores["test"][name] >= thresholds[name]).astype(np.int64)
        metrics[name] = metric_values(test_labels, scores["test"][name], thresholds[name])
        matrices[name] = confusion_matrix(test_labels, predicted).tolist()
        predictions[f"{name}_malware_score"] = scores["test"][name]
        predictions[f"{name}_prediction"] = predicted
        for family in sorted(test_meta["family"].unique()):
            selected = test_meta["family"].eq(family).to_numpy()
            family_rows.append({
                "model": name,
                "family": family,
                "samples": int(selected.sum()),
                "accuracy": float(accuracy_score(test_labels[selected], predicted[selected])),
            })
    intervals = bootstrap(test_labels, scores["test"], thresholds, test_meta["family"].to_numpy(), args.bootstrap_samples, args.seed)
    result = {
        "validation_samples": len(validation_labels),
        "test_samples": len(test_labels),
        "thresholds": thresholds,
        "metrics": metrics,
        "confusion_matrices": matrices,
        "bootstrap_samples": args.bootstrap_samples,
        "confidence_intervals": intervals,
    }
    with (output / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    pd.DataFrame([
        {"model": name, "threshold": thresholds[name], **metrics[name]}
        for name in MODELS
    ]).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(family_rows).to_csv(output / "family_metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    sns.set_theme(style="whitegrid", context="paper")
    plot_metrics(metrics, output)
    plot_families(family_rows, output)
    for name in MODELS:
        print(f"{LABELS[name]}: accuracy={metrics[name]['accuracy']:.4f}, macro-F1={metrics[name]['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
