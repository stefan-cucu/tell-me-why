import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import binomtest
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from common import (
    MLPClassifier,
    load_embeddings,
    load_settings,
    output_dir,
    set_seed,
    write_json,
)


DISPLAY_NAMES = {
    "majority": "Majority class",
    "logistic_regression": "Logistic regression",
    "decision_tree": "Decision tree",
    "xgboost": "XGBoost",
    "mlp": "MLP",
}


def expected_calibration_error(labels, malware_scores, bins=10):
    confidence = np.maximum(malware_scores, 1 - malware_scores)
    predictions = (malware_scores >= 0.5).astype(int)
    bin_edges = np.linspace(0, 1, bins + 1)
    error = 0.0

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            bin_accuracy = (predictions[selected] == labels[selected]).mean()
            error += selected.mean() * abs(
                bin_accuracy - confidence[selected].mean()
            )
    return float(error)


def calculate_metrics(labels, malware_scores, threshold):
    predictions = (malware_scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro")
        ),
        "malware_precision": float(
            precision_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "malware_recall": float(
            recall_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "goodware_precision": float(
            precision_score(labels, predictions, pos_label=0, zero_division=0)
        ),
        "goodware_recall": float(
            recall_score(labels, predictions, pos_label=0, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(labels, malware_scores)),
        "brier_score": float(brier_score_loss(labels, malware_scores)),
        "ece_10": expected_calibration_error(labels, malware_scores),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).tolist(),
    }


def bootstrap_intervals(
    labels,
    malware_scores,
    families,
    samples,
    seed,
    threshold,
):
    generator = np.random.default_rng(seed)
    family_rows = {
        family: np.flatnonzero(families == family)
        for family in np.unique(families)
    }
    sampled_metrics = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
    }

    for _ in range(samples):
        rows = np.concatenate(
            [
                generator.choice(indices, size=len(indices), replace=True)
                for indices in family_rows.values()
            ]
        )
        metrics = calculate_metrics(
            labels[rows],
            malware_scores[rows],
            threshold,
        )
        for metric_name in sampled_metrics:
            sampled_metrics[metric_name].append(metrics[metric_name])

    return {
        metric_name: {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
        }
        for metric_name, values in sampled_metrics.items()
    }


def load_mlp_scores(model_path, embeddings):
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )
    configuration = checkpoint["configuration"]
    model = MLPClassifier(
        input_dim=checkpoint["input_dim"],
        hidden_layers=tuple(configuration["hidden_layers"]),
        dropout=configuration["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.inference_mode():
        logits = model(torch.from_numpy(embeddings))
        return torch.softmax(logits, dim=1)[:, 1].numpy()


def load_model_scores(model_name, model_dir, embeddings):
    if model_name == "mlp":
        return load_mlp_scores(model_dir / "mlp.pt", embeddings)
    model = joblib.load(model_dir / f"{model_name}.joblib")
    return model.predict_proba(embeddings)[:, 1]


def configure_plots():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )


def save_figure(figure_dir, filename):
    plt.tight_layout()
    plt.savefig(figure_dir / filename, bbox_inches="tight")
    plt.close()


def plot_model_scores(metrics, figure_dir):
    model_names = [name for name in metrics if name != "majority"]
    values = [metrics[name]["macro_f1"] for name in model_names]
    lower = [
        metrics[name]["confidence_intervals"]["macro_f1"]["lower"]
        for name in model_names
    ]
    upper = [
        metrics[name]["confidence_intervals"]["macro_f1"]["upper"]
        for name in model_names
    ]
    errors = np.asarray(
        [
            [value - low for value, low in zip(values, lower)],
            [high - value for value, high in zip(values, upper)],
        ]
    )

    plt.figure(figsize=(7.0, 4.0))
    axis = plt.gca()
    bars = axis.bar(
        [DISPLAY_NAMES[name] for name in model_names],
        values,
        yerr=errors,
        capsize=4,
        color=sns.color_palette("deep", len(model_names)),
    )
    axis.set_ylim(max(0, min(values) - 0.05), 1.01)
    axis.set_ylabel("Test macro-F1")
    axis.set_title("Baseline classification performance")
    axis.tick_params(axis="x", rotation=20)

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    save_figure(figure_dir, "model_performance.pdf")


def plot_mlp_confusion(metrics, figure_dir):
    matrix = np.asarray(metrics["mlp"]["confusion_matrix"])
    plt.figure(figsize=(4.8, 4.0))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Goodware", "Malware"],
        yticklabels=["Goodware", "Malware"],
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("MLP confusion matrix")
    save_figure(figure_dir, "mlp_confusion_matrix.pdf")


def plot_family_scores(family_metrics, figure_dir):
    plot_data = family_metrics[family_metrics["model"] != "majority"].copy()
    plot_data["model"] = plot_data["model"].map(DISPLAY_NAMES)
    plot_data["family"] = plot_data["family"].str.capitalize()

    plt.figure(figsize=(7.5, 4.2))
    sns.barplot(
        data=plot_data,
        x="family",
        y="accuracy",
        hue="model",
    )
    plt.ylim(0, 1.03)
    plt.xlabel("")
    plt.ylabel("Accuracy")
    plt.title("Performance by class and malware family")
    plt.legend(loc="lower right", frameon=True, fontsize=8)
    save_figure(figure_dir, "family_performance.pdf")


def plot_mlp_history(metrics_dir, figure_dir):
    history = pd.read_csv(metrics_dir / "mlp_training_history.csv")
    plt.figure(figsize=(6.5, 3.8))
    plt.plot(
        history["epoch"],
        history["train_accuracy"],
        label="Training accuracy",
    )
    plt.plot(
        history["epoch"],
        history["validation_macro_f1"],
        label="Validation macro-F1",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1.02)
    plt.title("Selected MLP training history")
    plt.legend(frameon=True)
    save_figure(figure_dir, "mlp_training_curves.pdf")


def plot_mlp_calibration(labels, malware_scores, figure_dir):
    observed, predicted = calibration_curve(
        labels,
        malware_scores,
        n_bins=10,
        strategy="quantile",
    )
    plt.figure(figsize=(4.8, 4.0))
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#666666",
        label="Perfect calibration",
    )
    plt.plot(predicted, observed, marker="o", label="MLP")
    plt.xlabel("Mean predicted malware probability")
    plt.ylabel("Observed malware frequency")
    plt.title("MLP reliability diagram")
    plt.legend(frameon=True)
    save_figure(figure_dir, "mlp_calibration_curve.pdf")


def mcnemar_result(labels, scores, thresholds):
    if "xgboost" not in scores:
        return {
            "mlp_only_correct": 0,
            "xgboost_only_correct": 0,
            "discordant": 0,
            "p_value": 1.0,
        }

    mlp_correct = (
        (scores["mlp"] >= thresholds["mlp"]).astype(int) == labels
    )
    xgboost_correct = (
        (scores["xgboost"] >= thresholds["xgboost"]).astype(int) == labels
    )
    mlp_only = int((mlp_correct & ~xgboost_correct).sum())
    xgboost_only = int((~mlp_correct & xgboost_correct).sum())
    discordant = mlp_only + xgboost_only
    p_value = (
        float(binomtest(mlp_only, discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "mlp_only_correct": mlp_only,
        "xgboost_only_correct": xgboost_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the Chapter 3 classifiers on the test split."
    )
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    settings = load_settings(args.config)
    set_seed(settings["seed"])
    configure_plots()

    chapter_output = output_dir(settings)
    model_dir = chapter_output / "models"
    metrics_dir = chapter_output / "metrics"
    figure_dir = chapter_output / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    index, embedding_matrix = load_embeddings(settings)
    test_rows = index["split"].eq("test").to_numpy()
    test_embeddings = np.asarray(
        embedding_matrix[test_rows],
        dtype=np.float32,
    )
    reports = index.loc[test_rows].reset_index(drop=True)
    labels = (reports["label"] == "malware").astype(np.int64).to_numpy()

    with (metrics_dir / "selected_models.json").open(encoding="utf-8") as handle:
        selected_models = json.load(handle)

    model_names = [
        name
        for name in [
            "majority",
            "logistic_regression",
            "decision_tree",
            "xgboost",
            "mlp",
        ]
        if name in selected_models
        and (
            model_dir / (
                "mlp.pt" if name == "mlp" else f"{name}.joblib"
            )
        ).exists()
    ]
    thresholds = {
        name: float(selected_models[name]["decision_threshold"])
        for name in model_names
    }

    metrics = {}
    scores = {}
    predictions = reports.copy()
    family_results = []
    families = ["goodware"] + sorted(
        family
        for family in reports["family"].unique()
        if family != "goodware"
    )

    for model_number, model_name in enumerate(model_names):
        malware_scores = load_model_scores(
            model_name,
            model_dir,
            test_embeddings,
        )
        scores[model_name] = malware_scores
        threshold = thresholds[model_name]
        predicted_labels = (
            malware_scores >= threshold
        ).astype(np.int64)

        predictions[f"{model_name}_malware_score"] = malware_scores
        predictions[f"{model_name}_prediction"] = np.where(
            predicted_labels == 1,
            "malware",
            "goodware",
        )

        model_metrics = calculate_metrics(
            labels,
            malware_scores,
            threshold,
        )
        model_metrics["decision_threshold"] = threshold
        model_metrics["confidence_intervals"] = bootstrap_intervals(
            labels,
            malware_scores,
            reports["family"].to_numpy(),
            settings["evaluation"]["bootstrap_samples"],
            settings["seed"] + model_number,
            threshold,
        )
        metrics[model_name] = model_metrics

        for family in families:
            selected = reports["family"].eq(family).to_numpy()
            family_results.append(
                {
                    "model": model_name,
                    "family": family,
                    "samples": int(selected.sum()),
                    "correct": int(
                        (predicted_labels[selected] == labels[selected]).sum()
                    ),
                    "accuracy": float(
                        (
                            predicted_labels[selected]
                            == labels[selected]
                        ).mean()
                    ),
                }
            )

    family_metrics = pd.DataFrame(family_results)
    predictions.to_csv(metrics_dir / "test_predictions.csv", index=False)
    family_metrics.to_csv(metrics_dir / "family_metrics.csv", index=False)
    write_json(metrics_dir / "test_metrics.json", metrics)

    comparison = mcnemar_result(labels, scores, thresholds)
    write_json(metrics_dir / "mlp_xgboost_mcnemar.json", comparison)

    plot_model_scores(metrics, figure_dir)
    plot_mlp_confusion(metrics, figure_dir)
    plot_family_scores(family_metrics, figure_dir)
    plot_mlp_history(metrics_dir, figure_dir)
    plot_mlp_calibration(labels, scores["mlp"], figure_dir)

    ranking = sorted(
        metrics,
        key=lambda name: (
            metrics[name]["macro_f1"],
            metrics[name]["accuracy"],
        ),
        reverse=True,
    )
    print("Test ranking:")
    for model_name in ranking:
        print(
            model_name,
            metrics[model_name]["macro_f1"],
            metrics[model_name]["accuracy"],
        )
    print("MLP vs XGBoost:", comparison)


if __name__ == "__main__":
    main()
