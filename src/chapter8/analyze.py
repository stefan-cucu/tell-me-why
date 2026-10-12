import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from common import (
    classification_metrics,
    load_jsonl,
    load_report,
    normalize_label,
    parse_named_paths,
    prediction,
    target,
)


def arguments():
    parser = argparse.ArgumentParser(
        description="Create Chapter 8 metrics, plots, and qualitative extracts."
    )
    parser.add_argument("--prediction", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", required=True)
    parser.add_argument("--figures", required=True)
    parser.add_argument(
        "--example",
        action="append",
        default=[],
        metavar="NAME=REPORT_PATH",
        help="Write a Markdown comparison for a selected report.",
    )
    args = parser.parse_args()
    try:
        args.predictions = parse_named_paths(args.prediction)
        args.examples = parse_named_paths(args.example) if args.example else {}
    except ValueError as error:
        parser.error(str(error))
    return args


def output_metrics(records):
    total = len(records)
    direct_json = 0
    valid_label = 0
    classifier_agreement = []
    for record in records:
        parse_status = record.get("parse_status")
        if parse_status is not None:
            direct_json += parse_status in {"json", "json5", "strict_json"}
        elif "raw_parse_error" in record and record.get("raw_parse_error") is None:
            direct_json += 1
        valid_label += prediction(record) != "error"
        if record.get("classifier_label") is not None:
            classifier_agreement.append(
                prediction(record) == normalize_label(record["classifier_label"])
            )
    return {
        "direct_json_rate": direct_json / total if total else 0.0,
        "final_label_rate": valid_label / total if total else 0.0,
        "classifier_agreement_rate": (
            sum(classifier_agreement) / len(classifier_agreement)
            if classifier_agreement
            else None
        ),
    }


def common_records(models):
    common = set.intersection(*(set(records) for records in models.values()))
    common = sorted(
        path
        for path in common
        if all(target(records[path]) in {"malware", "benign"} for records in models.values())
    )
    if not common:
        raise ValueError("The prediction files have no shared labeled reports")
    return common


def write_metrics(metrics, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "classification-metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    with (output / "classification-metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "n",
                "accuracy",
                "macro_f1",
                "malware_precision",
                "malware_recall",
                "malware_f1",
                "benign_precision",
                "benign_recall",
                "benign_f1",
                "direct_json_rate",
                "final_label_rate",
                "classifier_agreement_rate",
            ]
        )
        for name, values in metrics.items():
            writer.writerow(
                [
                    name,
                    values["total"],
                    values["accuracy"],
                    values["macro_f1"],
                    values["per_class"]["malware"]["precision"],
                    values["per_class"]["malware"]["recall"],
                    values["per_class"]["malware"]["f1"],
                    values["per_class"]["benign"]["precision"],
                    values["per_class"]["benign"]["recall"],
                    values["per_class"]["benign"]["f1"],
                    values["output"]["direct_json_rate"],
                    values["output"]["final_label_rate"],
                    values["output"]["classifier_agreement_rate"],
                ]
            )


def save_figure(figure, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        figure.savefig(directory / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_metrics(metrics, figures):
    sns.set_theme(style="whitegrid", context="paper")
    names = list(metrics)
    series = {
        "Accuracy": [metrics[name]["accuracy"] for name in names],
        "Macro F1": [metrics[name]["macro_f1"] for name in names],
        "Malware F1": [metrics[name]["per_class"]["malware"]["f1"] for name in names],
        "Benign F1": [metrics[name]["per_class"]["benign"]["f1"] for name in names],
    }
    colors = ("#4F759B", "#2A9D8F", "#D28C45", "#8A6F9E")
    x = np.arange(len(names))
    width = 0.19
    figure, axis = plt.subplots(figsize=(9, 4.7))
    for index, ((label, values), color) in enumerate(zip(series.items(), colors)):
        positions = x + (index - 1.5) * width
        bars = axis.bar(positions, values, width, label=label, color=color)
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=2, fontsize=7)
    axis.set_xticks(x, names)
    axis.set_ylim(0, 1.11)
    axis.set_ylabel("Score")
    axis.legend(ncol=4, loc="upper center", frameon=False)
    axis.grid(axis="x", visible=False)
    figure.tight_layout()
    save_figure(figure, figures, "ch8-metrics")


def confusion_array(values):
    matrix = values["confusion_matrix"]
    return np.array(
        [
            [
                matrix[truth]["malware"],
                matrix[truth]["benign"],
                matrix[truth]["uncertain"] + matrix[truth]["error"],
            ]
            for truth in ("malware", "benign")
        ]
    )


def plot_confusions(metrics, figures):
    sns.set_theme(style="white", context="paper")
    columns = 2
    rows = math.ceil(len(metrics) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(8.6, 2.9 * rows), squeeze=False)
    for axis, (name, values) in zip(axes.flat, metrics.items()):
        sns.heatmap(
            confusion_array(values),
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            xticklabels=("Malware", "Benign", "Error"),
            yticklabels=("Malware", "Benign"),
            ax=axis,
        )
        axis.set_title(name, fontsize=10)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.tick_params(axis="x", rotation=20)
        axis.tick_params(axis="y", rotation=0)
    for axis in axes.flat[len(metrics) :]:
        axis.remove()
    figure.tight_layout()
    save_figure(figure, figures, "ch8-confusion")


def formatted_output(record):
    parts = []
    reasoning = str(record.get("generated_reasoning") or "").strip()
    if reasoning:
        parts.append(f"<think>\n{reasoning}\n</think>")
    parts.append(json.dumps(record.get("json_output") or {}, indent=2, ensure_ascii=False))
    return "\n".join(parts)


def write_examples(examples, models, output):
    if not examples:
        return
    directory = output / "examples"
    directory.mkdir(parents=True, exist_ok=True)
    for name, path in examples.items():
        report_path = path.resolve()
        key = str(report_path)
        missing = [model_name for model_name, records in models.items() if key not in records]
        if missing:
            raise ValueError(f"Example {report_path} is absent from: {', '.join(missing)}")
        first = next(iter(models.values()))[key]
        lines = [
            f"# {name}",
            "",
            f"Ground truth: `{target(first)}`",
            "",
            "## Input Log",
            "",
            "```json",
            json.dumps(load_report(report_path), indent=2, ensure_ascii=False),
            "```",
        ]
        for model_name, records in models.items():
            record = records[key]
            lines.extend(
                [
                    "",
                    f"## {model_name}",
                    "",
                    f"Predicted label: `{prediction(record)}`",
                    "",
                    "```text",
                    formatted_output(record),
                    "```",
                ]
            )
        (directory / f"{name}.md").write_text("\n".join(lines) + "\n")


def main():
    args = arguments()
    models = {name: load_jsonl(path) for name, path in args.predictions.items()}
    paths = common_records(models)
    metrics = {}
    for name, records in models.items():
        selected = [records[path] for path in paths]
        values = classification_metrics(selected)
        values["output"] = output_metrics(selected)
        metrics[name] = values
    output = Path(args.output)
    figures = Path(args.figures)
    write_metrics(metrics, output)
    plot_metrics(metrics, figures)
    plot_confusions(metrics, figures)
    write_examples(args.examples, models, output)
    print(f"Shared cohort: {len(paths)} reports")
    for name, values in metrics.items():
        print(f"{name}: accuracy={values['accuracy']:.4f}, macro_f1={values['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
