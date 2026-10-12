import argparse
import copy
import json
import os
import random
from collections import defaultdict
from pathlib import Path


CHAPTER_DIR = Path(__file__).resolve().parent
CLEAN_ROOT = CHAPTER_DIR.parents[1]
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "chapter4.json"

cache_dir = CLEAN_ROOT / ".cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.nn.functional as functional
from captum.attr import LayerIntegratedGradients
from transformers import AutoModel, AutoTokenizer


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout):
        super().__init__()
        layers = []
        previous_size = input_dim
        for layer_size in hidden_layers:
            layers.extend(
                [
                    nn.Linear(previous_size, layer_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_size = layer_size
        layers.append(nn.Linear(previous_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, embeddings):
        return self.network(embeddings)


class FullPipeline(nn.Module):
    def __init__(self, embedding_model, classifier):
        super().__init__()
        self.embedding_model = embedding_model
        self.classifier = classifier

    def forward(self, input_ids, attention_mask):
        output = self.embedding_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        pooled = last_token_pool(output.last_hidden_state, attention_mask)
        pooled = functional.normalize(pooled, p=2, dim=1)
        return self.classifier(pooled.float())


class ReportEmbedder:
    def __init__(self, settings, device):
        model_path = settings["embedding_model_snapshot"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            padding_side="left",
            use_fast=True,
        )
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(device)
        self.model.eval()
        self.model.config.use_cache = False
        self.device = device
        self.embedding_size = int(self.model.config.hidden_size)

    def encode_batch(self, texts):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=False,
            add_special_tokens=True,
            return_tensors="pt",
        )
        encoded = {
            name: tensor.to(self.device)
            for name, tensor in encoded.items()
        }
        with torch.inference_mode():
            output = self.model(**encoded, use_cache=False)
            embeddings = last_token_pool(
                output.last_hidden_state,
                encoded["attention_mask"],
            )
            embeddings = functional.normalize(embeddings, p=2, dim=1)
        return embeddings.float().cpu().numpy()

    def encode(self, texts, token_budget, max_batch_size):
        token_counts = [
            len(self.tokenizer(text, add_special_tokens=True)["input_ids"])
            for text in texts
        ]
        vectors = np.zeros(
            (len(texts), self.embedding_size),
            dtype=np.float32,
        )
        order = np.argsort(token_counts)
        batch = []
        longest = 0

        def flush(records):
            if not records:
                return
            positions = [position for position, _ in records]
            batch_texts = [texts[position] for position in positions]
            try:
                vectors[positions] = self.encode_batch(batch_texts)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(records) == 1:
                    raise
                middle = len(records) // 2
                flush(records[:middle])
                flush(records[middle:])

        for position in order:
            token_count = token_counts[position]
            next_longest = max(longest, token_count)
            next_size = len(batch) + 1
            if batch and (
                next_size > max_batch_size
                or next_longest * next_size > token_budget
            ):
                flush(batch)
                batch = []
                longest = 0
            batch.append((int(position), token_count))
            longest = max(longest, token_count)
        flush(batch)
        return vectors


def load_settings(config_path=None):
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.is_absolute():
        path = CLEAN_ROOT / path
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def load_report(path):
    with Path(path).open(encoding="utf-8") as handle:
        raw_report = json.load(handle)
    report = raw_report.get("summary", raw_report)
    if not isinstance(report, dict):
        raise ValueError(f"Report {path} does not contain a JSON object")
    return report


def absolute_report_path(settings, path):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(settings["source_root"]) / path


def serialize_report(report):
    return json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def serialize_report_with_spans(report):
    parts = []
    spans = []
    cursor = 0

    def append(text):
        nonlocal cursor
        parts.append(text)
        cursor += len(text)

    append("{")
    for category_index, category in enumerate(sorted(report)):
        if category_index:
            append(",")
        append(json.dumps(category, ensure_ascii=False))
        append(":")
        values = report[category]
        if not isinstance(values, list):
            raise TypeError(
                f"Expected a list in category {category!r}, "
                f"found {type(values).__name__}"
            )
        append("[")
        for position, value in enumerate(values):
            if position:
                append(",")
            encoded_value = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            encoded_start = cursor
            append(encoded_value)
            if isinstance(value, str):
                value_start = encoded_start + 1
                value_end = cursor - 1
            else:
                value_start = encoded_start
                value_end = cursor
            spans.append(
                {
                    "category": category,
                    "position": position,
                    "value": str(value),
                    "start": value_start,
                    "end": value_end,
                }
            )
        append("]")
    append("}")
    return "".join(parts), spans


def last_token_pool(hidden_states, attention_mask):
    uses_left_padding = bool(
        (attention_mask[:, -1].sum() == attention_mask.shape[0]).item()
    )
    if uses_left_padding:
        return hidden_states[:, -1]
    last_token = attention_mask.sum(dim=1) - 1
    batch = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch, last_token]


def load_chapter3_artifacts(settings):
    chapter3_dir = Path(settings["chapter3_output_dir"])
    index = pd.read_csv(chapter3_dir / "embeddings" / "index.csv")
    matrix = np.load(
        chapter3_dir / "embeddings" / "embeddings.npy",
        mmap_mode="r",
    )
    if len(index) != len(matrix):
        raise RuntimeError("Embedding index and matrix have different lengths")
    return index, matrix


def load_classifier(settings, device):
    checkpoint_path = (
        Path(settings["chapter3_output_dir"]) / "models" / "mlp.pt"
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    configuration = checkpoint["configuration"]
    hidden_layers = configuration["hidden_layers"]
    model = MLPClassifier(
        input_dim=int(checkpoint["input_dim"]),
        hidden_layers=tuple(hidden_layers),
        dropout=float(configuration["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def load_threshold(settings):
    metrics_dir = Path(settings["chapter3_output_dir"]) / "metrics"
    metrics_path = metrics_dir / "selected_models.json"
    with metrics_path.open(encoding="utf-8") as handle:
        selected_models = json.load(handle)
    return float(selected_models["mlp"]["decision_threshold"])


def malware_probabilities(model, vectors):
    device = next(model.parameters()).device
    with torch.inference_mode():
        embeddings = torch.as_tensor(
            vectors,
            dtype=torch.float32,
            device=device,
        )
        return (
            torch.softmax(model(embeddings), dim=1)[:, 1]
            .cpu()
            .numpy()
        )


def load_test_predictions(settings, classifier):
    index, matrix = load_chapter3_artifacts(settings)
    test_rows = index["split"].eq("test").to_numpy()
    reports = index.loc[test_rows].copy().reset_index(drop=True)
    test_vectors = np.asarray(matrix[test_rows], dtype=np.float32)
    scores = malware_probabilities(classifier, test_vectors)
    threshold = load_threshold(settings)
    reports["malware_probability"] = scores
    reports["prediction"] = np.where(
        scores >= threshold,
        "malware",
        "goodware",
    )
    return index, reports


def select_ablation_samples(reports, settings):
    configuration = settings["category_ablation"]
    candidates = reports[
        reports["prediction"].eq(reports["label"])
        & reports["token_count"].le(configuration["max_tokens"])
    ]
    selected = []
    for family in configuration["families"]:
        family_reports = candidates[candidates["family"].eq(family)]
        count = min(
            int(configuration["samples_per_family"]),
            len(family_reports),
        )
        if count:
            selected.append(
                family_reports.sample(count, random_state=settings["seed"])
            )
    if not selected:
        raise RuntimeError("No reports matched the category-ablation settings")
    return pd.concat(selected).sort_values(["family", "sample_id"])


def run_category_ablation(
    settings,
    embedder,
    classifier,
    selected,
    results_dir,
):
    embedding_settings = settings["embedding"]
    rows = []
    for sample in selected.to_dict(orient="records"):
        report = load_report(
            absolute_report_path(settings, sample["path"])
        )
        variants = []
        categories = []
        for category in settings["categories"]:
            if report.get(category):
                ablated_report = copy.deepcopy(report)
                ablated_report[category] = []
                variants.append(serialize_report(ablated_report))
                categories.append(category)

        vectors = embedder.encode(
            variants,
            token_budget=int(embedding_settings["token_budget"]),
            max_batch_size=int(embedding_settings["max_batch_size"]),
        )
        scores = malware_probabilities(classifier, vectors)
        original_score = float(sample["malware_probability"])
        for category, score in zip(categories, scores):
            signed_change = original_score - float(score)
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "family": sample["family"],
                    "label": sample["label"],
                    "category": category,
                    "original_malware_probability": original_score,
                    "ablated_malware_probability": float(score),
                    "signed_change": signed_change,
                    "absolute_change": abs(signed_change),
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(results_dir / "category_ablation.csv", index=False)
    return frame


def select_local_sample(reports, settings, method):
    configuration = settings[method]
    sample_id = configuration.get("sample_id")
    if sample_id:
        matches = reports[reports["sample_id"].eq(sample_id)]
        if len(matches) != 1:
            raise RuntimeError(
                f"Configured {method} sample {sample_id!r} was not found "
                "exactly once in the test split"
            )
        return matches.iloc[0]

    candidates = reports[
        reports["family"].eq("virlock")
        & reports["prediction"].eq(reports["label"])
        & reports["token_count"].le(configuration["max_tokens"])
    ].sort_values("sample_id")
    if candidates.empty:
        raise RuntimeError(f"No suitable report found for {method}")
    return candidates.iloc[settings["seed"] % len(candidates)]


def choose_entries(report, categories, maximum):
    entries = []
    for category in categories:
        for position, value in enumerate(report.get(category, [])):
            entries.append(
                {
                    "category": category,
                    "position": position,
                    "value": str(value),
                }
            )
    if len(entries) <= maximum:
        return entries

    entries_by_category = defaultdict(list)
    for entry in entries:
        entries_by_category[entry["category"]].append(entry)

    selected = []
    while entries_by_category and len(selected) < maximum:
        for category in list(entries_by_category):
            selected.append(entries_by_category[category].pop(0))
            if not entries_by_category[category]:
                del entries_by_category[category]
            if len(selected) == maximum:
                break
    return selected


def apply_entry_mask(report, entries, mask):
    masked_report = copy.deepcopy(report)
    removals = defaultdict(list)
    for keep, entry in zip(mask, entries):
        if keep < 0.5:
            removals[entry["category"]].append(entry["position"])
    for category, positions in removals.items():
        for position in sorted(positions, reverse=True):
            del masked_report[category][position]
    return masked_report


def run_kernel_shap(
    settings,
    embedder,
    classifier,
    sample,
    results_dir,
    coalitions,
):
    configuration = settings["shap"]
    report = load_report(absolute_report_path(settings, sample["path"]))
    entries = choose_entries(
        report,
        settings["categories"],
        int(configuration["max_entries"]),
    )
    if not entries:
        raise RuntimeError("The selected SHAP report has no entries")

    def predict(masks):
        texts = [
            serialize_report(apply_entry_mask(report, entries, mask))
            for mask in np.atleast_2d(masks)
        ]
        vectors = embedder.encode(
            texts,
            token_budget=int(configuration["embedding_token_budget"]),
            max_batch_size=int(configuration["embedding_batch_size"]),
        )
        return malware_probabilities(classifier, vectors)

    background = np.zeros((1, len(entries)), dtype=np.float32)
    explained = np.ones((1, len(entries)), dtype=np.float32)
    explainer = shap.KernelExplainer(predict, background)
    values = np.asarray(
        explainer.shap_values(
            explained,
            nsamples=coalitions,
            silent=True,
        )
    ).reshape(-1)

    frame = pd.DataFrame(
        [
            {**entry, "shap_value": float(value)}
            for entry, value in zip(entries, values)
        ]
    )
    frame = frame.loc[
        frame["shap_value"].abs().sort_values(ascending=False).index
    ]
    frame.to_csv(results_dir / "shap_entries.csv", index=False)
    write_sample(results_dir / "shap_sample.json", sample)
    return frame


def aggregate_entry_attributions(spans, offsets, token_scores):
    rows = []
    for entry in spans:
        signed_sum = 0.0
        absolute_sum = 0.0
        contributing_tokens = 0
        for token_score, (token_start, token_end) in zip(
            token_scores,
            offsets,
        ):
            if token_end <= token_start:
                continue
            overlap = max(
                0,
                min(token_end, entry["end"])
                - max(token_start, entry["start"]),
            )
            if not overlap:
                continue
            overlap_fraction = overlap / (token_end - token_start)
            signed_sum += float(token_score) * overlap_fraction
            absolute_sum += abs(float(token_score)) * overlap_fraction
            contributing_tokens += 1
        rows.append(
            {
                "category": entry["category"],
                "position": entry["position"],
                "value": entry["value"],
                "signed_sum": signed_sum,
                "absolute_sum": absolute_sum,
                "contributing_tokens": contributing_tokens,
            }
        )

    frame = pd.DataFrame(rows)
    return frame.loc[
        frame["signed_sum"].abs().sort_values(ascending=False).index
    ]


def run_integrated_gradients(
    settings,
    embedder,
    classifier,
    sample,
    results_dir,
    steps,
):
    report = load_report(absolute_report_path(settings, sample["path"]))
    text, spans = serialize_report_with_spans(report)
    if text != serialize_report(report):
        raise RuntimeError("Tracked serialization differs from model input")

    encoded = embedder.tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping").squeeze(0).tolist()
    input_ids = encoded["input_ids"].to(embedder.device)
    attention_mask = encoded["attention_mask"].to(embedder.device)
    baseline_token = embedder.tokenizer.pad_token_id
    if baseline_token is None:
        baseline_token = embedder.tokenizer.eos_token_id
    baseline_ids = torch.full_like(input_ids, baseline_token)

    embedder.model.train()
    for parameter in embedder.model.parameters():
        parameter.requires_grad_(True)

    pipeline = FullPipeline(embedder.model, classifier).to(embedder.device)
    method = LayerIntegratedGradients(
        pipeline,
        embedder.model.embed_tokens,
    )
    attributions = method.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=1,
        n_steps=steps,
        internal_batch_size=1,
    )
    token_scores = (
        attributions.sum(dim=-1)
        .squeeze(0)
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    token_ids = input_ids.squeeze(0).detach().cpu().tolist()
    tokens = embedder.tokenizer.convert_ids_to_tokens(token_ids)
    special_ids = set(embedder.tokenizer.all_special_ids)

    token_frame = pd.DataFrame(
        [
            {
                "position": position,
                "token": token,
                "start": int(offsets[position][0]),
                "end": int(offsets[position][1]),
                "attribution": float(token_scores[position]),
            }
            for position, (token_id, token) in enumerate(
                zip(token_ids, tokens)
            )
            if token_id not in special_ids
        ]
    )
    token_frame.to_csv(
        results_dir / "integrated_gradients_tokens.csv",
        index=False,
    )

    entry_frame = aggregate_entry_attributions(
        spans,
        offsets,
        token_scores,
    )
    entry_frame.to_csv(
        results_dir / "integrated_gradients_entries.csv",
        index=False,
    )
    write_sample(
        results_dir / "integrated_gradients_sample.json",
        sample,
    )

    non_special_scores = np.asarray(
        [
            score
            for token_id, score in zip(token_ids, token_scores)
            if token_id not in special_ids
        ]
    )
    assigned_signed = float(entry_frame["signed_sum"].sum())
    assigned_absolute = float(entry_frame["absolute_sum"].sum())
    write_json(
        results_dir / "integrated_gradients_metadata.json",
        {
            "steps": steps,
            "serialized_characters": len(text),
            "tokenizer_tokens": len(token_ids),
            "non_special_tokens": len(non_special_scores),
            "behavioral_entries": len(entry_frame),
            "entry_signed_sum": assigned_signed,
            "entry_absolute_sum": assigned_absolute,
            "excluded_signed_sum": (
                float(non_special_scores.sum()) - assigned_signed
            ),
            "excluded_absolute_sum": (
                float(np.abs(non_special_scores).sum())
                - assigned_absolute
            ),
        },
    )
    embedder.model.eval()
    return entry_frame


def write_sample(path, sample):
    write_json(
        path,
        {
            "sample_id": sample["sample_id"],
            "path": sample["path"],
            "family": sample["family"],
            "token_count": int(sample["token_count"]),
            "malware_probability": float(sample["malware_probability"]),
        },
    )


def run_prevalence(settings, index, shap_frame, results_dir):
    top_entries = shap_frame.head(10)
    entries = [
        (row["category"], row["value"])
        for row in top_entries.to_dict(orient="records")
    ]
    counts = {
        entry: {"goodware": 0, "malware": 0}
        for entry in entries
    }
    reports = index[index["split"].isin(settings["prevalence_splits"])]
    totals = reports["label"].value_counts().to_dict()

    for sample in reports.to_dict(orient="records"):
        report = load_report(
            absolute_report_path(settings, sample["path"])
        )
        report_entries = {
            category: {str(value) for value in report.get(category, [])}
            for category in settings["categories"]
        }
        for category, value in entries:
            if value in report_entries.get(category, set()):
                counts[(category, value)][sample["label"]] += 1

    shap_values = {
        (row["category"], row["value"]): float(row["shap_value"])
        for row in top_entries.to_dict(orient="records")
    }
    rows = []
    for (category, value), label_counts in counts.items():
        rows.append(
            {
                "category": category,
                "value": value,
                "shap_value": shap_values[(category, value)],
                "malware_prevalence": (
                    label_counts["malware"] / totals["malware"]
                ),
                "goodware_prevalence": (
                    label_counts["goodware"] / totals["goodware"]
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(
        results_dir / "shap_entry_prevalence.csv",
        index=False,
    )
    return frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Chapter 4 explainability experiments."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--chapter3-output", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["ablation", "shap", "ig", "prevalence"],
        default=["ablation", "shap", "ig", "prevalence"],
    )
    parser.add_argument("--max-ablation-samples", type=int, default=None)
    parser.add_argument("--shap-coalitions", type=int, default=None)
    parser.add_argument("--ig-steps", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    settings = load_settings(args.config)
    if args.chapter3_output:
        settings["chapter3_output_dir"] = args.chapter3_output
    if args.output_dir:
        settings["output_dir"] = args.output_dir
    if args.device:
        settings["device"] = args.device

    set_seed(int(settings["seed"]))
    results_dir = Path(settings["output_dir"]) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(settings["device"])

    model_stages = {"ablation", "shap", "ig"}
    needs_model = bool(model_stages.intersection(args.stages))
    classifier = load_classifier(settings, device) if needs_model else None
    index = None
    reports = None
    embedder = None
    if needs_model:
        index, reports = load_test_predictions(settings, classifier)
        embedder = ReportEmbedder(settings, device)

    summary = {}
    if "ablation" in args.stages:
        selected = select_ablation_samples(reports, settings)
        if args.max_ablation_samples is not None:
            selected = selected.head(args.max_ablation_samples)
        selected.to_csv(
            results_dir / "ablation_samples.csv",
            index=False,
        )
        frame = run_category_ablation(
            settings,
            embedder,
            classifier,
            selected,
            results_dir,
        )
        summary["ablation_reports"] = int(frame["sample_id"].nunique())
        summary["ablation_combinations"] = len(frame)
        print(
            "Category ablation: "
            f"{len(frame)} report-category combinations"
        )

    if "shap" in args.stages:
        sample = select_local_sample(reports, settings, "shap")
        coalitions = (
            args.shap_coalitions
            or int(settings["shap"]["coalitions"])
        )
        frame = run_kernel_shap(
            settings,
            embedder,
            classifier,
            sample,
            results_dir,
            coalitions,
        )
        summary["shap_entries"] = len(frame)
        summary["shap_coalitions"] = coalitions
        print(f"Kernel SHAP: {len(frame)} entries")

    if "ig" in args.stages:
        sample = select_local_sample(
            reports,
            settings,
            "integrated_gradients",
        )
        steps = (
            args.ig_steps
            or int(settings["integrated_gradients"]["steps"])
        )
        frame = run_integrated_gradients(
            settings,
            embedder,
            classifier,
            sample,
            results_dir,
            steps,
        )
        summary["integrated_gradients_entries"] = len(frame)
        summary["integrated_gradients_steps"] = steps
        print(
            f"Integrated Gradients: {len(frame)} behavioral entries"
        )

    if "prevalence" in args.stages:
        if index is None:
            index, _ = load_chapter3_artifacts(settings)
        shap_frame = pd.read_csv(results_dir / "shap_entries.csv")
        frame = run_prevalence(
            settings,
            index,
            shap_frame,
            results_dir,
        )
        summary["prevalence_entries"] = len(frame)
        print(f"Prevalence: {len(frame)} entries")

    write_json(results_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
