import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer

from common import (
    length_bucket,
    load_report,
    load_settings,
    output_dir,
    report_family,
    report_hash,
    report_id,
    report_label,
    serialize_report,
    set_seed,
    source_root,
    write_json,
)


def discover_reports(data_dir):
    data_dir = Path(data_dir)
    goodware_dir = data_dir / "goodware_dataset" / "reports_summary"
    malware_dir = data_dir / "malware_dataset"

    report_paths = sorted(goodware_dir.glob("*.json"))
    for family_dir in sorted(malware_dir.iterdir()):
        reports_dir = family_dir / "reports_summary"
        if reports_dir.is_dir():
            report_paths.extend(sorted(reports_dir.glob("*.json")))
    return report_paths


def count_tokens(tokenizer, text, max_tokens):
    if len(text) <= 150_000:
        return len(
            tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]
        )

    approximate_count = 1
    chunk_size = 50_000
    for start in range(0, len(text), chunk_size):
        approximate_count += len(
            tokenizer(
                text[start : start + chunk_size],
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]
        )
        if approximate_count > max_tokens + 1024:
            return max_tokens + 1

    return len(
        tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
    )


def load_candidates(settings):
    embedding_settings = settings["embedding"]
    tokenizer = AutoTokenizer.from_pretrained(
        embedding_settings["model_snapshot"],
        local_files_only=True,
    )
    tokenizer.model_max_length = 10**9

    root = source_root(settings)
    report_paths = discover_reports(settings["data_dir"])
    duplicate_groups = defaultdict(list)
    skipped = {
        "invalid": 0,
        "too_long": 0,
        "duplicate": 0,
        "conflict": 0,
    }

    for path in tqdm(report_paths, desc="Reading reports"):
        try:
            relative_path = path.relative_to(root)
            family = report_family(relative_path)
            serialized = serialize_report(load_report(path))
            token_count = count_tokens(
                tokenizer,
                serialized,
                embedding_settings["max_tokens"],
            )
        except Exception:
            skipped["invalid"] += 1
            continue

        if token_count > embedding_settings["max_tokens"]:
            skipped["too_long"] += 1
            continue

        duplicate_groups[report_hash(serialized)].append(
            {
                "sample_id": report_id(relative_path),
                "path": str(relative_path),
                "label": report_label(family),
                "family": family,
                "token_count": token_count,
                "length_bucket": length_bucket(token_count),
                "_content_hash": report_hash(serialized),
            }
        )

    reports = []
    for copies in duplicate_groups.values():
        labels = {copy["label"] for copy in copies}
        malware_families = {
            copy["family"] for copy in copies if copy["family"] != "goodware"
        }
        if len(labels) > 1 or len(malware_families) > 1:
            skipped["conflict"] += len(copies)
            continue

        copies.sort(key=lambda copy: copy["path"])
        reports.append(copies[0])
        skipped["duplicate"] += len(copies) - 1

    reports.sort(key=lambda report: report["path"])
    print(f"Discovered reports: {len(report_paths):,}")
    print(f"Usable unique reports: {len(reports):,}")
    for reason, count in skipped.items():
        print(f"Skipped {reason}: {count:,}")
    return pd.DataFrame(reports)


def stratification_labels(reports):
    family_and_length = reports["family"] + ":" + reports["length_bucket"]
    group_sizes = family_and_length.value_counts()
    if len(group_sizes) and group_sizes.min() >= 4:
        return family_and_length
    return reports["family"]


def assign_splits(reports, settings):
    split_settings = settings["splits"]
    total_fraction = (
        split_settings["train"]
        + split_settings["validation"]
        + split_settings["test"]
    )
    if abs(total_fraction - 1.0) > 1e-9:
        raise ValueError("Train, validation, and test fractions must sum to 1")

    malware = reports[reports["label"] == "malware"]
    goodware = reports[reports["label"] == "goodware"]

    malware_for_classification, adversarial = train_test_split(
        malware,
        test_size=split_settings["adversarial_malware"],
        random_state=settings["seed"],
        stratify=stratification_labels(malware),
    )
    classification_reports = pd.concat(
        [goodware, malware_for_classification],
        ignore_index=False,
    )

    holdout_fraction = split_settings["validation"] + split_settings["test"]
    training, holdout = train_test_split(
        classification_reports,
        test_size=holdout_fraction,
        random_state=settings["seed"],
        stratify=stratification_labels(classification_reports),
    )
    validation_fraction = split_settings["validation"] / holdout_fraction
    test, validation = train_test_split(
        holdout,
        test_size=validation_fraction,
        random_state=settings["seed"],
        stratify=stratification_labels(holdout),
    )

    assigned = reports.copy()
    assigned["split"] = ""
    assigned.loc[training.index, "split"] = "train"
    assigned.loc[validation.index, "split"] = "validation"
    assigned.loc[test.index, "split"] = "test"
    assigned.loc[adversarial.index, "split"] = "adversarial"
    return assigned


def validate_splits(reports, max_tokens):
    if not reports["split"].isin(
        ["train", "validation", "test", "adversarial"]
    ).all():
        raise RuntimeError("At least one report was not assigned to a split")
    if reports["sample_id"].duplicated().any():
        raise RuntimeError("A report appears in more than one split")
    if reports["_content_hash"].duplicated().any():
        raise RuntimeError("Duplicate report content remains in the splits")
    if (reports["token_count"] > max_tokens).any():
        raise RuntimeError("An over-context report entered the splits")


def save_splits(reports, settings):
    visible_columns = [
        "sample_id",
        "path",
        "label",
        "family",
        "token_count",
        "length_bucket",
    ]
    split_records = {}
    for split_name in ["train", "validation", "test", "adversarial"]:
        selected = reports[reports["split"] == split_name]
        split_records[split_name] = selected[visible_columns].to_dict(
            orient="records"
        )
        print(f"{split_name}: {len(selected):,}")

    destination = output_dir(settings) / "splits.json"
    write_json(
        destination,
        {
            "seed": settings["seed"],
            "max_tokens": settings["embedding"]["max_tokens"],
            "splits": split_records,
        },
    )
    print(f"Saved {destination}")


def main():
    parser = argparse.ArgumentParser(
        description="Create the Chapter 3 train, validation, test, and adversarial splits."
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    settings = load_settings(args.config)
    set_seed(settings["seed"])
    reports = load_candidates(settings)
    reports = assign_splits(reports, settings)
    validate_splits(reports, settings["embedding"]["max_tokens"])
    save_splits(reports, settings)


if __name__ == "__main__":
    main()
