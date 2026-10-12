import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-names", nargs="+", default=["train", "test"])
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--tokenizer", default="unsloth/Qwen3-8B-Base")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def report_label(path):
    return "goodware" if "goodware_dataset" in str(path).lower() else "malware"


def report_text(path):
    with Path(path).open(encoding="utf-8") as handle:
        report = json.load(handle)
    if isinstance(report, dict) and isinstance(report.get("summary"), dict):
        report = report["summary"]
    if not isinstance(report, dict):
        raise ValueError(f"Report is not a JSON object: {path}")
    return json.dumps(report, ensure_ascii=False, sort_keys=True)


def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a quantile from an empty list")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def add_length_buckets(records):
    by_split = defaultdict(list)
    for record in records:
        by_split[record["split"]].append(record)
    boundaries = {}
    for split, rows in by_split.items():
        lengths = [row["tokens"] for row in rows]
        short_limit = quantile(lengths, 1 / 3)
        medium_limit = quantile(lengths, 2 / 3)
        boundaries[split] = {
            "short_max": short_limit,
            "medium_max": medium_limit,
        }
        for row in rows:
            if row["tokens"] <= short_limit:
                row["length_bucket"] = "short"
            elif row["tokens"] <= medium_limit:
                row["length_bucket"] = "medium"
            else:
                row["length_bucket"] = "long"
    return boundaries


def allocation(sizes, fraction, target):
    counts = {key: math.floor(size * fraction) for key, size in sizes.items()}
    remaining = target - sum(counts.values())
    order = sorted(
        sizes,
        key=lambda key: (-(sizes[key] * fraction - counts[key]), key),
    )
    for key in order:
        if remaining == 0:
            break
        if counts[key] < sizes[key]:
            counts[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("Could not allocate the requested sample size")
    return counts


def sample_split(rows, fraction, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["label"], row["length_bucket"])].append(row)
    target = round(len(rows) * fraction)
    counts = allocation({key: len(value) for key, value in groups.items()}, fraction, target)
    selected = []
    for number, key in enumerate(sorted(groups)):
        candidates = sorted(groups[key], key=lambda item: item["path"])
        rng = random.Random(seed + number)
        selected.extend(rng.sample(candidates, counts[key]))
    random.Random(seed).shuffle(selected)
    if len(selected) != target:
        raise RuntimeError(f"Expected {target} selected reports, got {len(selected)}")
    return selected, counts


def build_manifest(split_path, split_names, fraction, tokenizer, seed):
    if not 0 < fraction <= 1:
        raise ValueError("--fraction must be in the interval (0, 1]")
    with Path(split_path).open(encoding="utf-8") as handle:
        split_data = json.load(handle)
    records = []
    for split in split_names:
        if split not in split_data:
            raise KeyError(f"Split not found: {split}")
        paths = split_data[split]
        if len(paths) != len(set(paths)):
            raise ValueError(f"Split {split} contains duplicate paths")
        for path in paths:
            text = report_text(path)
            records.append({
                "path": path,
                "split": split,
                "label": report_label(path),
                "tokens": len(tokenizer(text, add_special_tokens=False)["input_ids"]),
            })
    boundaries = add_length_buckets(records)
    selected = []
    split_stats = {}
    for split_number, split in enumerate(split_names):
        rows = [record for record in records if record["split"] == split]
        chosen, counts = sample_split(rows, fraction, seed + split_number * 1000)
        selected.extend(chosen)
        split_stats[split] = {
            "available": len(rows),
            "selected": len(chosen),
            "fraction": len(chosen) / len(rows) if rows else 0,
            "available_by_stratum": {
                f"{label}:{bucket}": count
                for (label, bucket), count in sorted(Counter((row["label"], row["length_bucket"]) for row in rows).items())
            },
            "selected_by_stratum": {
                f"{label}:{bucket}": count
                for (label, bucket), count in sorted(counts.items())
            },
        }
    return {
        "metadata": {
            "source_splits": str(split_path),
            "split_names": split_names,
            "fraction": fraction,
            "seed": seed,
            "tokenizer": tokenizer.name_or_path,
            "length_boundaries": boundaries,
            "splits": split_stats,
        },
        "samples": selected,
    }


def main():
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    manifest = build_manifest(
        args.splits,
        args.split_names,
        args.fraction,
        tokenizer,
        args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    for split, stats in manifest["metadata"]["splits"].items():
        print(f"{split}: selected {stats['selected']}/{stats['available']} ({stats['fraction']:.2%})")
        print("  selected strata:", stats["selected_by_stratum"])
    print(f"Saved {len(manifest['samples'])} reports to {output}")


if __name__ == "__main__":
    main()
