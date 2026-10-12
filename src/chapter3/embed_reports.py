import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from common import (
    last_token_pool,
    load_report,
    load_settings,
    output_dir,
    read_split_records,
    serialize_report,
    set_seed,
    source_root,
    write_json,
)


def embedding_dir(settings):
    return output_dir(settings) / "embeddings"


def shard_dir(settings):
    return embedding_dir(settings) / "shards"


def prepare_index(settings):
    reports = read_split_records(settings)
    reports = reports.sort_values("sample_id").reset_index(drop=True)
    reports.insert(0, "embedding_row", np.arange(len(reports), dtype=np.int64))

    destination = embedding_dir(settings)
    destination.mkdir(parents=True, exist_ok=True)
    shard_dir(settings).mkdir(parents=True, exist_ok=True)
    reports.to_csv(destination / "index.csv", index=False)
    print(f"Prepared {len(reports):,} reports for embedding")


def completed_rows(settings, shard_index):
    completed = set()
    pattern = f"shard_{shard_index}_chunk_*.npz"
    for path in sorted(shard_dir(settings).glob(pattern)):
        with np.load(path) as chunk:
            completed.update(int(value) for value in chunk["embedding_row"])
    return completed


def create_batches(reports, token_budget, max_batch_size):
    batch = []
    longest_report = 0

    for report in reports.to_dict(orient="records"):
        candidate_longest = max(longest_report, int(report["token_count"]))
        candidate_size = len(batch) + 1
        exceeds_budget = candidate_longest * candidate_size > token_budget
        exceeds_batch_size = candidate_size > max_batch_size

        if batch and (exceeds_budget or exceeds_batch_size):
            yield batch
            batch = []
            longest_report = 0

        batch.append(report)
        longest_report = max(longest_report, int(report["token_count"]))

    if batch:
        yield batch


def embed_texts(model, tokenizer, texts, device):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=False,
        add_special_tokens=True,
        return_tensors="pt",
    )
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

    with torch.inference_mode():
        result = model(**encoded, use_cache=False)
        embeddings = last_token_pool(
            result.last_hidden_state,
            encoded["attention_mask"],
        )
        embeddings = functional.normalize(embeddings, p=2, dim=1)
    return embeddings.float().cpu().numpy()


def embed_batch(model, tokenizer, reports, device, root):
    texts = [
        serialize_report(load_report(root / report["path"]))
        for report in reports
    ]
    try:
        return embed_texts(model, tokenizer, texts, device)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(reports) == 1:
            raise
        middle = len(reports) // 2
        left = embed_batch(model, tokenizer, reports[:middle], device, root)
        right = embed_batch(model, tokenizer, reports[middle:], device, root)
        return np.concatenate([left, right], axis=0)


def load_embedding_model(settings, device):
    model_path = settings["embedding"]["model_snapshot"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        padding_side="left",
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    model.config.use_cache = False
    return model, tokenizer


def embed_shard(settings, shard_index, shard_count, device_name):
    set_seed(settings["seed"])
    index_path = embedding_dir(settings) / "index.csv"
    reports = pd.read_csv(index_path)
    reports = reports.sort_values(
        ["token_count", "embedding_row"]
    ).reset_index(drop=True)
    reports = reports.iloc[shard_index::shard_count].copy()

    already_embedded = completed_rows(settings, shard_index)
    reports = reports[~reports["embedding_row"].isin(already_embedded)]
    if reports.empty:
        print(f"Shard {shard_index}: already complete")
        return

    device = torch.device(device_name)
    model, tokenizer = load_embedding_model(settings, device)
    embedding_settings = settings["embedding"]
    batches = list(
        create_batches(
            reports,
            token_budget=embedding_settings["token_budget"],
            max_batch_size=embedding_settings["max_batch_size"],
        )
    )

    pending_rows = []
    pending_embeddings = []
    existing_chunks = list(
        shard_dir(settings).glob(f"shard_{shard_index}_chunk_*.npz")
    )
    chunk_number = len(existing_chunks)
    root = source_root(settings)

    progress = tqdm(batches, desc=f"Embedding shard {shard_index}")
    for batch in progress:
        embeddings = embed_batch(model, tokenizer, batch, device, root)
        pending_rows.extend(int(report["embedding_row"]) for report in batch)
        pending_embeddings.extend(embeddings)

        progress.set_postfix(
            reports=len(pending_rows),
            max_tokens=max(int(report["token_count"]) for report in batch),
        )
        if len(pending_rows) >= embedding_settings["chunk_size"]:
            save_chunk(
                settings,
                shard_index,
                chunk_number,
                pending_rows,
                pending_embeddings,
            )
            chunk_number += 1
            pending_rows = []
            pending_embeddings = []

    if pending_rows:
        save_chunk(
            settings,
            shard_index,
            chunk_number,
            pending_rows,
            pending_embeddings,
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Shard {shard_index}: complete")


def save_chunk(settings, shard_index, chunk_number, rows, embeddings):
    destination = (
        shard_dir(settings)
        / f"shard_{shard_index}_chunk_{chunk_number:05d}.npz"
    )
    np.savez(
        destination,
        embedding_row=np.asarray(rows, dtype=np.int64),
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )


def merge_shards(settings):
    reports = pd.read_csv(embedding_dir(settings) / "index.csv")
    chunk_paths = sorted(shard_dir(settings).glob("shard_*_chunk_*.npz"))
    if not chunk_paths:
        raise RuntimeError("No embedding chunks were found")

    with np.load(chunk_paths[0]) as first_chunk:
        dimensions = int(first_chunk["embeddings"].shape[1])

    output_path = embedding_dir(settings) / "embeddings.npy"
    matrix = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(reports), dimensions),
    )
    seen = np.zeros(len(reports), dtype=bool)

    for path in tqdm(chunk_paths, desc="Merging embedding shards"):
        with np.load(path) as chunk:
            rows = chunk["embedding_row"].astype(np.int64)
            embeddings = chunk["embeddings"].astype(np.float32)

        if embeddings.shape != (len(rows), dimensions):
            raise RuntimeError(f"Unexpected embedding shape in {path}")
        if seen[rows].any():
            raise RuntimeError(f"Duplicate embedding rows in {path}")

        matrix[rows] = embeddings
        seen[rows] = True

    matrix.flush()
    missing_rows = np.flatnonzero(~seen)
    if len(missing_rows):
        raise RuntimeError(
            f"Missing {len(missing_rows)} embeddings; "
            f"first rows: {missing_rows[:10].tolist()}"
        )

    norms = np.linalg.norm(np.asarray(matrix), axis=1)
    metadata = {
        "reports": len(reports),
        "dimensions": dimensions,
        "chunks": len(chunk_paths),
        "minimum_norm": float(norms.min()),
        "maximum_norm": float(norms.max()),
        "mean_norm": float(norms.mean()),
        "all_finite": bool(np.isfinite(matrix).all()),
    }
    write_json(embedding_dir(settings) / "metadata.json", metadata)
    print(metadata)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Qwen embeddings for the Chapter 3 splits."
    )
    parser.add_argument("--config", default=None)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("prepare")

    shard = commands.add_parser("shard")
    shard.add_argument("--index", type=int, required=True)
    shard.add_argument("--count", type=int, default=2)
    shard.add_argument("--device", required=True)

    commands.add_parser("merge")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = load_settings(args.config)

    if args.command == "prepare":
        prepare_index(settings)
    elif args.command == "shard":
        embed_shard(
            settings,
            shard_index=args.index,
            shard_count=args.count,
            device_name=args.device,
        )
    else:
        merge_shards(settings)


if __name__ == "__main__":
    main()
