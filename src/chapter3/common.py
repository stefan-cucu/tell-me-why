import hashlib
import json
import os
import random
from pathlib import Path

CHAPTER_DIR = Path(__file__).resolve().parent
CLEAN_ROOT = CHAPTER_DIR.parents[1]
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "chapter3.json"


def configure_runtime():
    cache_dir = CLEAN_ROOT / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("TORCH_HOME", str(cache_dir / "torch"))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


configure_runtime()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def load_settings(config_path=None):
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.is_absolute():
        path = CLEAN_ROOT / path
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def output_dir(settings):
    return Path(settings["output_dir"])


def source_root(settings):
    return Path(settings["source_root"])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_report(path):
    with Path(path).open(encoding="utf-8") as handle:
        raw_report = json.load(handle)
    report = raw_report.get("summary", raw_report)
    if not isinstance(report, dict):
        raise ValueError("The report summary must be a JSON object")
    return report


def serialize_report(report):
    return json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def report_family(path):
    parts = Path(path).parts
    if "goodware_dataset" in parts:
        return "goodware"
    malware_index = parts.index("malware_dataset")
    return parts[malware_index + 1].lower()


def report_label(family):
    return "goodware" if family == "goodware" else "malware"


def report_id(relative_path):
    value = str(relative_path).encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:20]


def report_hash(serialized_report):
    return hashlib.sha256(serialized_report.encode("utf-8")).hexdigest()


def length_bucket(token_count):
    if token_count <= 2048:
        return "short"
    if token_count <= 8192:
        return "medium"
    if token_count <= 16384:
        return "long"
    return "very_long"


def json_value(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path, value):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=json_value)


def read_split_records(settings):
    split_path = output_dir(settings) / "splits.json"
    with split_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    records = []
    for split_name, split_records in payload["splits"].items():
        for record in split_records:
            records.append({**record, "split": split_name})
    return pd.DataFrame(records)


def load_embeddings(settings):
    embedding_dir = output_dir(settings) / "embeddings"
    index = pd.read_csv(embedding_dir / "index.csv")
    matrix = np.load(embedding_dir / "embeddings.npy", mmap_mode="r")
    if len(index) != len(matrix):
        raise RuntimeError("Embedding index and matrix have different lengths")
    return index, matrix


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_layers=(512, 256), dropout=0.3):
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


def last_token_pool(hidden_states, attention_mask):
    uses_left_padding = bool(
        (attention_mask[:, -1].sum() == attention_mask.shape[0]).item()
    )
    if uses_left_padding:
        return hidden_states[:, -1]

    last_token = attention_mask.sum(dim=1) - 1
    batch = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch, last_token]
