import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer


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


def serialize_report(report):
    return json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_json_report(path):
    try:
        with Path(path).open(encoding="utf-8") as handle:
            data = json.load(handle)
        return serialize_report(data.get("summary", data))
    except Exception as error:
        print(f"Error loading {path}: {error}")
        return ""


class ReportEmbedder:
    def __init__(self, model_path, device):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            padding_side="left",
        )
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()
        self.model.config.use_cache = False

    def embed(self, report):
        if isinstance(report, str):
            try:
                report = json.loads(report)
            except json.JSONDecodeError:
                text = report
            else:
                text = serialize_report(report)
        else:
            text = serialize_report(report)
        encoded = self.tokenizer(
            [text],
            padding=True,
            truncation=False,
            add_special_tokens=True,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        with torch.inference_mode():
            output = self.model(**encoded, use_cache=False)
            embedding = last_token_pool(output.last_hidden_state, encoded["attention_mask"])
            embedding = functional.normalize(embedding, p=2, dim=1)
        return embedding[0].float().cpu()
