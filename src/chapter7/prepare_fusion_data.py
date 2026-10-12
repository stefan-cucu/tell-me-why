
import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
import json5
from datasets import DatasetDict
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from fusion.classifier import load_classifier
from fusion.data import RESPONSE_MARKER, encoder_text_from_text

LABELS = {0: "benign", 1: "malware"}


def get_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def last_token_pool(hidden_states, attention_mask):
    lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
    batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_idx, lengths]


def find_json_span(text):
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, idx + 1
    return None


def remove_classifier_signal(think_body):
    think_body = re.sub(
        r"^Classifier signal:.*(?:\n|$)",
        "",
        think_body,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    think_body = re.sub(
        r"^(?:Final )?Decision:.*(?:\n|$)",
        "",
        think_body,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    think_body = re.sub(
        r"^Treat this classifier decision as authoritative\.\s*(?:\n|$)",
        "",
        think_body,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return think_body.strip()


def classifier_reasoning_prefix(label, confidence):
    return (
        "<think>\n"
        f"Classifier signal: classifier predicts {label} with confidence {confidence:.4f}.\n"
        "Treat this classifier decision as authoritative.\n"
    )


def rewrite_json_object(parsed, label, confidence):
    rewritten = {
        "label": label,
        "confidence": round(float(confidence), 4),
    }
    for key, value in parsed.items():
        if key not in {"label", "confidence"}:
            rewritten[key] = value
    return rewritten


def parse_teacher_response(response):
    think_match = re.search(r"<think>(.*?)</think>", response, flags=re.DOTALL | re.IGNORECASE)
    if not think_match:
        raise ValueError("Teacher response does not contain a complete <think>...</think> block")

    teacher_reasoning = remove_classifier_signal(think_match.group(1))
    if not teacher_reasoning:
        raise ValueError("Teacher reasoning is empty")

    suffix = response[think_match.end():]
    span = find_json_span(suffix)
    if span is None:
        raise ValueError("Teacher response does not contain a JSON object after </think>")

    start, end = span
    payload = suffix[start:end]
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as json_error:
        try:
            parsed = json5.loads(payload)
        except Exception as json5_error:
            raise ValueError(
                f"Teacher response JSON is invalid: {json_error}; JSON5: {json5_error}"
            ) from json5_error
    if not isinstance(parsed, dict):
        raise ValueError("Teacher response JSON is not an object")
    return teacher_reasoning, suffix, start, end, parsed


def rewrite_response(response, label, confidence):
    teacher_reasoning, suffix, start, end, parsed = parse_teacher_response(response)

    assistant_prefix = classifier_reasoning_prefix(label, confidence)
    reasoning_response = (
        assistant_prefix
        + teacher_reasoning
        + f"\nFinal decision: {label} with confidence {confidence:.4f}.\n"
        + "</think>"
    )
    replacement = json.dumps(rewrite_json_object(parsed, label, confidence), ensure_ascii=False)
    rewritten = reasoning_response + suffix[:start] + replacement + suffix[end:]
    return rewritten, assistant_prefix, teacher_reasoning


def rewrite_text(text, label, confidence):
    if RESPONSE_MARKER not in text:
        raise ValueError("Dataset text does not contain response marker")
    prompt, response = text.split(RESPONSE_MARKER, 1)
    rewritten, assistant_prefix, teacher_reasoning = rewrite_response(response, label, confidence)
    return prompt + RESPONSE_MARKER + rewritten, assistant_prefix, teacher_reasoning


def compute_split_decisions(split, tokenizer, embedding_model, classifier, args, device):
    size = len(split)
    labels = [None] * size
    confidences = [None] * size
    prob_benign = [None] * size
    prob_malware = [None] * size
    if "token_count" in split.column_names:
        order = sorted(range(size), key=lambda idx: int(split[idx]["token_count"]))
    else:
        order = list(range(size))

    batches = range(0, size, args.batch_size)
    for start in tqdm(batches, total=(size + args.batch_size - 1) // args.batch_size, desc="Computing classifier decisions"):
        indices = order[start:start + args.batch_size]
        texts = [encoder_text_from_text(split[idx]["text"]) for idx in indices]
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.encoder_max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = embedding_model(**batch, output_hidden_states=False, return_dict=True)
            pooled = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            pooled = F.normalize(pooled.float(), p=2, dim=-1)
            logits = classifier(pooled)["logits"]
            probs = torch.softmax(logits.float(), dim=-1)
            pred = (probs[:, 1] >= args.classifier_threshold).long()
            conf = torch.where(pred == 1, probs[:, 1], probs[:, 0])
        batch_labels = pred.detach().cpu().tolist()
        batch_confidences = conf.detach().cpu().tolist()
        batch_benign = probs[:, 0].detach().cpu().tolist()
        batch_malware = probs[:, 1].detach().cpu().tolist()
        for position, index in enumerate(indices):
            labels[index] = LABELS[int(batch_labels[position])]
            confidences[index] = float(batch_confidences[position])
            prob_benign[index] = float(batch_benign[position])
            prob_malware[index] = float(batch_malware[position])
    return labels, confidences, prob_benign, prob_malware


def process_split(name, split, tokenizer, llm_tokenizer, embedding_model, classifier, args, device):
    original_size = len(split)
    valid_indices = []
    malformed_reasons = Counter()
    for idx, item in enumerate(tqdm(split, desc=f"Preflighting {name}")):
        try:
            if RESPONSE_MARKER not in item["text"]:
                raise ValueError("Dataset text does not contain response marker")
            parse_teacher_response(item["text"].split(RESPONSE_MARKER, 1)[1])
            valid_indices.append(idx)
        except ValueError as exc:
            malformed_reasons[str(exc)] += 1

    split = split.select(valid_indices)
    if not len(split):
        raise ValueError(f"No valid teacher responses remain in {name}")
    if malformed_reasons:
        print(f"Skipping {original_size - len(split)} malformed {name} responses: {dict(malformed_reasons)}")

    labels, confidences, prob_benign, prob_malware = compute_split_decisions(
        split, tokenizer, embedding_model, classifier, args, device
    )
    source_label_column = "teacher_label" if "teacher_label" in split.column_names else "target_label"
    teacher_labels = [str(value).lower() for value in split[source_label_column]]
    rows = []
    disagreements = 0

    for idx, item in enumerate(tqdm(split, desc=f"Rewriting {name}")):
        label = labels[idx]
        confidence = confidences[idx]
        teacher_label = teacher_labels[idx]
        disagrees = teacher_label in {"benign", "malware"} and teacher_label != label
        disagreements += int(disagrees)
        if args.drop_disagreements and disagrees:
            continue

        rewritten_text, assistant_prefix, teacher_reasoning = rewrite_text(
            item["text"], label, confidence
        )
        row = dict(item)
        row["text"] = rewritten_text
        row["assistant_prefix"] = assistant_prefix
        row["assistant_prefix_token_count"] = len(
            llm_tokenizer(assistant_prefix, add_special_tokens=False)["input_ids"]
        )
        row["teacher_reasoning_token_count"] = len(
            llm_tokenizer(teacher_reasoning, add_special_tokens=False)["input_ids"]
        )
        row["token_count"] = len(
            llm_tokenizer(rewritten_text, add_special_tokens=False)["input_ids"]
        )
        row["classifier_label"] = label
        row["classifier_confidence"] = confidence
        row["classifier_prob_benign"] = prob_benign[idx]
        row["classifier_prob_malware"] = prob_malware[idx]
        row["original_target_label"] = item.get("target_label")
        row["original_teacher_label"] = item.get("teacher_label")
        row["classifier_disagrees_with_teacher"] = bool(disagrees)
        rows.append(row)

    stats = {
        "original_examples": original_size,
        "retained_examples": len(rows),
        "dropped_examples": original_size - len(rows),
        "drop_disagreements": args.drop_disagreements,
        "classifier_labels_before_filter": dict(Counter(labels)),
        "classifier_labels_after_filter": dict(Counter(row["classifier_label"] for row in rows)),
        "teacher_labels_before_filter": dict(Counter(teacher_labels)),
        "teacher_labels_after_filter": dict(Counter(row["teacher_label"] for row in rows)),
        "classifier_teacher_disagreements": disagreements,
        "classifier_teacher_disagreement_rate": disagreements / len(split) if len(split) else 0,
        "mean_classifier_confidence": sum(confidences) / len(confidences) if confidences else 0,
        "malformed_teacher_responses": original_size - len(split),
        "malformed_teacher_response_reasons": dict(malformed_reasons),
    }
    return rows, stats


def main(args):
    output = Path(args.output_path)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        shutil.rmtree(output)

    dtype = get_dtype()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using dtype={dtype}, device={device}")
    dataset = DatasetDict.load_from_disk(args.dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_tokenizer, trust_remote_code=True)

    embedding_model = AutoModel.from_pretrained(
        args.embedding_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    embedding_model.eval()
    classifier = load_classifier(args.classifier_path, device=device)

    output_dataset = DatasetDict()
    metadata = {
        "source_dataset": args.dataset_path,
        "embedding_model": args.embedding_model,
        "llm_tokenizer": args.llm_tokenizer,
        "classifier_path": args.classifier_path,
        "classifier_threshold": args.classifier_threshold,
        "encoder_max_length": args.encoder_max_length,
        "drop_disagreements": args.drop_disagreements,
        "reasoning_format": "classifier_signal_then_teacher_reasoning",
        "splits": {},
    }
    for split_name, split in dataset.items():
        if args.splits and split_name not in args.splits:
            output_dataset[split_name] = split
            continue
        if args.max_samples_per_split is not None:
            split = split.select(range(min(args.max_samples_per_split, len(split))))
        rows, stats = process_split(
            split_name,
            split,
            tokenizer,
            llm_tokenizer,
            embedding_model,
            classifier,
            args,
            device,
        )
        output_dataset[split_name] = split.from_list(rows)
        metadata["splits"][split_name] = stats
        print(f"{split_name}: {json.dumps(stats, indent=2)}")

    output_dataset.save_to_disk(str(output))
    with open(output / "classifier_consistent_metadata.json", "w") as output_file:
        json.dump(metadata, output_file, indent=2)
    print(f"Saved classifier-consistent dataset to {output}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset_path", required=True)
    parser.add_argument("--output", dest="output_path", required=True)
    parser.add_argument("--classifier", dest="classifier_path", required=True)
    parser.add_argument("--llm-tokenizer", dest="llm_tokenizer", required=True)
    parser.add_argument("--embedding-model", dest="embedding_model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--classifier-threshold", dest="classifier_threshold", type=float, default=0.5)
    parser.add_argument("--encoder-max-length", dest="encoder_max_length", type=int, default=16384)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=4)
    parser.add_argument("--splits", nargs="*")
    parser.add_argument("--max-samples-per-split", dest="max_samples_per_split", type=int)
    parser.add_argument(
        "--drop-disagreements",
        dest="drop_disagreements",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
