from collections import Counter

import torch
from datasets import DatasetDict


RESPONSE_MARKER = "### Response:\n"
INPUT_MARKER = "### Input:\n"


def response_prompt_from_text(text):
    if RESPONSE_MARKER not in text:
        raise ValueError("Dataset text does not contain response marker")
    return text.split(RESPONSE_MARKER, 1)[0] + RESPONSE_MARKER


def encoder_text_from_text(text):
    if INPUT_MARKER not in text or RESPONSE_MARKER not in text:
        return text
    after_input = text.split(INPUT_MARKER, 1)[1]
    return after_input.split("\n\n" + RESPONSE_MARKER.rstrip("\n"), 1)[0].strip()


def label_to_binary(label):
    value = str(label or "").lower().strip()
    if "malware" in value:
        return 1
    if "benign" in value or "goodware" in value:
        return 0
    return -100


class FusionSFTCollator:
    def __init__(
        self,
        llm_tokenizer,
        encoder_tokenizer,
        max_seq_length=32000,
        encoder_max_length=4096,
        truncate_strategy="preserve_response",
        decision_source="target",
        classifier_label_source="target",
    ):
        self.llm_tokenizer = llm_tokenizer
        self.encoder_tokenizer = encoder_tokenizer
        self.max_seq_length = max_seq_length
        self.encoder_max_length = encoder_max_length
        self.truncate_strategy = truncate_strategy
        self.decision_source = decision_source
        self.classifier_label_source = classifier_label_source
        if self.decision_source not in {"target", "classifier_column", "runtime_classifier"}:
            raise ValueError(f"Unknown decision_source: {self.decision_source}")
        if self.classifier_label_source not in {"target", "classifier_column"}:
            raise ValueError(f"Unknown classifier_label_source: {self.classifier_label_source}")
        self.pad_token_id = llm_tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = llm_tokenizer.eos_token_id

    def _encode_supervised_text(self, feature):
        text = feature["text"]
        prompt = response_prompt_from_text(text)
        response = text.split(RESPONSE_MARKER, 1)[1]
        assistant_prefix = str(feature.get("assistant_prefix") or "")
        if assistant_prefix:
            if not response.startswith(assistant_prefix):
                raise ValueError("assistant_prefix is not the start of the response")
            prompt += assistant_prefix
            response = response[len(assistant_prefix):]
        prompt_ids = self.llm_tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = self.llm_tokenizer(response, add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + response_ids

        if self.truncate_strategy == "none":
            if len(full_ids) > self.max_seq_length:
                raise ValueError(
                    f"No-truncation mode received an over-limit example: "
                    f"{len(full_ids)} tokens > max_seq_length={self.max_seq_length}. "
                    "Filter the dataset before collation."
                )
            labels = [-100] * len(prompt_ids) + response_ids
            return full_ids, labels

        if self.truncate_strategy != "preserve_response":
            raise ValueError(f"Unknown truncate_strategy: {self.truncate_strategy}")

        if len(response_ids) >= self.max_seq_length:
            input_ids = response_ids[:self.max_seq_length]
            labels = input_ids.copy()
            return input_ids, labels

        prompt_budget = self.max_seq_length - len(response_ids)
        prompt_ids = prompt_ids[:prompt_budget]
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        return input_ids, labels

    def __call__(self, features):
        encoded = [self._encode_supervised_text(feature) for feature in features]
        max_len = max(len(input_ids) for input_ids, _ in encoded)
        input_batch = []
        label_batch = []
        attention_batch = []
        for input_ids, labels in encoded:
            pad_len = max_len - len(input_ids)
            input_batch.append(input_ids + [self.pad_token_id] * pad_len)
            label_batch.append(labels + [-100] * pad_len)
            attention_batch.append([1] * len(input_ids) + [0] * pad_len)

        encoder_texts = [encoder_text_from_text(f["text"]) for f in features]
        encoder = self.encoder_tokenizer(
            encoder_texts,
            padding=True,
            truncation=True,
            max_length=self.encoder_max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

        target_labels = [label_to_binary(f.get("target_label") or f.get("teacher_label")) for f in features]
        column_labels = [label_to_binary(f.get("classifier_label")) for f in features]
        column_confidences = [float(f.get("classifier_confidence") or 0.0) for f in features]

        if self.classifier_label_source == "classifier_column":
            cls_labels = [col if col != -100 else target for col, target in zip(column_labels, target_labels)]
        else:
            cls_labels = target_labels

        batch = {
            "input_ids": torch.tensor(input_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_batch, dtype=torch.long),
            "labels": torch.tensor(label_batch, dtype=torch.long),
            "encoder_input_ids": encoder["input_ids"],
            "encoder_attention_mask": encoder["attention_mask"],
            "classifier_labels": torch.tensor(cls_labels, dtype=torch.long),
        }
        if self.decision_source == "target":
            decision_labels = target_labels
            decision_confidences = [1.0 if label != -100 else 0.0 for label in decision_labels]
            batch["decision_labels"] = torch.tensor(decision_labels, dtype=torch.long)
            batch["decision_confidences"] = torch.tensor(decision_confidences, dtype=torch.float32)
        elif self.decision_source == "classifier_column":
            decision_labels = [col if col != -100 else target for col, target in zip(column_labels, target_labels)]
            decision_confidences = [conf if col != -100 else (1.0 if target != -100 else 0.0) for col, target, conf in zip(column_labels, target_labels, column_confidences)]
            batch["decision_labels"] = torch.tensor(decision_labels, dtype=torch.long)
            batch["decision_confidences"] = torch.tensor(decision_confidences, dtype=torch.float32)
        return batch


def token_bucket(token_count):
    if token_count <= 8192:
        return "00000_08192"
    if token_count <= 12288:
        return "08193_12288"
    if token_count <= 16384:
        return "12289_16384"
    if token_count <= 20480:
        return "16385_20480"
    if token_count <= 24576:
        return "20481_24576"
    if token_count <= 28672:
        return "24577_28672"
    if token_count <= 32000:
        return "28673_32000"
    return "32001_plus"


def get_token_counts(dataset):
    if "token_count" in dataset.column_names:
        return [int(x) for x in dataset["token_count"]]
    return [None] * len(dataset)


def split_fit_stats(dataset, max_seq_length):
    counts = get_token_counts(dataset)
    known_counts = [x for x in counts if x is not None]
    kept = sum(1 for x in counts if x is not None and x <= max_seq_length)
    skipped = sum(1 for x in counts if x is not None and x > max_seq_length)
    buckets = Counter(token_bucket(x) for x in known_counts)
    return {
        "original_examples": len(dataset),
        "kept_examples": kept,
        "skipped_examples": skipped,
        "max_seq_length": max_seq_length,
        "keep_ratio": kept / len(dataset) if len(dataset) else 0,
        "max_token_count": max(known_counts) if known_counts else 0,
        "bucket_counts": dict(buckets),
    }


def filter_split_to_context(dataset, max_seq_length):
    if "token_count" not in dataset.column_names:
        raise ValueError("No-truncation filtering requires a token_count column in the dataset.")
    stats = split_fit_stats(dataset, max_seq_length)
    filtered = dataset.filter(lambda item: int(item["token_count"]) <= max_seq_length, desc=f"Filtering <= {max_seq_length} tokens")
    return filtered, stats


def load_fusion_dataset(
    dataset_path,
    train_split="train",
    eval_split="eval",
    max_train_samples=None,
    max_eval_samples=None,
    max_seq_length=None,
    truncate_strategy="preserve_response",
    no_eval=False,
    length_order="none",
):
    dataset_dict = DatasetDict.load_from_disk(dataset_path)
    train_dataset = dataset_dict[train_split]
    eval_dataset = None if no_eval else (dataset_dict[eval_split] if eval_split and eval_split in dataset_dict else None)
    filter_stats = {}

    if truncate_strategy == "none":
        if max_seq_length is None:
            raise ValueError("max_seq_length is required for no-truncation filtering")
        train_dataset, filter_stats["train"] = filter_split_to_context(train_dataset, max_seq_length)
        if eval_dataset is not None:
            eval_dataset, filter_stats["eval"] = filter_split_to_context(eval_dataset, max_seq_length)
    elif truncate_strategy == "preserve_response":
        filter_stats["train"] = split_fit_stats(train_dataset, max_seq_length) if max_seq_length else {"original_examples": len(train_dataset)}
        if eval_dataset is not None:
            filter_stats["eval"] = split_fit_stats(eval_dataset, max_seq_length) if max_seq_length else {"original_examples": len(eval_dataset)}
    else:
        raise ValueError(f"Unknown truncate_strategy: {truncate_strategy}")

    if length_order not in {"none", "ascending", "descending"}:
        raise ValueError(f"Unknown length_order: {length_order}")
    if length_order != "none":
        if "token_count" not in train_dataset.column_names:
            raise ValueError("length_order requires token_count column")
        reverse = length_order == "descending"
        train_dataset = train_dataset.sort("token_count", reverse=reverse)
        if eval_dataset is not None and "token_count" in eval_dataset.column_names:
            eval_dataset = eval_dataset.sort("token_count", reverse=reverse)

    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
    if eval_dataset is not None and max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))
    return train_dataset, eval_dataset, filter_stats
