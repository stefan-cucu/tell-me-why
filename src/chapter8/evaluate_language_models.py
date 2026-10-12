import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import DatasetDict
from tqdm import tqdm

from common import (
    append_jsonl,
    classification_metrics,
    extract_reasoning,
    load_jsonl,
    normalize_label,
    parse_completion,
    resolved,
    response_prompt,
)


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["malware", "benign", "uncertain"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "suspicious_behaviors": {"type": "array", "items": {"type": "string"}},
        "benign_behaviors": {"type": "array", "items": {"type": "string"}},
        "missing_or_weak_evidence": {"type": "array", "items": {"type": "string"}},
        "explanation": {"type": "string"},
    },
    "required": [
        "label",
        "confidence",
        "suspicious_behaviors",
        "benign_behaviors",
        "missing_or_weak_evidence",
        "explanation",
    ],
}


def arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate the base language model or explanation SFT on a shared cohort."
    )
    parser.add_argument("--backend", choices=("base", "sft"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--host")
    parser.add_argument("--ollama-model", default="qwen3:8b")
    parser.add_argument("--base-model")
    parser.add_argument("--adapter")
    parser.add_argument("--max-sequence-length", type=int, default=28672)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    if args.backend == "base" and not args.host:
        parser.error("--host is required for the base Ollama backend")
    if args.backend == "sft" and (not args.base_model or not args.adapter):
        parser.error("--base-model and --adapter are required for the SFT backend")
    return args


def reference_rows(args):
    reference = load_jsonl(args.reference)
    paths = list(reference)
    dataset = DatasetDict.load_from_disk(args.dataset)[args.split]
    rows = {resolved(row["path"]): dict(row) for row in dataset}
    missing = [path for path in paths if path not in rows]
    if missing:
        raise ValueError(f"{len(missing)} reference paths are absent from the dataset")
    selected = [rows[path] for path in paths]
    return selected[: args.limit] if args.limit is not None else selected


class OllamaGenerator:
    def __init__(self, args):
        import ollama

        self.args = args
        self.client = ollama.Client(host=args.host, timeout=args.timeout)

    def generate(self, prompt):
        response = self.client.generate(
            model=self.args.ollama_model,
            prompt=prompt,
            think=True,
            format=OUTPUT_SCHEMA,
            options={
                "temperature": 0,
                "num_ctx": self.args.context_length,
                "num_predict": self.args.max_new_tokens,
                "repeat_penalty": self.args.repetition_penalty,
                "seed": 42,
            },
            keep_alive=self.args.keep_alive,
        )
        reasoning = str(getattr(response, "thinking", "") or "").strip()
        answer = str(getattr(response, "response", "") or "").strip()
        return f"<think>\n{reasoning}\n</think>\n{answer}", reasoning


class SFTGenerator:
    def __init__(self, args):
        import torch
        from peft import PeftModel
        from unsloth import FastLanguageModel

        self.args = args
        self.torch = torch
        model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=args.max_sequence_length,
            dtype=None,
            load_in_4bit=True,
        )
        self.model = PeftModel.from_pretrained(model, args.adapter)
        FastLanguageModel.for_inference(self.model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        length = inputs["input_ids"].shape[1]
        if length > self.args.max_sequence_length:
            raise ValueError(
                f"Prompt has {length} tokens, exceeding {self.args.max_sequence_length}"
            )
        inputs = inputs.to(self.model.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=False,
                repetition_penalty=self.args.repetition_penalty,
                no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        completion = self.tokenizer.decode(
            output[0][length:], skip_special_tokens=True
        ).strip()
        return completion, extract_reasoning(completion)


def summarize(records, args):
    metrics = classification_metrics(records)
    parse_counts = Counter(record.get("parse_status", "error") for record in records)
    total = metrics["total"]
    metrics.update(
        {
            "backend": args.backend,
            "model": args.ollama_model if args.backend == "base" else args.adapter,
            "dataset": args.dataset,
            "reference": args.reference,
            "json_parse_rate": (
                sum(parse_counts[key] for key in ("json", "json5")) / total
                if total
                else 0.0
            ),
            "label_parse_rate": 1 - parse_counts["error"] / total if total else 0.0,
            "parse_counts": dict(parse_counts),
        }
    )
    return metrics


def main():
    args = arguments()
    output = Path(args.output)
    summary = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        output.unlink(missing_ok=True)
        summary.unlink(missing_ok=True)
    rows = reference_rows(args)
    latest = load_jsonl(output) if output.exists() else {}
    completed = {
        path: record
        for path, record in latest.items()
        if record.get("status", "completed") == "completed"
    }
    generator = OllamaGenerator(args) if args.backend == "base" else SFTGenerator(args)

    for index, item in enumerate(tqdm(rows, desc=f"Evaluating {args.backend}"), 1):
        path = resolved(item["path"])
        if path in completed:
            continue
        target_label = normalize_label(item.get("target_label") or item.get("teacher_label"))
        try:
            completion, reasoning = generator.generate(response_prompt(item["text"]))
            label, payload, parse_status, parse_error = parse_completion(completion)
            record = {
                "status": "completed",
                "idx": index - 1,
                "path": path,
                "target_label": target_label,
                "predicted_label": label,
                "is_correct": label == target_label,
                "parse_status": parse_status,
                "parse_error": parse_error,
                "json_output": payload,
                "generated_reasoning": reasoning,
                "raw_completion": completion,
            }
        except Exception as error:
            record = {
                "status": "error",
                "idx": index - 1,
                "path": path,
                "target_label": target_label,
                "error": str(error),
            }
        append_jsonl(output, record)
        latest[path] = record
        if record["status"] == "completed":
            completed[path] = record
        if args.log_every and index % args.log_every == 0:
            current = [latest[resolved(row["path"])] for row in rows if resolved(row["path"]) in latest]
            current_summary = summarize(current, args)
            summary.write_text(json.dumps(current_summary, indent=2) + "\n")

    records = [latest[resolved(row["path"])] for row in rows if resolved(row["path"]) in latest]
    metrics = summarize(records, args)
    summary.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
