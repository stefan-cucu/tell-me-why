
import argparse
import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["UNSLOTH_NUM_WORKERS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

from datasets import Dataset, DatasetDict
from tqdm import tqdm

MAX_SEQ_LENGTH = 32000
DTYPE = None
LOAD_IN_4BIT = True
MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
SYSTEM_PROMPT = """You are an expert malware analyst labeling dynamic sandbox behavioral-summary logs.

Classify the sample as malware, benign, or uncertain based only on behaviors explicitly present in the provided log.

Respond in exactly this format:
<think>
Briefly reason about the strongest suspicious evidence, strongest benign evidence, weak or missing evidence, and the final decision.
</think>
{
  "label": "malware | benign | uncertain",
  "confidence": 0.0,
  "suspicious_behaviors": [],
  "benign_behaviors": [],
  "missing_or_weak_evidence": [],
  "explanation": ""
}

Do not output markdown, comments, or any extra text outside the <think> block and JSON object."""

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def load_jsonl(path):
    records = []
    bad_lines = 0
    if not os.path.exists(path):
        print(f"Warning: {path} not found, skipping.")
        return records, bad_lines

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
    return records, bad_lines


def normalize_label(label):
    label = str(label or "").lower().strip()
    if "malware" in label:
        return "malware"
    if "benign" in label:
        return "benign"
    if "uncertain" in label:
        return "uncertain"
    return "error"


def normalize_teacher_output(record):
    output = record.get("json_output") or {}
    if not isinstance(output, dict):
        output = {}

    normalized = {
        "label": record.get("predicted_label") or output.get("label") or "error",
        "confidence": output.get("confidence", 0.0),
        "suspicious_behaviors": output.get("suspicious_behaviors") or [],
        "benign_behaviors": output.get("benign_behaviors") or [],
        "missing_or_weak_evidence": output.get("missing_or_weak_evidence") or [],
        "explanation": output.get("explanation") or "",
    }
    normalized["label"] = normalize_label(normalized["label"])

    try:
        normalized["confidence"] = float(normalized["confidence"])
    except (TypeError, ValueError):
        normalized["confidence"] = 0.0
    normalized["confidence"] = max(0.0, min(1.0, normalized["confidence"]))
    return normalized


def load_log_summary(path):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("summary"), dict):
        return data["summary"]
    return data


def clean_reasoning_text(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_first_json_start(text):
    for idx, ch in enumerate(text):
        if ch == "{":
            return idx
    return -1


def build_evidence_reasoning(output):
    lines = []
    suspicious = output.get("suspicious_behaviors") or []
    benign = output.get("benign_behaviors") or []
    missing = output.get("missing_or_weak_evidence") or []

    if suspicious:
        lines.append("Suspicious evidence: " + "; ".join(str(x) for x in suspicious[:4]))
    else:
        lines.append("Suspicious evidence: no explicit suspicious behavior was identified by the teacher.")

    if benign:
        lines.append("Benign evidence: " + "; ".join(str(x) for x in benign[:3]))
    else:
        lines.append("Benign evidence: no meaningful benign behavior outweighed the suspicious evidence.")

    if missing:
        lines.append("Missing or weak evidence: " + "; ".join(str(x) for x in missing[:3]))

    lines.append(f"Decision: classify as {output['label']} with confidence {output['confidence']:.2f}.")
    return "\n".join(lines)


def extract_teacher_reasoning(record, output):
    raw = record.get("raw_response_cot") or ""
    think_match = re.search(r"<think>(.*?)</think>", raw, flags=re.DOTALL | re.IGNORECASE)
    if think_match:
        reasoning = clean_reasoning_text(think_match.group(1))
        if reasoning:
            return reasoning, "think_tag"

    json_start = find_first_json_start(raw)
    if json_start > 0:
        prefix = clean_reasoning_text(raw[:json_start])
        prefix = re.sub(r"```(?:json)?\s*$", "", prefix, flags=re.IGNORECASE).strip()
        if len(prefix.split()) >= 8:
            return prefix, "prefix_before_json"

    return build_evidence_reasoning(output), "evidence_synthesis"


def format_response(reasoning, output):
    final_output = {
        "label": output["label"],
        "confidence": output["confidence"],
        "suspicious_behaviors": output.get("suspicious_behaviors") or [],
        "benign_behaviors": output.get("benign_behaviors") or [],
        "missing_or_weak_evidence": output.get("missing_or_weak_evidence") or [],
        "explanation": output.get("explanation") or "",
    }
    return "<think>\n" + reasoning.strip() + "\n</think>\n" + json.dumps(final_output, ensure_ascii=False)


def collect_correct_records(result_files):
    by_path = {}
    stats = {}

    for path in result_files:
        rows, bad_lines = load_jsonl(path)
        correct = [r for r in rows if r.get("is_correct") is True]
        stats[path] = {"rows": len(rows), "correct": len(correct), "bad_lines": bad_lines}
        for record in correct:
            sample_path = record.get("path")
            if sample_path and sample_path not in by_path:
                by_path[sample_path] = record

    return list(by_path.values()), stats


def build_dataset(tokenizer, result_files, split_lookup, min_correct, seed, max_samples=None):
    records, stats = collect_correct_records(result_files)
    print("Teacher result files:")
    for path, item in stats.items():
        print(f"  {path}: rows={item['rows']} correct={item['correct']} bad_lines={item['bad_lines']}")

    print(f"Unique correct records: {len(records)}")
    if len(records) < min_correct:
        raise ValueError(f"Need at least {min_correct} unique correct records, found {len(records)}")

    rng = random.Random(seed)
    records = sorted(records, key=lambda r: r["path"])
    rng.shuffle(records)
    if max_samples is not None:
        records = records[:max_samples]

    examples = []
    skipped_missing = 0
    skipped_too_long = 0
    skipped_bad_label = 0
    reasoning_sources = Counter()

    for record in tqdm(records, desc="Building SFT examples"):
        sample_path = record["path"]
        teacher_output = normalize_teacher_output(record)
        if teacher_output["label"] not in {"malware", "benign", "uncertain"}:
            skipped_bad_label += 1
            continue

        try:
            log_summary = load_log_summary(sample_path)
        except Exception:
            skipped_missing += 1
            continue

        reasoning, reasoning_source = extract_teacher_reasoning(record, teacher_output)
        response_text = format_response(reasoning, teacher_output)
        input_text = json.dumps(log_summary, ensure_ascii=False, sort_keys=True)
        text = ALPACA_PROMPT.format(SYSTEM_PROMPT, input_text, response_text) + tokenizer.eos_token

        token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if token_count > MAX_SEQ_LENGTH:
            skipped_too_long += 1
            continue

        source_split = record.get("split") or split_lookup.get(sample_path)
        if source_split is None:
            skipped_missing += 1
            continue

        reasoning_sources[reasoning_source] += 1
        examples.append({
            "text": text,
            "path": sample_path,
            "source_split": source_split,
            "target_label": record.get("target_label"),
            "teacher_label": teacher_output["label"],
            "token_count": token_count,
            "reasoning_source": reasoning_source,
        })

    print(f"Built {len(examples)} examples")
    print(f"Skipped missing logs: {skipped_missing}")
    print(f"Skipped too long: {skipped_too_long}")
    print(f"Skipped bad label: {skipped_bad_label}")
    print("Teacher labels:", dict(Counter(e["teacher_label"] for e in examples)))
    print("Reasoning sources:", dict(reasoning_sources))

    if not examples:
        raise ValueError("No training examples were built.")
    if len(examples) < min_correct:
        raise ValueError(f"Need at least {min_correct} usable examples after token filtering, built {len(examples)}")

    return Dataset.from_list(examples)


def split_dataset(dataset, train_split, eval_split):
    train = dataset.filter(
        lambda item: item["source_split"] == train_split,
        desc=f"Selecting {train_split}",
    )
    evaluation = dataset.filter(
        lambda item: item["source_split"] == eval_split,
        desc=f"Selecting {eval_split}",
    )
    if len(train) == 0:
        raise ValueError(f"No examples remain for training split {train_split}")
    if len(evaluation) == 0:
        raise ValueError(f"No examples remain for evaluation split {eval_split}")
    return DatasetDict({"train": train, "eval": evaluation})

def load_or_create_dataset(tokenizer, args):
    if os.path.exists(args.dataset) and not args.rebuild_dataset:
        print(f"Loading cached dataset from {args.dataset}...")
        dataset_dict = DatasetDict.load_from_disk(args.dataset)
    else:
        with open(args.splits, encoding="utf-8") as split_file:
            source_splits = json.load(split_file)
        split_lookup = {
            sample_path: split_name
            for split_name, paths in source_splits.items()
            for sample_path in paths
        }
        dataset = build_dataset(
            tokenizer=tokenizer,
            result_files=args.result_files,
            split_lookup=split_lookup,
            min_correct=args.min_correct,
            seed=args.seed,
            max_samples=args.max_samples,
        )
        dataset_dict = split_dataset(dataset, args.train_split, args.eval_split)
        print(f"Saving dataset cache to {args.dataset}...")
        dataset_dict.save_to_disk(args.dataset)

    print(dataset_dict)
    for split_name, split in dataset_dict.items():
        counts = Counter(split["teacher_label"])
        print(f"{split_name} label counts: {dict(counts)}")
        if "reasoning_source" in split.column_names:
            print(f"{split_name} reasoning sources: {dict(Counter(split['reasoning_source']))}")
    return dataset_dict


def percentile(values, q):
    if not values:
        return 0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def split_summary(split):
    token_counts = [int(x) for x in split["token_count"]] if "token_count" in split.column_names else []
    labels = Counter(split["teacher_label"]) if "teacher_label" in split.column_names else Counter()
    reasoning_sources = Counter(split["reasoning_source"]) if "reasoning_source" in split.column_names else Counter()

    summary = {
        "examples": len(split),
        "token_mean": sum(token_counts) / len(token_counts) if token_counts else 0,
        "token_p50": percentile(token_counts, 0.50),
        "token_p90": percentile(token_counts, 0.90),
        "token_p95": percentile(token_counts, 0.95),
        "token_p99": percentile(token_counts, 0.99),
        "token_max": max(token_counts) if token_counts else 0,
    }
    for label in ("malware", "benign", "uncertain"):
        summary[f"label_{label}"] = labels.get(label, 0)
        summary[f"label_{label}_ratio"] = labels.get(label, 0) / len(split) if len(split) else 0
    for source in ("think_tag", "prefix_before_json", "evidence_synthesis"):
        summary[f"reasoning_{source}"] = reasoning_sources.get(source, 0)
        summary[f"reasoning_{source}_ratio"] = reasoning_sources.get(source, 0) / len(split) if len(split) else 0
    return summary, token_counts


def build_wandb_run_name(args):
    if args.wandb_run_name:
        return args.wandb_run_name
    pieces = [
        Path(args.output).name,
        f"lr{args.learning_rate:g}",
        f"ep{args.num_train_epochs:g}",
        f"bs{args.batch_size}x{args.gradient_accumulation_steps}",
        f"r{args.lora_rank}",
    ]
    return "_".join(pieces)


def configure_wandb(args, dataset_dict):
    if args.wandb_mode == "disabled":
        return None, []

    run_name = build_wandb_run_name(args)
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ["WANDB_RUN_NAME"] = run_name
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    config = {
        "base_model": args.base_model,
        "output_model": Path(args.output).name,
        "dataset_path": args.dataset,
        "max_seq_length": MAX_SEQ_LENGTH,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "seed": args.seed,
    }

    split_metrics = {}
    token_histograms = []
    for split_name, split in dataset_dict.items():
        summary, token_counts = split_summary(split)
        for key, value in summary.items():
            split_metrics[f"dataset/{split_name}_{key}"] = value
        token_histograms.append((split_name, token_counts))

    return {
        "run_name": run_name,
        "config": config,
        "split_metrics": split_metrics,
        "token_histograms": token_histograms,
    }, ["wandb"]


def train(args):
    if args.build_dataset_only:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        load_or_create_dataset(tokenizer, args)
        print("Dataset built; exiting because --build_dataset_only was set.")
        return

    from unsloth import FastLanguageModel
    import torch
    from trl import SFTTrainer
    from transformers import TrainerCallback, TrainingArguments

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=MAX_SEQ_LENGTH,
        use_rslora=False,
        loftq_config=None,
    )
    print("Model loaded")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    dataset_dict = load_or_create_dataset(tokenizer, args)
    wandb_info, report_to = configure_wandb(args, dataset_dict)
    output_dir = args.output
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict.get("eval")

    class UsefulWandbCallback(TrainerCallback):
        def __init__(self, info):
            self.info = info
            self.best_eval_loss = None
            self.logged_dataset = False

        def _wandb(self):
            if self.info is None:
                return None
            try:
                import wandb
            except ImportError:
                return None
            if wandb.run is None:
                wandb.init(
                    project=args.wandb_project,
                    name=self.info["run_name"],
                    config=self.info["config"],
                    mode=None if args.wandb_mode == "online" else args.wandb_mode,
                )
            return wandb

        def on_train_begin(self, args_, state, control, **kwargs):
            if self.logged_dataset:
                return
            wandb = self._wandb()
            if wandb is None:
                return

            wandb.config.update(self.info["config"], allow_val_change=True)
            payload = dict(self.info["split_metrics"])
            for split_name, token_counts in self.info["token_histograms"]:
                if token_counts:
                    payload[f"dataset/{split_name}_token_count_hist"] = wandb.Histogram(token_counts)
            wandb.log(payload, step=0)
            self.logged_dataset = True

        def on_evaluate(self, args_, state, control, metrics=None, **kwargs):
            if not metrics or "eval_loss" not in metrics:
                return
            wandb = self._wandb()
            if wandb is None:
                return

            eval_loss = metrics["eval_loss"]
            if self.best_eval_loss is None or eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
            payload = {"eval/best_loss": self.best_eval_loss}
            if eval_loss < 20:
                payload["eval/perplexity"] = math.exp(eval_loss)
            if self.best_eval_loss is not None and self.best_eval_loss < 20:
                payload["eval/best_perplexity"] = math.exp(self.best_eval_loss)
            wandb.log(payload, step=state.global_step)

    callbacks = [UsefulWandbCallback(wandb_info)] if wandb_info is not None else []
    load_best_model = eval_dataset is not None and args.save_steps == args.eval_steps

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        packing=False,
        dataset_num_proc=1,
        dataset_kwargs={"append_concat_token": False},
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            output_dir=os.path.join(output_dir, "checkpoints"),
            optim="adamw_8bit",
            seed=args.seed,
            average_tokens_across_devices=False,
            dataloader_num_workers=0,
            eval_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=args.eval_steps if eval_dataset is not None else None,
            report_to=report_to,
            run_name=wandb_info["run_name"] if wandb_info is not None else None,
            logging_first_step=True,
            load_best_model_at_end=load_best_model,
            metric_for_best_model="eval_loss" if load_best_model else None,
            greater_is_better=False if load_best_model else None,
        ),
        callbacks=callbacks,
    )

    sample_text = train_dataset[0]["text"]
    input_ids = tokenizer(sample_text, truncation=True, max_length=MAX_SEQ_LENGTH)["input_ids"]
    print(f"First example tokens: {len(input_ids)}")
    print(f"Last 5 token IDs: {input_ids[-5:]}")
    print(f"Expected EOS ID: {tokenizer.eos_token_id}")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-files", nargs="+", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default=MODEL_NAME)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--build-dataset-only", action="store_true")
    parser.add_argument("--min-correct", type=int, default=2000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume-from-checkpoint")

    parser.add_argument("--num-train-epochs", type=float, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--wandb-project", default="thesis-distillation-sft")
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "online"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
