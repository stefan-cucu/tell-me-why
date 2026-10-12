
import argparse
import json
import os
import re
from pathlib import Path
from collections import Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["UNSLOTH_NUM_WORKERS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

from datasets import DatasetDict
from tqdm import tqdm

MAX_SEQ_LENGTH = 32000
DEFAULT_BASE_MODEL = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
LABELS = ["malware", "benign", "uncertain", "error"]



def build_metric_payload(records, prefix="validation"):
    total = len(records)
    correct = sum(1 for r in records if r["is_correct"])
    parsed = sum(1 for r in records if r["predicted_label"] != "error")
    payload = {
        f"{prefix}/total": total,
        f"{prefix}/correct": correct,
        f"{prefix}/accuracy": correct / total if total else 0,
        f"{prefix}/parse_rate": parsed / total if total else 0,
    }
    target_counts = Counter(r["target_label"] for r in records)
    pred_counts = Counter(r["predicted_label"] for r in records)
    for label in LABELS:
        payload[f"{prefix}/pred_{label}"] = pred_counts.get(label, 0)
        payload[f"{prefix}/pred_{label}_ratio"] = pred_counts.get(label, 0) / total if total else 0
    for label in ["malware", "benign", "uncertain"]:
        payload[f"{prefix}/target_{label}"] = target_counts.get(label, 0)
        payload[f"{prefix}/target_{label}_ratio"] = target_counts.get(label, 0) / total if total else 0
    for true_label in ["malware", "benign", "uncertain"]:
        for pred_label in LABELS:
            payload[f"{prefix}/cm_{true_label}_as_{pred_label}"] = sum(
                1 for r in records if r["target_label"] == true_label and r["predicted_label"] == pred_label
            )
    malware_rows = [r for r in records if r["target_label"] == "malware"]
    benign_rows = [r for r in records if r["target_label"] == "benign"]
    if malware_rows:
        payload[f"{prefix}/malware_recall"] = sum(1 for r in malware_rows if r["predicted_label"] == "malware") / len(malware_rows)
    if benign_rows:
        payload[f"{prefix}/benign_recall"] = sum(1 for r in benign_rows if r["predicted_label"] == "benign") / len(benign_rows)
    return payload


def init_wandb(args):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        return None
    run_name = args.wandb_run_name or f"validate_{Path(args.model_path).name}_{args.split}"
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ["WANDB_MODE"] = args.wandb_mode
    return wandb.init(
        project=args.wandb_project,
        name=run_name,
        mode=None if args.wandb_mode == "online" else args.wandb_mode,
        config={
            "model_path": args.model_path,
            "base_model": args.base_model,
            "dataset_path": args.dataset_path,
            "split": args.split,
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    )

def response_prompt_from_text(text):
    marker = "### Response:\n"
    if marker not in text:
        raise ValueError("Dataset text does not contain response marker")
    return text.split(marker, 1)[0] + marker


def normalize_label(label):
    label = str(label or "").lower().strip()
    if "malware" in label:
        return "malware"
    if "benign" in label:
        return "benign"
    if "uncertain" in label:
        return "uncertain"
    return "error"


def extract_json_object(text):
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    if start == -1:
        return None, "No JSON object start found"

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
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
                return cleaned[start:idx + 1], None

    return None, "No complete JSON object found"


def parse_prediction(text):
    json_text, err = extract_json_object(text)
    if err:
        label_match = re.search(r'"label"\s*:\s*"([^"\n]+)"', text, flags=re.IGNORECASE)
        if label_match:
            return normalize_label(label_match.group(1)), {"label": label_match.group(1)}, err
        return "error", {}, err

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        label_match = re.search(r'"label"\s*:\s*"([^"\n]+)"', json_text, flags=re.IGNORECASE)
        if label_match:
            return normalize_label(label_match.group(1)), {"label": label_match.group(1)}, f"JSON parse error: {exc}"
        return "error", {}, f"JSON parse error: {exc}"

    return normalize_label(parsed.get("label")), parsed, None


def get_confusion_matrix(y_true, y_pred):
    matrix = {t: {p: 0 for p in LABELS} for t in ["malware", "benign", "uncertain"]}
    for true, pred in zip(y_true, y_pred):
        true = normalize_label(true)
        pred = normalize_label(pred)
        if true not in matrix:
            matrix[true] = {p: 0 for p in LABELS}
        if pred not in matrix[true]:
            matrix[true][pred] = 0
        matrix[true][pred] += 1
    return matrix


def print_metrics(records):
    total = len(records)
    correct = sum(1 for r in records if r["is_correct"])
    parsed = sum(1 for r in records if r["predicted_label"] != "error")

    print("\nEvaluation summary")
    print(f"  total: {total}")
    print(f"  correct: {correct}")
    if total:
        print(f"  accuracy: {correct / total * 100:.2f}%")
        print(f"  parse rate: {parsed / total * 100:.2f}%")

    print("  predicted labels:", dict(Counter(r["predicted_label"] for r in records)))
    print("  target labels:", dict(Counter(r["target_label"] for r in records)))

    matrix = get_confusion_matrix(
        [r["target_label"] for r in records],
        [r["predicted_label"] for r in records],
    )
    print("\nConfusion matrix")
    print(f"{'true \\ pred':>14} | " + " | ".join(f"{p[:9]:>9}" for p in LABELS))
    print("-" * 62)
    for true in ["malware", "benign", "uncertain"]:
        if sum(matrix[true].values()) == 0:
            continue
        print(f"{true:>14} | " + " | ".join(f"{matrix[true][p]:>9}" for p in LABELS))


def load_eval_dataset(dataset_path, split, num_samples):
    dataset_dict = DatasetDict.load_from_disk(dataset_path)
    if split not in dataset_dict:
        available = ", ".join(dataset_dict.keys())
        raise ValueError(f"Split '{split}' not found in {dataset_path}. Available: {available}")

    dataset = dataset_dict[split]
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    return dataset


def evaluate(args):
    import torch
    from unsloth import FastLanguageModel
    from peft import PeftModel

    print(f"Loading dataset split '{args.split}' from {args.dataset_path}...")
    dataset = load_eval_dataset(args.dataset_path, args.split, args.num_samples)
    print(f"Loaded {len(dataset)} examples")

    is_adapter = os.path.exists(os.path.join(args.model_path, "adapter_config.json")) and not os.path.exists(os.path.join(args.model_path, "config.json"))
    if is_adapter:
        print(f"Loading base model from {args.base_model}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        print(f"Loading LoRA adapter from {args.model_path}...")
        model = PeftModel.from_pretrained(model, args.model_path)
    else:
        print(f"Loading full model from {args.model_path}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    wandb_run = init_wandb(args)
    records = []
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    with open(args.output_jsonl, "w") as out_f:
        for idx, item in enumerate(tqdm(dataset, desc="Validating")):
            prompt = response_prompt_from_text(item["text"])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(model.device)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=args.top_p,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            completion_ids = generated[0][input_len:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            pred_label, parsed_json, parse_error = parse_prediction(completion)
            target_label = normalize_label(item.get("target_label") or item.get("teacher_label"))

            record = {
                "idx": idx,
                "path": item.get("path"),
                "target_label": target_label,
                "teacher_label": normalize_label(item.get("teacher_label")),
                "predicted_label": pred_label,
                "is_correct": pred_label == target_label,
                "parse_error": parse_error,
                "json_output": parsed_json,
                "raw_completion": completion,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            records.append(record)

            if args.log_every and (idx + 1) % args.log_every == 0:
                so_far = sum(1 for r in records if r["is_correct"])
                accuracy = so_far / len(records)
                print(f"Processed {idx + 1}/{len(dataset)} accuracy={accuracy * 100:.2f}%")
                if wandb_run is not None:
                    wandb_run.log({
                        "validation/progress_examples": idx + 1,
                        "validation/progress_accuracy": accuracy,
                        "validation/progress_parse_rate": sum(1 for r in records if r["predicted_label"] != "error") / len(records),
                    }, step=idx + 1)

    print_metrics(records)
    if wandb_run is not None:
        wandb_run.log(build_metric_payload(records), step=len(records))
        wandb_run.finish()
    print(f"\nWrote predictions to {args.output_jsonl}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model_path", required=True)
    parser.add_argument("--dataset", dest="dataset_path", required=True)
    parser.add_argument("--output", dest="output_jsonl", required=True)
    parser.add_argument("--base-model", dest="base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--split", default="eval", choices=["train", "eval"])
    parser.add_argument("--num-samples", dest="num_samples", type=int)
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    parser.add_argument("--log-every", dest="log_every", type=int, default=10)
    parser.add_argument("--wandb-project", dest="wandb_project", default="thesis-distillation-validation")
    parser.add_argument("--wandb-run-name", dest="wandb_run_name")
    parser.add_argument(
        "--wandb-mode",
        dest="wandb_mode",
        choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "online"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
