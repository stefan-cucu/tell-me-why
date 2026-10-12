
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from fusion.data import encoder_text_from_text, response_prompt_from_text
from fusion.inference import (
    BINARY_LABELS,
    classifier_authoritative_fallback_json,
    classifier_output_from_decision,
    load_dataset,
    load_model,
    normalize_label,
    parse_prediction,
)

DEFAULT_BASE_MODEL = "unsloth/Qwen3-8B-unsloth-bnb-4bit"


def classifier_reasoning_prefix(label, confidence):
    return (
        "<think>\n"
        f"Classifier signal: classifier predicts {label} with confidence {confidence:.4f}.\n"
        "Treat this classifier decision as authoritative.\n"
    )


def authoritative_json_prefix(label, confidence):
    return (
        "{\n"
        f'  "label": "{label}",\n'
        f'  "confidence": {confidence:.4f},\n'
        '  "suspicious_behaviors": '
    )


def tokenize_text(tokenizer, text, device):
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return {key: value.to(device) for key, value in encoded.items()}


def append_tokens(base, suffix):
    return {
        "input_ids": torch.cat([base["input_ids"], suffix["input_ids"]], dim=1),
        "attention_mask": torch.cat([base["attention_mask"], suffix["attention_mask"]], dim=1),
    }


def generate_reasoning_and_json(model, tokenizer, prompt_inputs, encoder, label, confidence, args):
    device = prompt_inputs["input_ids"].device
    prompt_len = prompt_inputs["input_ids"].shape[1]
    if args.free_json:
        
        reasoning_prefix = "<think>\n"
    else:
        reasoning_prefix = classifier_reasoning_prefix(label, confidence)
    prefix_tokens = tokenize_text(tokenizer, reasoning_prefix, device)
    reasoning_inputs = append_tokens(prompt_inputs, prefix_tokens)
    reasoning_start = reasoning_inputs["input_ids"].shape[1]

    generated_reasoning_ids = model.generate(
        input_ids=reasoning_inputs["input_ids"],
        attention_mask=reasoning_inputs["attention_mask"],
        encoder_input_ids=encoder["input_ids"],
        encoder_attention_mask=encoder["attention_mask"],
        max_new_tokens=args.max_reasoning_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature if args.temperature > 0 else None,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        stop_strings=("</think>",),
        decode_tokenizer=tokenizer,
        completion_start_idx=reasoning_start,
        stop_check_interval=args.reasoning_stop_check_interval,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    raw_reasoning = tokenizer.decode(
        generated_reasoning_ids[0][reasoning_start:], skip_special_tokens=True
    )
    reasoning_closed = "</think>" in raw_reasoning
    generated_reasoning = raw_reasoning.split("</think>", 1)[0].strip()

    closing = "\n</think>\n"
    if args.free_json:
        
        json_prefix = "{\n"
    else:
        json_prefix = authoritative_json_prefix(label, confidence)
    authoritative_tail = reasoning_prefix + generated_reasoning + closing + json_prefix
    authoritative_tokens = tokenize_text(tokenizer, authoritative_tail, device)
    json_inputs = append_tokens(prompt_inputs, authoritative_tokens)
    json_continuation_start = json_inputs["input_ids"].shape[1]

    generated_json_ids = model.generate(
        input_ids=json_inputs["input_ids"],
        attention_mask=json_inputs["attention_mask"],
        encoder_input_ids=encoder["input_ids"],
        encoder_attention_mask=encoder["attention_mask"],
        max_new_tokens=args.max_json_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature if args.temperature > 0 else None,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        stop_on_complete_json=True,
        decode_tokenizer=tokenizer,
        completion_start_idx=prompt_len,
        stop_check_interval=args.json_stop_check_interval,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    json_continuation = tokenizer.decode(
        generated_json_ids[0][json_continuation_start:], skip_special_tokens=True
    ).strip()
    completion = authoritative_tail + json_continuation
    return {
        "completion": completion,
        "reasoning_prefix": reasoning_prefix,
        "generated_reasoning": generated_reasoning,
        "raw_reasoning_continuation": raw_reasoning,
        "reasoning_closed_by_model": reasoning_closed,
        "reasoning_truncated": not reasoning_closed,
        "json_prefix": json_prefix,
        "generated_json_continuation": json_continuation,
        "reasoning_tokens": len(tokenizer(generated_reasoning, add_special_tokens=False)["input_ids"]),
        "json_continuation_tokens": len(tokenizer(json_continuation, add_special_tokens=False)["input_ids"]),
    }


def summarize(records):
    total = len(records)
    correct = sum(record["is_correct"] for record in records)
    raw_parsed = sum(record["raw_parse_error"] is None for record in records)
    closed = sum(record["reasoning_closed_by_model"] for record in records)
    nonempty_reasoning = sum(bool(record["generated_reasoning"].strip()) for record in records)
    agreement = sum(record["predicted_label"] == record["classifier_label"] for record in records)
    confidence_agreement = sum(
        abs(float(record["json_output"].get("confidence", -1)) - record["classifier_confidence"]) <= 5e-4
        for record in records
    )
    metrics = {
        "validation/total": total,
        "validation/accuracy": correct / total if total else 0.0,
        "validation/raw_json5_parse_rate": raw_parsed / total if total else 0.0,
        "validation/reasoning_closure_rate": closed / total if total else 0.0,
        "validation/nonempty_reasoning_rate": nonempty_reasoning / total if total else 0.0,
        "validation/classifier_label_agreement": agreement / total if total else 0.0,
        "validation/classifier_confidence_agreement": confidence_agreement / total if total else 0.0,
        "validation/mean_reasoning_tokens": (
            sum(record["reasoning_tokens"] for record in records) / total if total else 0.0
        ),
    }
    print("\nEvaluation summary")
    print(f"  total: {total}")
    print(f"  accuracy: {metrics['validation/accuracy'] * 100:.2f}%")
    print(f"  raw JSON5 parse rate: {metrics['validation/raw_json5_parse_rate'] * 100:.2f}%")
    print(f"  reasoning closure rate: {metrics['validation/reasoning_closure_rate'] * 100:.2f}%")
    print(f"  non-empty reasoning rate: {metrics['validation/nonempty_reasoning_rate'] * 100:.2f}%")
    print(f"  classifier label agreement: {metrics['validation/classifier_label_agreement'] * 100:.2f}%")
    print(f"  classifier confidence agreement: {metrics['validation/classifier_confidence_agreement'] * 100:.2f}%")
    print(f"  mean generated reasoning tokens: {metrics['validation/mean_reasoning_tokens']:.1f}")
    print("  predicted labels:", dict(Counter(record["predicted_label"] for record in records)))
    print("  target labels:", dict(Counter(record["target_label"] for record in records)))
    return metrics


def init_wandb(args):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        return None
    os.environ["WANDB_MODE"] = args.wandb_mode
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"validate_{Path(args.fusion_dir).name}_{args.split}",
        config=vars(args),
    )


def evaluate(args):
    dataset = load_dataset(args.dataset_path, args.split, args.num_samples)
    print(f"Loaded {len(dataset)} examples from {args.dataset_path}/{args.split}")
    model, fusion_config = load_model(args)
    device = next(model.memory_aux_head.parameters()).device

    llm_tokenizer = AutoTokenizer.from_pretrained(args.fusion_dir, trust_remote_code=True)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    encoder_tokenizer = AutoTokenizer.from_pretrained(
        fusion_config.embedding_model_name, trust_remote_code=True
    )
    if encoder_tokenizer.pad_token is None:
        encoder_tokenizer.pad_token = encoder_tokenizer.eos_token

    wandb_run = init_wandb(args)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    records = []
    with open(args.output_jsonl, "w") as output_file:
        for idx, item in enumerate(tqdm(dataset, desc="Validating reasoning fusion")):
            prompt = response_prompt_from_text(item["text"])
            encoder_text = encoder_text_from_text(item["text"])
            prompt_inputs = llm_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_seq_length,
                add_special_tokens=False,
            )
            encoder = encoder_tokenizer(
                encoder_text,
                return_tensors="pt",
                truncation=True,
                max_length=args.encoder_max_length,
                add_special_tokens=True,
            )
            prompt_inputs = {key: value.to(device) for key, value in prompt_inputs.items()}
            encoder = {key: value.to(device) for key, value in encoder.items()}

            with torch.no_grad():
                decision = model.classifier_decision(
                    encoder["input_ids"], encoder["attention_mask"]
                )
            classifier_label, classifier_confidence, classifier_probs = classifier_output_from_decision(
                decision
            )
            if classifier_label not in BINARY_LABELS.values() or classifier_confidence is None:
                raise RuntimeError(f"Classifier returned an invalid decision for example {idx}")

            with torch.no_grad():
                generated = generate_reasoning_and_json(
                    model,
                    llm_tokenizer,
                    prompt_inputs,
                    encoder,
                    classifier_label,
                    classifier_confidence,
                    args,
                )
            raw_label, raw_json, raw_parse_error = parse_prediction(generated["completion"])
            if args.free_json:
                final_json = dict(raw_json) if isinstance(raw_json, dict) else {}
                final_json["forced_from_classifier"] = False
                predicted_label = raw_label
            else:
                final_json = classifier_authoritative_fallback_json(
                    classifier_label, classifier_confidence, raw_json
                )
                final_json["forced_from_classifier"] = True
                predicted_label = classifier_label
            target_label = normalize_label(item.get("target_label") or item.get("teacher_label"))
            record = {
                "idx": idx,
                "path": item.get("path"),
                "target_label": target_label,
                "teacher_label": normalize_label(item.get("teacher_label")),
                "predicted_label": predicted_label,
                "raw_predicted_label": raw_label,
                "classifier_label": classifier_label,
                "classifier_confidence": classifier_confidence,
                "classifier_probs": classifier_probs,
                "is_correct": predicted_label == target_label,
                "raw_is_correct": raw_label == target_label,
                "raw_parse_error": raw_parse_error,
                "json_output": final_json,
                "raw_json_output": raw_json,
                **generated,
            }
            output_file.write(json.dumps(record) + "\n")
            output_file.flush()
            records.append(record)

            if args.log_every and (idx + 1) % args.log_every == 0:
                progress = summarize(records)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"progress/{key.split('/', 1)[-1]}": value for key, value in progress.items()},
                        step=idx + 1,
                    )

    metrics = summarize(records)
    if wandb_run is not None:
        wandb_run.log(metrics, step=len(records))
        wandb_run.finish()
    print(f"\nWrote predictions to {args.output_jsonl}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion-model", dest="fusion_dir", required=True)
    parser.add_argument("--dataset", dest="dataset_path", required=True)
    parser.add_argument("--output", dest="output_jsonl", required=True)
    parser.add_argument("--base-model", dest="base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--split", default="eval", choices=["train", "eval"])
    parser.add_argument("--num-samples", dest="num_samples", type=int)
    parser.add_argument("--max-seq-length", dest="max_seq_length", type=int, default=28672)
    parser.add_argument("--encoder-max-length", dest="encoder_max_length", type=int, default=16384)
    parser.add_argument("--max-reasoning-tokens", dest="max_reasoning_tokens", type=int, default=768)
    parser.add_argument("--max-json-tokens", dest="max_json_tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=1.05)
    parser.add_argument("--no-repeat-ngram-size", dest="no_repeat_ngram_size", type=int, default=12)
    parser.add_argument("--reasoning-stop-check-interval", dest="reasoning_stop_check_interval", type=int, default=2)
    parser.add_argument("--json-stop-check-interval", dest="json_stop_check_interval", type=int, default=4)
    parser.add_argument("--log-every", dest="log_every", type=int, default=10)
    parser.add_argument("--classifier-threshold", dest="classifier_threshold", type=float, default=0.5)
    parser.add_argument("--free-json", dest="free_json", action="store_true")
    parser.add_argument("--load-llm-in-4bit", dest="load_llm_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", dest="wandb_project", default="thesis-fusion-validation")
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
