
import argparse
import math
import os
from collections import Counter
from pathlib import Path


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments
from peft import PeftModel

from fusion.data import FusionSFTCollator, load_fusion_dataset
from fusion.model import build_fusion_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["UNSLOTH_NUM_WORKERS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

DEFAULT_BASE_MODEL = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"



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


def summarize_dataset(split):
    labels = Counter(str(x).lower() for x in split["target_label"]) if "target_label" in split.column_names else Counter()
    teacher_labels = Counter(str(x).lower() for x in split["teacher_label"]) if "teacher_label" in split.column_names else Counter()
    reasoning = Counter(split["reasoning_source"]) if "reasoning_source" in split.column_names else Counter()
    token_counts = [int(x) for x in split["token_count"]] if "token_count" in split.column_names else []
    teacher_reasoning_tokens = [int(x) for x in split["teacher_reasoning_token_count"]] if "teacher_reasoning_token_count" in split.column_names else []
    assistant_prefix_tokens = [int(x) for x in split["assistant_prefix_token_count"]] if "assistant_prefix_token_count" in split.column_names else []
    classifier_confidences = [float(x) for x in split["classifier_confidence"]] if "classifier_confidence" in split.column_names else []
    summary = {
        "examples": len(split),
        "token_mean": sum(token_counts) / len(token_counts) if token_counts else 0,
        "token_p50": percentile(token_counts, 0.50),
        "token_p90": percentile(token_counts, 0.90),
        "token_p95": percentile(token_counts, 0.95),
        "token_p99": percentile(token_counts, 0.99),
        "token_max": max(token_counts) if token_counts else 0,
        "teacher_reasoning_token_mean": sum(teacher_reasoning_tokens) / len(teacher_reasoning_tokens) if teacher_reasoning_tokens else 0,
        "teacher_reasoning_token_p95": percentile(teacher_reasoning_tokens, 0.95),
        "assistant_prefix_token_mean": sum(assistant_prefix_tokens) / len(assistant_prefix_tokens) if assistant_prefix_tokens else 0,
        "classifier_confidence_mean": sum(classifier_confidences) / len(classifier_confidences) if classifier_confidences else 0,
    }
    for label in ("malware", "benign", "uncertain"):
        summary[f"target_{label}"] = labels.get(label, 0)
        summary[f"target_{label}_ratio"] = labels.get(label, 0) / len(split) if len(split) else 0
        summary[f"teacher_{label}"] = teacher_labels.get(label, 0)
        summary[f"teacher_{label}_ratio"] = teacher_labels.get(label, 0) / len(split) if len(split) else 0
    for source in ("think_tag", "prefix_before_json", "evidence_synthesis"):
        summary[f"reasoning_{source}"] = reasoning.get(source, 0)
        summary[f"reasoning_{source}_ratio"] = reasoning.get(source, 0) / len(split) if len(split) else 0
    return summary, token_counts


def build_wandb_dataset_payload(train_dataset, eval_dataset):
    payload = {}
    histograms = []
    for split_name, split in (("train", train_dataset), ("eval", eval_dataset)):
        if split is None:
            continue
        summary, token_counts = summarize_dataset(split)
        for key, value in summary.items():
            payload[f"dataset/{split_name}_{key}"] = value
        histograms.append((split_name, token_counts))
    return payload, histograms


class FusionWandbCallback(TrainerCallback):
    def __init__(self, args, train_dataset, eval_dataset, trainable_params, total_params, filter_stats=None):
        self.args = args
        self.dataset_payload, self.token_histograms = build_wandb_dataset_payload(train_dataset, eval_dataset)
        self.filter_stats = filter_stats or {}
        self.trainable_params = trainable_params
        self.total_params = total_params
        self.logged_dataset = False

    def _wandb(self):
        if self.args.wandb_mode == "disabled":
            return None
        try:
            import wandb
        except ImportError:
            return None
        return wandb

    def on_train_begin(self, args_, state, control, model=None, **kwargs):
        if self.logged_dataset:
            return
        wandb = self._wandb()
        if wandb is None or wandb.run is None:
            return
        payload = dict(self.dataset_payload)
        payload["fusion/trainable_params"] = self.trainable_params
        payload["fusion/total_params"] = self.total_params
        payload["fusion/trainable_param_ratio"] = self.trainable_params / self.total_params if self.total_params else 0
        payload["fusion/aux_ce_weight"] = self.args.aux_ce_weight
        payload["fusion/num_memory_tokens"] = self.args.num_memory_tokens
        payload["fusion/cross_attn_hidden_size"] = self.args.cross_attn_hidden_size
        payload["fusion/num_cross_attention_heads"] = self.args.num_cross_attention_heads
        payload["fusion/encoder_max_length"] = self.args.encoder_max_length
        payload["fusion/train_sft_adapter"] = int(self.args.train_sft_adapter)
        payload["fusion/truncate_strategy"] = self.args.truncate_strategy
        payload["fusion/no_eval"] = int(self.args.no_eval)
        payload["fusion/length_order"] = self.args.length_order
        payload["fusion/decision_source"] = self.args.decision_source
        payload["fusion/classifier_label_source"] = self.args.classifier_label_source
        payload["fusion/classifier_threshold"] = self.args.classifier_threshold
        for split_name, stats in self.filter_stats.items():
            for key, value in stats.items():
                if key == "bucket_counts":
                    for bucket, count in value.items():
                        payload[f"fit/{split_name}_bucket_{bucket}"] = count
                elif isinstance(value, (int, float)):
                    payload[f"fit/{split_name}_{key}"] = value
        for split_name, token_counts in self.token_histograms:
            if token_counts:
                payload[f"dataset/{split_name}_token_count_hist"] = wandb.Histogram(token_counts)
        wandb.log(payload, step=0)
        self.logged_dataset = True

    def on_log(self, args_, state, control, model=None, logs=None, **kwargs):
        wandb = self._wandb()
        if wandb is None or wandb.run is None or model is None:
            return
        metrics = getattr(model, "last_fusion_metrics", None)
        if metrics:
            wandb.log(metrics, step=state.global_step)

def parse_layers(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value)
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def get_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_llm(args, dtype):
    quant_config = None
    if args.load_llm_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    device_map = {"": 0} if torch.cuda.is_available() else None
    llm = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        quantization_config=quant_config,
        device_map=device_map,
    )
    llm.config.use_cache = False
    if args.gradient_checkpointing:
        llm.gradient_checkpointing_enable()

    if args.sft_adapter_path:
        state = "trainable" if args.train_sft_adapter else "frozen"
        print(f"Loading SFT adapter from {args.sft_adapter_path} ({state})")
        llm = PeftModel.from_pretrained(llm, args.sft_adapter_path, is_trainable=args.train_sft_adapter)

    if args.gradient_checkpointing and hasattr(llm, "enable_input_require_grads"):
        llm.enable_input_require_grads()
    return llm


def load_embedding_tower(args, dtype):
    device_map = {"": 0} if torch.cuda.is_available() else None
    model = AutoModel.from_pretrained(
        args.embedding_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()
    return model


def train(args):
    dtype = get_dtype()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using dtype={dtype}, device={device}")

    llm_tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter_path or args.base_model, trust_remote_code=True)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_tokenizer.padding_side = "right"

    encoder_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, trust_remote_code=True)
    if encoder_tokenizer.pad_token is None:
        encoder_tokenizer.pad_token = encoder_tokenizer.eos_token
    encoder_tokenizer.padding_side = "right"

    train_dataset, eval_dataset, filter_stats = load_fusion_dataset(
        args.dataset_path,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_seq_length=args.max_seq_length,
        truncate_strategy=args.truncate_strategy,
        no_eval=args.no_eval,
        length_order=args.length_order,
    )
    print(f"Train examples: {len(train_dataset)}")
    if "train" in filter_stats:
        item = filter_stats["train"]
        print(
            "Train fit stats: "
            f"original={item.get('original_examples')} kept={item.get('kept_examples')} "
            f"skipped={item.get('skipped_examples')} keep_ratio={item.get('keep_ratio', 0):.3f}"
        )
    if eval_dataset is not None:
        print(f"Eval examples: {len(eval_dataset)}")
        if "eval" in filter_stats:
            item = filter_stats["eval"]
            print(
                "Eval fit stats: "
                f"original={item.get('original_examples')} kept={item.get('kept_examples')} "
                f"skipped={item.get('skipped_examples')} keep_ratio={item.get('keep_ratio', 0):.3f}"
            )

    llm = load_llm(args, dtype)
    embedding_model = load_embedding_tower(args, dtype)
    model = build_fusion_model(
        llm=llm,
        embedding_model=embedding_model,
        classifier_path=args.classifier_path,
        num_memory_tokens=args.num_memory_tokens,
        cross_attn_hidden_size=args.cross_attn_hidden_size,
        num_cross_attention_heads=args.num_cross_attention_heads,
        cross_attn_layers=parse_layers(args.cross_attn_layers),
        aux_ce_weight=args.aux_ce_weight,
        embedding_model_name=args.embedding_model,
        classifier_threshold=args.classifier_threshold,
    )
    model.move_fusion_modules(device=device, dtype=dtype)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    collator = FusionSFTCollator(
        llm_tokenizer=llm_tokenizer,
        encoder_tokenizer=encoder_tokenizer,
        max_seq_length=args.max_seq_length,
        encoder_max_length=args.encoder_max_length,
        truncate_strategy=args.truncate_strategy,
        decision_source=args.decision_source,
        classifier_label_source=args.classifier_label_source,
    )

    report_to = [] if args.wandb_mode == "disabled" else ["wandb"]
    if args.wandb_mode != "disabled":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ["WANDB_RUN_NAME"] = args.wandb_run_name or Path(args.output_dir).name
        os.environ["WANDB_MODE"] = args.wandb_mode

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        optim=args.optim,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=report_to,
        run_name=args.wandb_run_name or Path(args.output_dir).name,
        logging_first_step=True,
        prediction_loss_only=True,
        seed=args.seed,
    )

    callbacks = []
    if args.wandb_mode != "disabled":
        callbacks.append(FusionWandbCallback(args, train_dataset, eval_dataset, trainable, total, filter_stats))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print(f"Saving fusion model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    llm_tokenizer.save_pretrained(args.output_dir)
    encoder_tokenizer.save_pretrained(os.path.join(args.output_dir, "embedding_tokenizer"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-adapter", dest="sft_adapter_path", required=True)
    parser.add_argument("--classifier", dest="classifier_path", required=True)
    parser.add_argument("--dataset", dest="dataset_path", required=True)
    parser.add_argument("--output", dest="output_dir", required=True)
    parser.add_argument("--base-model", dest="base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--embedding-model", dest="embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--max-seq-length", dest="max_seq_length", type=int, default=28672)
    parser.add_argument("--encoder-max-length", dest="encoder_max_length", type=int, default=16384)
    parser.add_argument("--truncate-strategy", dest="truncate_strategy", choices=["none", "preserve_response"], default="none")
    parser.add_argument("--no-eval", dest="no_eval", action="store_true")
    parser.add_argument("--length-order", dest="length_order", choices=["none", "ascending", "descending"], default="ascending")
    parser.add_argument("--num-memory-tokens", dest="num_memory_tokens", type=int, default=16)
    parser.add_argument("--cross-attn-hidden-size", dest="cross_attn_hidden_size", type=int, default=128)
    parser.add_argument("--num-cross-attention-heads", dest="num_cross_attention_heads", type=int, default=4)
    parser.add_argument("--cross-attn-layers", dest="cross_attn_layers", default="28,35")
    parser.add_argument("--aux-ce-weight", dest="aux_ce_weight", type=float, default=0.2)
    parser.add_argument("--classifier-threshold", dest="classifier_threshold", type=float, default=0.5)
    parser.add_argument("--decision-source", dest="decision_source", choices=["target", "classifier_column", "runtime_classifier"], default="classifier_column")
    parser.add_argument("--classifier-label-source", dest="classifier_label_source", choices=["target", "classifier_column"], default="classifier_column")
    parser.add_argument("--max-train-samples", dest="max_train_samples", type=int)
    parser.add_argument("--max-eval-samples", dest="max_eval_samples", type=int)

    parser.add_argument("--num-train-epochs", dest="num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=-1)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", dest="gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=5e-5)
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument("--warmup-steps", dest="warmup_steps", type=int, default=100)
    parser.add_argument("--logging-steps", dest="logging_steps", type=int, default=1)
    parser.add_argument("--save-steps", dest="save_steps", type=int, default=250)
    parser.add_argument("--eval-steps", dest="eval_steps", type=int, default=250)
    parser.add_argument("--save-total-limit", dest="save_total_limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume-from-checkpoint", dest="resume_from_checkpoint")
    parser.add_argument("--load-llm-in-4bit", dest="load_llm_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-sft-adapter", dest="train_sft_adapter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", dest="wandb_project", default="thesis-fusion-sft")
    parser.add_argument("--wandb-run-name", dest="wandb_run_name")
    parser.add_argument(
        "--wandb-mode",
        dest="wandb_mode",
        choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "online"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
