import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["UNSLOTH_NUM_WORKERS"] = "1"
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import json
from tqdm import tqdm
import argparse

from create_data import (
    VALID_CATEGORIES,
    DEFAULT_CATEGORY_IMPORTANCE,
    SYSTEM_PROMPT_TEMPLATE,
    load_references,
    sample_references,
    format_references,
    format_importance_ranking,
    build_message,
    merge_diff,
    parse_diff,
    get_reasoning,
)

MAX_SEQ_LENGTH = 32000
DTYPE = None
LOAD_IN_4BIT = True

MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
NUM_TRAIN_EPOCHS = 8
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 5e-5

DELTA_THRESHOLD = 0.15


def build_dataset(tokenizer, success_file, failed_file, sum_path,
                  corpus_path, reference_size=300):
    print(f"Loading multi-turn data from {success_file} and {failed_file}...")

    
    sum_entries, corpus_terms = load_references(sum_path, corpus_path)
    importance_ranking_str = format_importance_ranking(DEFAULT_CATEGORY_IMPORTANCE)

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    dataset_data = []
    total_steps_kept = 0
    total_steps_skipped = 0

    for filepath, is_success in [(success_file, True), (failed_file, False)]:
        if not os.path.exists(filepath):
            print(f"  Warning: {filepath} not found, skipping.")
            continue

        with open(filepath, 'r') as f:
            lines = f.readlines()

        print(f"  Processing {filepath}: {len(lines)} samples (keep_all={is_success})")

        for line in tqdm(lines, desc=f"  {os.path.basename(filepath)}"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            try:
                original_data = data.get("original_data")
                full_input = data.get("full_input")
                all_completions = data.get("all_completions", [])
                all_thinkings = data.get("all_thinkings", [])
                score_progression = data.get("score_progression", [])

                if not all_completions or len(score_progression) < 2:
                    continue

                if not original_data:
                    if full_input and isinstance(full_input, dict):
                        if 'summary' in full_input:
                            original_data = full_input['summary']
                        else:
                            original_data = full_input
                    else:
                        continue

                initial_score = score_progression[0]

                accumulated_diff = {}

                for step_idx, raw_completion in enumerate(all_completions):
                    if step_idx + 1 >= len(score_progression):
                        break

                    score_before = score_progression[step_idx]
                    score_after = score_progression[step_idx + 1]
                    delta = score_after - score_before

                    if not is_success and delta <= DELTA_THRESHOLD:
                        total_steps_skipped += 1

                        parsed, err = parse_diff(raw_completion)
                        if parsed and not err:
                            if isinstance(parsed, dict) and 'summary' in parsed:
                                parsed = parsed['summary']
                            accumulated_diff = merge_diff(accumulated_diff, parsed)
                        continue

                    user_msg = build_message(
                        original_data, initial_score, accumulated_diff,
                        score_before, step_idx
                    )

                    thinking = None
                    if step_idx < len(all_thinkings) and all_thinkings[step_idx]:
                        thinking = all_thinkings[step_idx]
                    if not thinking:
                        thinking = get_reasoning(raw_completion)
                    if not thinking:
                        thinking = "I need to add standard Windows behaviors and replace suspicious items to push P(goodware) higher."

                    parsed_diff, parse_err = parse_diff(raw_completion)
                    if not parsed_diff or parse_err:
                        total_steps_skipped += 1
                        continue

                    if isinstance(parsed_diff, dict) and 'summary' in parsed_diff:
                        parsed_diff = parsed_diff['summary']

                    if isinstance(parsed_diff, dict) and "resolved_apis" in parsed_diff.get("edit", {}):
                        del parsed_diff["edit"]["resolved_apis"]

                    response_str = f"<think>\n{thinking}\n</think>\n```json\n{json.dumps(parsed_diff)}\n```"

                    ref_data = sample_references(sum_entries, corpus_terms, reference_size)
                    reference_text = format_references(ref_data)
                    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                        valid_categories=str(VALID_CATEGORIES),
                        importance_ranking=importance_ranking_str,
                        reference_data=reference_text,
                    )

                    text = alpaca_prompt.format(
                        system_prompt,
                        user_msg,
                        response_str
                    ) + tokenizer.eos_token

                    tokens = tokenizer(text, return_tensors="pt").input_ids
                    if tokens.shape[1] > MAX_SEQ_LENGTH:
                        total_steps_skipped += 1
                        accumulated_diff = merge_diff(accumulated_diff, parsed_diff)
                        continue

                    dataset_data.append({"text": text})
                    total_steps_kept += 1

                    accumulated_diff = merge_diff(accumulated_diff, parsed_diff)

            except Exception as e:
                continue

    print(f"Dataset built: {len(dataset_data)} examples ({total_steps_kept} steps kept, {total_steps_skipped} skipped)")
    return Dataset.from_list(dataset_data)


def train(output_dir, success_file, failed_file, cache_path, sum_path,
          corpus_path, reference_size, resume_from_checkpoint=None):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        max_seq_length = MAX_SEQ_LENGTH,
        use_rslora = False,
        loftq_config = None,
    )
    print('Model loaded')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"EOS Token: {tokenizer.eos_token}")
    print(f"EOS ID: {tokenizer.eos_token_id}")

    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    if cache_path and os.path.exists(cache_path):
        dataset = Dataset.load_from_disk(cache_path)
    else:
        dataset = build_dataset(
            tokenizer,
            success_file,
            failed_file,
            sum_path,
            corpus_path,
            reference_size,
        )
        if cache_path:
            dataset.save_to_disk(cache_path)

    max_samples = 10000
    if len(dataset) > max_samples:
        print(f"Using only the first {max_samples} samples out of {len(dataset)}")
        dataset = dataset.select(range(max_samples))

    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        tokenizer = tokenizer,
        packing = False,
        dataset_num_proc = 1,
        dataset_kwargs = {
            "append_concat_token": False,
        },
        args = TrainingArguments(
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
            warmup_steps = 30,
            num_train_epochs = NUM_TRAIN_EPOCHS,
            max_steps = 600,
            learning_rate = LEARNING_RATE,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            save_steps = 100,
            save_total_limit = 5,
            output_dir = os.path.join(output_dir, "checkpoints"),
            optim = "adamw_8bit",
            seed = 3407,
            average_tokens_across_devices=False,
            dataloader_num_workers = 0,
        ),
    )

    sample_text = dataset[0]['text']
    input_ids = tokenizer(sample_text, truncation=True, max_length=MAX_SEQ_LENGTH)["input_ids"]

    print(f"Last 5 tokens IDs: {input_ids[-5:]}")
    print(f"Expected EOS ID: {tokenizer.eos_token_id}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--success-data", required=True)
    parser.add_argument("--failed-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache")
    parser.add_argument("--sum", default="sum.json")
    parser.add_argument("--corpus", default="corpus.json")
    parser.add_argument("--reference-size", type=int, default=300)
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(
        args.output,
        args.success_data,
        args.failed_data,
        args.cache,
        args.sum,
        args.corpus,
        args.reference_size,
        args.resume,
    )

