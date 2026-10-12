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
from create_data import load_references, sample_references, format_references, VALID_CATEGORIES, DEFAULT_CATEGORY_IMPORTANCE

MAX_SEQ_LENGTH = 32000
DTYPE = None
LOAD_IN_4BIT = True

MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
NUM_TRAIN_EPOCHS = 8
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 5e-5

def build_dataset(tokenizer, filepath, sum_path, corpus_path, reference_size, report_root=None):
    print(f"Loading dataset from {filepath}...")
    
    dataset_data = []
    
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    SYSTEM_PROMPT = """You are a cybersecurity expert specializing in malware evasion analysis.
Your task is to generate an adversarial perturbation (a "diff" in JSON format) that, when applied to a malware behavioral log, confuses a classifier into thinking it is "goodware". 

You have TWO types of operations available:
1. **add**: Insert new entries into categories. These should be characteristic of legitimate software behavior.
2. **edit**: Replace existing suspicious/malicious entries with benign-looking alternatives that a goodware program would produce. 
You are not allowed to edit api calls (resolved_apis), but you CAN add new api entries.

Rules:
1. Output only the JSON diff and nothing else.
2. Use this exact format:
{
  "add": {
    "category": ["new_entry_1", "new_entry_2"]
  },
  "edit": {
    "category": [
      {"old": "exact_existing_entry", "new": "replacement_entry"}
    ]
  }
}
3. Valid categories: """ + str(VALID_CATEGORIES) + """
4. Each "old" value in edits MUST exactly match an entry in the original log.
5. Replacement entries should look like legitimate software behavior.
6. Added entries must be realistic and not duplicate existing entries.
7. Do not remove entries — only add or replace.
8. Make sure to make atleast 100 additions and edits to each log file.
9. Focus your edits on the most impactful categories (see importance ranking below).

## Category Importance (from classifier ablation analysis)
The classifier relies on these categories most heavily. Focus your perturbations on the top-ranked ones:
{importance_ranking}

## Reference: Common Goodware Behavioral Entries
{reference_data}
"""

    
    sorted_cats = sorted(DEFAULT_CATEGORY_IMPORTANCE.items(), key=lambda x: x[1], reverse=True)
    importance_lines = []
    for rank, (cat, score) in enumerate(sorted_cats, 1):
        importance_lines.append(f"  {rank}. {cat} (importance: {score:.4f})")
    importance_ranking_str = "\n".join(importance_lines)

    print("Loading reference data (sum.json + corpus.json) for prompt context...")
    sum_entries, corpus_terms = load_references(sum_path, corpus_path)

    with open(filepath, 'r') as f:
        for line in tqdm(f.readlines(), desc="Processing lines"):
            try:
                data = json.loads(line)
                
                full_input = data.get("full_input")
                family = data.get("original_family", "")
                filename = data.get("filename", "")
                
                if not full_input and report_root and family and filename and family != "unknown" and filename != "unknown":
                    clean_name = filename.removesuffix(".json")
                    report_path = os.path.join(report_root, family, "reports_summary", f"{clean_name}.json")
                    if os.path.exists(report_path):
                        with open(report_path, "r") as report_file:
                            full_input = json.load(report_file)
                
                if not full_input:
                    
                    
                    parsed_diff = data.get("parsed_diff", {})
                    if not parsed_diff:
                        continue
                    thinking = data.get("thinking", "I need to add standard Windows behaviors and replace suspicious items to bypass the classifier.")
                    if "resolved_apis" in parsed_diff:
                        del parsed_diff["resolved_apis"]
                    response_str = f"<think>\n{thinking}\n</think>\n```json\n{json.dumps(parsed_diff)}\n```"
                    
                    ref_data = sample_references(sum_entries, corpus_terms, reference_size)
                    reference_text = format_references(ref_data)
                    current_system_prompt = SYSTEM_PROMPT.replace("{reference_data}", reference_text).replace("{importance_ranking}", importance_ranking_str)
                    
                    text = alpaca_prompt.format(
                        current_system_prompt,
                        '{"files": [], "read_files": [], "write_files": [], "keys": [], "read_keys": [], "write_keys": []}',
                        response_str
                    ) + tokenizer.eos_token
                    tokens = tokenizer(text, return_tensors="pt").input_ids
                    if tokens.shape[1] > MAX_SEQ_LENGTH:
                        continue
                    dataset_data.append({"text": text})
                    continue
                
                if 'summary' in full_input and isinstance(full_input['summary'], dict):
                    target_data = full_input['summary']
                else:
                    target_data = full_input
                    
                input_json_str = json.dumps(target_data)
                
                thinking = data.get("thinking", "")
                if not thinking:
                    thinking = "I need to add standard Windows behaviors and replace suspicious items to bypass the classifier."
                    
                parsed_diff = data.get("parsed_diff", {})
                
                if "resolved_apis" in parsed_diff:
                    del parsed_diff["resolved_apis"]
                
                response_str = f"<think>\n{thinking}\n</think>\n```json\n{json.dumps(parsed_diff)}\n```"
                
                
                ref_data = sample_references(sum_entries, corpus_terms, reference_size)
                reference_text = format_references(ref_data)
                current_system_prompt = SYSTEM_PROMPT.replace("{reference_data}", reference_text).replace("{importance_ranking}", importance_ranking_str)
                
                text = alpaca_prompt.format(
                    current_system_prompt,
                    input_json_str,
                    response_str
                ) + tokenizer.eos_token
                
                tokens = tokenizer(text, return_tensors="pt").input_ids
                if tokens.shape[1] > MAX_SEQ_LENGTH:
                    continue
                
                dataset_data.append({"text": text})
            except Exception as e:
                pass
                
    print(f"Successfully loaded {len(dataset_data)} samples.")
    return Dataset.from_list(dataset_data)

def train(output_dir, data_path, cache_path, sum_path, corpus_path,
          reference_size, report_root=None, resume_from_checkpoint=None):
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
        print(f"Loading dataset from {cache_path}...")
        dataset = Dataset.load_from_disk(cache_path)
    else:
        dataset = build_dataset(
            tokenizer, data_path, sum_path, corpus_path, reference_size, report_root
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
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache")
    parser.add_argument("--sum", default="sum.json")
    parser.add_argument("--corpus", default="corpus.json")
    parser.add_argument("--reference-size", type=int, default=300)
    parser.add_argument("--report-root")
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(
        args.output,
        args.data,
        args.cache,
        args.sum,
        args.corpus,
        args.reference_size,
        args.report_root,
        args.resume,
    )
