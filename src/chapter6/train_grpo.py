import os
import glob
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_SOCKET_IFNAME'] = 'lo'

os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS' 
os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
os.environ['VLLM_DISABLE_FLASHINFER'] = '1'
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
import torch
import wandb
import json
import random
import ast
import re
import numpy as np
from collections import deque
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from typing import List
import argparse
import augmented_dataset
from common import MLPClassifier, ReportEmbedder, load_json_report

from create_data import (
    VALID_CATEGORIES,
    DEFAULT_CATEGORY_IMPORTANCE,
    SYSTEM_PROMPT_TEMPLATE,
    FEW_SHOT_EXAMPLES,
    load_references,
    sample_references,
    format_references,
    format_importance_ranking,
    build_message,
)

MAX_SEQ_LENGTH = 32000

CLASSES = ['goodware', 'malware']

_rolling_goodware_probs = deque(maxlen=120)


def parse_diff(text):
    text_no_think = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    match = re.search(r'```json\s*(.*?)\s*```', text_no_think, re.DOTALL)
    if match:
        try: return json.loads(match.group(1)), None
        except Exception as e: pass

    start = text_no_think.find('{')
    end = text_no_think.rfind('}')
    if start != -1 and end != -1 and end > start:
        try: return json.loads(text_no_think[start:end+1]), None
        except Exception as e: return None, str(e)

    return None, "No valid JSON found in completion"


def format_reward(completions: List[str], **kwargs) -> List[float]:
    rewards = []
    for completion in completions:
        if "```json" in completion:
            rewards.append(0)
        else:
            rewards.append(0)
    return rewards

def json_reward(completions: List[str], **kwargs) -> List[float]:
    rewards = []
    for completion in completions:
        data, _ = parse_diff(completion)
        rewards.append(1.0 if data is not None else -1.0)
    return rewards

def structure_reward(completions: List[str], **kwargs) -> List[float]:
    allowed_keys = {'files', 'read_files', 'write_files', 'delete_files',
                     'keys', 'read_keys', 'write_keys', 'delete_keys',
                     'executed_commands', 'mutexes', 'created_services', 'started_services'}
    rewards = []
    for completion in completions:
        data, _ = parse_diff(completion)
        if data is None:
            rewards.append(-1.0)
            continue
        try:
            if not isinstance(data, dict) or not data:
                rewards.append(-0.75)
                continue
            
            root_keys = set(data.keys())
            if not root_keys.issubset({"add", "edit"}):
                if root_keys.issubset(allowed_keys) and all(isinstance(v, list) for v in data.values()):
                    rewards.append(0.5)
                    continue
                else:
                    rewards.append(-0.5)
                    continue
            
            valid = True
            
            if "add" in data:
                add_data = data["add"]
                if not isinstance(add_data, dict) or not set(add_data.keys()).issubset(allowed_keys):
                    valid = False
                elif not all(isinstance(v, list) and all(isinstance(item, str) for item in v) for v in add_data.values()):
                    valid = False
            
            if valid and "edit" in data:
                edit_data = data["edit"]
                if not isinstance(edit_data, dict) or not set(edit_data.keys()).issubset(allowed_keys):
                    valid = False
                else:
                    for v in edit_data.values():
                        if not isinstance(v, list):
                            valid = False
                            break
                        for item in v:
                            if not isinstance(item, dict) or "old" not in item or "new" not in item:
                                valid = False
                                break
                                
            if valid:
                rewards.append(1.0)
            else:
                rewards.append(-0.5)
        except Exception:
            rewards.append(-1.0)
    return rewards

def reasoning_reward(completions: List[str], **kwargs) -> List[float]:
    rewards = []
    for completion in completions:
        match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
        if match and len(match.group(1).strip()) >= 20:
            rewards.append(1.0)
        else:
            rewards.append(-0.5)
    return rewards

def evasion_reward(completions: List[str], prompts: List[str], classifier: MLPClassifier,
                   device: str, embedder: ReportEmbedder, **kwargs) -> List[float]:
    rewards = []
    goodware_probs = []

    with torch.no_grad():
        for completion, prompt in zip(completions, prompts):
            try:
                reports = re.findall(r'(\{.+\})', prompt)
                if not reports:
                    rewards.append(0.0)
                    continue

                original_report = json.loads(reports[-1])
                diff, _ = parse_diff(completion)
                if diff is None:
                    rewards.append(0.0)
                    continue

                if isinstance(diff, dict) and 'summary' in diff and isinstance(diff['summary'], dict):
                    diff = diff['summary']

                target_data = original_report.get('summary', original_report)
                modified_data = augmented_dataset.apply_perturbation_diff(target_data, diff)
                embedding_tensor = embedder.embed(modified_data).unsqueeze(0).to(device)
                logits = classifier(embedding_tensor)
                probs = torch.softmax(logits, dim=1)
                goodware_prob = probs[0, CLASSES.index('goodware')].item()
                goodware_probs.append(goodware_prob)

                if goodware_prob > 0.5:
                    rewards.append(2.0 + goodware_prob)
                elif goodware_prob > 0.1:
                    rewards.append(goodware_prob * 2)
                else:
                    rewards.append(goodware_prob - 0.5)
            except Exception:
                rewards.append(0.0)
    return rewards, goodware_probs

def combined_reward(prompts: List[str], completions: List[str], classifier: MLPClassifier,
                    device: str, embedder: ReportEmbedder, **kwargs) -> List[float]:
    json_rewards = json_reward(completions)
    struct_rewards = structure_reward(completions)
    think_rewards = reasoning_reward(completions)
    format_rewards = format_reward(completions)
    evasion_rewards, goodware_probs = evasion_reward(
        completions, prompts, classifier, device, embedder
    )

    weights = {
        'format': 1.5,
        'json': 1.0,
        'structure': 1.0,
        'thinking': 0.5,
        'evasion': 4.0,
    }

    combined_rewards = []
    for f, j, s, t, e in zip(format_rewards, json_rewards, struct_rewards, think_rewards, evasion_rewards):
        reward = (weights['format'] * f +
                  weights['json'] * j +
                  weights['structure'] * s +
                  weights['thinking'] * t +
                  weights['evasion'] * e)
        combined_rewards.append(reward)

    n = len(completions)
    bypass_count = sum(1 for p in goodware_probs if p > 0.5)
    invalid_json_count = sum(1 for j in json_rewards if j < 0)

    _rolling_goodware_probs.extend(goodware_probs)
    rolling_bypass = sum(1 for p in _rolling_goodware_probs if p > 0.5) / max(len(_rolling_goodware_probs), 1)

    wandb_metrics = {
        "reward/combined_mean": np.mean(combined_rewards),
        "reward/format_mean": np.mean(format_rewards),
        "reward/json_validity_mean": np.mean(json_rewards),
        "reward/structure_mean": np.mean(struct_rewards),
        "reward/thinking_mean": np.mean(think_rewards),
        "reward/evasion_mean": np.mean(evasion_rewards),
        "classifier/bypass_rate": bypass_count / n if n > 0 else 0.0,
        "classifier/bypass_rate_rolling30": rolling_bypass,
        "classifier/mean_goodware_prob": np.mean(goodware_probs) if goodware_probs else 0.0,
        "quality/invalid_json_rate": invalid_json_count / n if n > 0 else 0.0,
    }
    wandb.log(wandb_metrics)


    return combined_rewards

def load_samples(split_file):
    with open(split_file, 'r') as f:
        splits = json.load(f)

    adv_paths = splits.get("adv_gen", [])
    samples = []
    for path in adv_paths:
        report = load_json_report(path)
        if report:
            try:
                data = json.loads(report)
            except Exception:
                try:
                    data = ast.literal_eval(report)
                except Exception:
                    continue
            samples.append({
                    'report': data,
                    'family': 'malware',
                    'filename': os.path.basename(path),
                    'path': path
                })
    return samples


def get_initial_score(report_data, classifier, device, embedder):
    
    try:
        embedding_tensor = embedder.embed(report_data).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = classifier(embedding_tensor)
            probs = torch.softmax(logits, dim=1)
            return probs[0, CLASSES.index('goodware')].item()
    except Exception:
        return 0.05


def build_dataset(tokenizer, split_file, classifier, device, embedder,
                  sum_path, corpus_path, reference_size=300, num_samples=400,
                  max_prompt_tokens=16000):
    samples = load_samples(split_file)
    
    print("Loading reference data (sum.json + corpus.json) for GRPO dataset generation...")
    sum_entries, corpus_terms = load_references(sum_path, corpus_path)
    importance_ranking_str = format_importance_ranking(DEFAULT_CATEGORY_IMPORTANCE)

    sample_ref_data = sample_references(sum_entries, corpus_terms, reference_size)
    sample_ref_text = format_references(sample_ref_data)
    sample_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        valid_categories=str(VALID_CATEGORIES),
        importance_ranking=importance_ranking_str,
        reference_data=sample_ref_text,
    )

    print("Computing initial classifier scores for all samples...")
    filtered_samples = []
    for sample in samples:
        report_data = sample['report'].get('summary', sample['report'])
        initial_score = get_initial_score(report_data, classifier, device, embedder)
        
        user_msg = build_message(report_data, initial_score, {}, initial_score, step=0)
        
        messages = ([{"role": "system", "content": sample_system_prompt}] + 
                    FEW_SHOT_EXAMPLES + 
                    [{"role": "user", "content": user_msg}])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        if len(tokenizer.encode(prompt, add_special_tokens=False)) < max_prompt_tokens:
            filtered_samples.append({
                **sample,
                'initial_score': initial_score,
            })
    
    print(f"  {len(filtered_samples)} samples pass token filter (from {len(samples)} total)")

    if not filtered_samples:
        raise ValueError("No reports fit within the prompt-token limit")

    prompts = []
    for _ in range(num_samples):
        sample = random.choice(filtered_samples)
        report_data = sample['report'].get('summary', sample['report'])
        initial_score = sample['initial_score']
        
        ref_data = sample_references(sum_entries, corpus_terms, reference_size)
        reference_text = format_references(ref_data)
        current_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            valid_categories=str(VALID_CATEGORIES),
            importance_ranking=importance_ranking_str,
            reference_data=reference_text,
        )
        
        user_msg = build_message(report_data, initial_score, {}, initial_score, step=0)
        
        messages = ([{"role": "system", "content": current_system_prompt}] + 
                    FEW_SHOT_EXAMPLES + 
                    [{"role": "user", "content": user_msg}])
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    return Dataset.from_dict({"prompt": prompts})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-device", default="cuda:1")
    parser.add_argument("--sum", default="sum.json")
    parser.add_argument("--corpus", default="corpus.json")
    parser.add_argument("--reference-size", type=int, default=300)
    parser.add_argument("--max-prompt-tokens", type=int, default=16000)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()

    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt == "latest":
        ckpt_dirs = sorted(glob.glob(os.path.join(args.output, "checkpoint-*")),
                           key=lambda x: int(x.split("-")[-1]))
        if ckpt_dirs:
            resume_ckpt = ckpt_dirs[-1]
            print(f"Resuming from latest checkpoint: {resume_ckpt}")
        else:
            resume_ckpt = None
            print("No checkpoints found, starting from scratch.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        max_lora_rank=16,
        fast_inference=True, 
        gpu_memory_utilization=0.5,  
        vllm_enable_prefix_caching=True,
        vllm_enforce_eager=False,
        vllm_enable_chunked_prefill=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    device = torch.device(args.device)
    classifier = MLPClassifier(1024).to(device)
    classifier.load_state_dict(torch.load(args.classifier, map_location=device))
    classifier.eval()
    embedder = ReportEmbedder(args.embedding_model, args.embedding_device)

    wandb.init(project="grpo-adversarial-multiturn", config=vars(args))

    dataset = build_dataset(
        tokenizer, args.splits, classifier, args.device, embedder,
        args.sum, args.corpus,
        args.reference_size, num_samples=args.samples,
        max_prompt_tokens=args.max_prompt_tokens
    )

    def reward_fn(prompts, completions, **kwargs):
        return combined_reward(
            prompts, completions, classifier, args.device, embedder, **kwargs
        )

    grpo_config = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_steps=args.save_steps,
        optim="adamw_8bit",
        max_completion_length=16000,
        max_prompt_length=16000,
        temperature=0.9,
        report_to="wandb",
        max_grad_norm=1.0, 
        beta=0.01,
        use_vllm=True,
        num_generations=4, 
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer, 
    )

    torch.cuda.empty_cache()
    print("Starting Multi-Turn GRPO Diff training...")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    print("Saving final model...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    wandb.finish()

if __name__ == "__main__":
    main()
