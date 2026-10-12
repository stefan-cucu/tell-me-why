import os

import json
import json5
import random
import argparse
import ast
import re
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

import augmented_dataset
from common import MLPClassifier, ReportEmbedder, load_json_report
from create_data import load_references, sample_references, format_references, VALID_CATEGORIES, DEFAULT_CATEGORY_IMPORTANCE

CLASSES = ['goodware', 'malware']

SYSTEM_PROMPT = """You are a cybersecurity expert specializing in malware evasion analysis.
Your task is to generate an adversarial perturbation (a "diff" in JSON format) that, when applied to a malware behavioral log, confuses a classifier into thinking it is "goodware". 

You have TWO types of operations available:
1. **add**: Insert new entries into categories. These should be characteristic of legitimate software behavior.
2. **edit**: Replace existing suspicious/malicious entries with benign-looking alternatives that a goodware program would produce. 
You are not allowed to edit api calls (resolved_apis), but you CAN add new api entries.

CRITICAL RULES:
1. You must first think about your strategy inside <think> tags.
2. After thinking, your final output must be ONLY the JSON diff enclosed in ```json blocks.
3. Do not output any conversational text, introductions, or overviews.
4. Use this exact format:
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
5. Valid categories: """ + str(VALID_CATEGORIES) + """
6. Each "old" value in edits MUST exactly match an entry in the original log.
7. Replacement entries should look like legitimate software behavior.
8. Added entries must be realistic and not duplicate existing entries.
9. Do not remove entries — only add or replace.
10. Make sure to make atleast 100 additions and edits to each log file.
11. Focus your edits on the most impactful categories (see importance ranking below).

## Category Importance (from classifier ablation analysis)
The classifier relies on these categories most heavily. Focus your perturbations on the top-ranked ones:
{importance_ranking}

## Reference: Common Goodware Behavioral Entries
{reference_data}
"""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": '{"files": ["C:\\\\Users\\\\John\\\\AppData\\\\Local\\\\Temp\\\\malware.exe"], "read_files": ["C:\\\\Users\\\\John\\\\AppData\\\\Local\\\\Temp\\\\malware.exe"], "write_files": ["C:\\\\Windows\\\\System32\\\\config.bin"], "keys":["HKEY_LOCAL_MACHINE\\\\Software\\\\Malware"]}'
    },
    {
        "role": "assistant",
        "content": """<think>
The input shows basic malware behavior executing from a Temp directory and modifying a system registry key.
To mask this, I will add behaviors commonly seen in legitimate Windows GUI applications, such as loading fonts and checking UI themes.
I also need to hide the blatant malicious registry key by editing it to look like a benign system configuration read.
</think>
```json
{
  "add": {
    "read_files": [
      "C:\\\\Windows\\\\Fonts\\\\staticcache.dat",
      "C:\\\\Windows\\\\System32\\\\uxtheme.dll"
    ],
    "read_keys": [
      "HKEY_LOCAL_MACHINE\\\\Software\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Policies\\\\System"
    ]
  },
  "edit": {
    "keys": [
      {
        "old": "HKEY_LOCAL_MACHINE\\\\Software\\\\Malware",
        "new": "HKEY_LOCAL_MACHINE\\\\System\\\\CurrentControlSet\\\\Control\\\\Nls\\\\Locale"
      }
    ]
  }
}
```"""
    }
]

def normalize_diff(data):
    
    if isinstance(data, dict):
        return {k: normalize_diff(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [normalize_diff(i) for i in data]
    elif isinstance(data, str):
        return data.replace('\\\\', '\\')
    return data

def parse_diff(text):
    text_no_think = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    match = re.search(r'```json\s*(.*?)\s*```', text_no_think, re.DOTALL)
    if match:
        try:
            return normalize_diff(json5.loads(match.group(1))), None
        except Exception:
            pass

    start = text_no_think.find('{')
    end = text_no_think.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return normalize_diff(json5.loads(text_no_think[start:end+1])), None
        except Exception as e:
            return None, str(e)

    return None, "No valid JSON found in completion"


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


def evaluate_candidates(original_report_data, completions, classifier, device, embedder, verbose=False):
    
    best_prob = 0.0
    with torch.no_grad():
        for i, completion in enumerate(completions):
            diff, parse_err = parse_diff(completion)

            if diff is None:
                if verbose:
                    print(f"  [attempt {i}] parse failed: {parse_err}")
                    print(f"  [attempt {i}] raw completion snippet: {completion[:200]}")
                continue

            if isinstance(diff, dict) and 'summary' in diff and isinstance(diff['summary'], dict):
                diff = diff['summary']

            try:
                modified_data = augmented_dataset.apply_perturbation_diff(original_report_data, diff)
                embedding_tensor = embedder.embed(modified_data).unsqueeze(0).to(device)
                logits = classifier(embedding_tensor)
                probs = torch.softmax(logits, dim=1)
                goodware_prob = probs[0, CLASSES.index('goodware')].item()
                best_prob = max(best_prob, goodware_prob)

                if verbose:
                    print(f"  [attempt {i}] goodware_prob={goodware_prob:.4f}")

                if goodware_prob > 0.5:
                    return True, modified_data, best_prob, completion

            except Exception as e:
                if verbose:
                    print(f"  [attempt {i}] exception in perturbation/embedding: {e}")
                continue

    return False, None, best_prob, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sum", default="sum.json")
    parser.add_argument("--corpus", default="corpus.json")
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    print("Loading malware validation splits...")
    samples = load_samples(args.splits)
    if not samples:
        print("No malware data found in split file!")
        return

    reports = random.sample(samples, min(args.samples, len(samples)))
    print(f"Selected {len(reports)} samples for evaluation.")

    print("Loading classifier...")
    classifier = MLPClassifier(1024).to(device)
    if os.path.exists(args.classifier):
        classifier.load_state_dict(torch.load(args.classifier, map_location=device))
        classifier.eval()
    else:
        print(f"ERROR: Classifier checkpoint {args.classifier} not found.")
        return
    embedder = ReportEmbedder(args.embedding_model, args.embedding_device)

    use_lora = bool(args.adapter and args.adapter.lower() != "none")
    print(f"Loading vLLM model from {args.model} (LoRA: {args.adapter if use_lora else 'disabled'})...")

    try:
        llm = LLM(
            model=args.model,
            tensor_parallel_size=1,
            trust_remote_code=True,
            enable_prefix_caching=True,
            max_model_len=48000,
            enable_lora=use_lora
        )
    except Exception as e:
        print(f"Failed to load with prefix caching. Trying defaults. Error: {e}")
        llm = LLM(
            model=args.model,
            tensor_parallel_size=1,
            trust_remote_code=True,
            enable_lora=use_lora
        )

    tokenizer = llm.get_tokenizer()
    prompts = []
    original_reports_data = []
    eval_samples_kept = []

    print("Loading reference data (sum.json + corpus.json) for prompt context...")
    sum_entries, corpus_terms = load_references(args.sum, args.corpus)

    
    sorted_cats = sorted(DEFAULT_CATEGORY_IMPORTANCE.items(), key=lambda x: x[1], reverse=True)
    importance_lines = []
    for rank, (cat, score) in enumerate(sorted_cats, 1):
        importance_lines.append(f"  {rank}. {cat} (importance: {score:.4f})")
    importance_ranking_str = "\n".join(importance_lines)

    for sample in reports:
        report_data = sample['report'].get('summary', sample['report'])
        input_json = json.dumps(report_data)

        
        ref_data = sample_references(sum_entries, corpus_terms, 300)
        reference_text = format_references(ref_data)
        current_system_prompt = SYSTEM_PROMPT.replace("{reference_data}", reference_text).replace("{importance_ranking}", importance_ranking_str)

        messages = [
            {"role": "system", "content": current_system_prompt}
        ] + FEW_SHOT_EXAMPLES + [
            {"role": "user", "content": input_json}
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
        if token_count > 32000:
            print(f"Skipping {sample['filename']} ({token_count} tokens > 32k)")
            continue

        original_reports_data.append(report_data)
        prompts.append(prompt)
        eval_samples_kept.append(sample)

    print(f"Prepared {len(prompts)} prompts after token filtering.")

    
    
    sampling_params = SamplingParams(
        n=args.attempts,
        temperature=0.9,
        max_tokens=16000,
        stop=["<|end_of_text|>", "<|im_end|>"]
    )

    from vllm.lora.request import LoRARequest
    lora_request = LoRARequest("lora_adapter", 1, args.adapter) if use_lora else None

    if use_lora:
        print(f"Generating with LoRA: {args.adapter}...")
    else:
        print("Generating with merged model (no LoRA)...")

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    print("Evaluating outputs...")
    effective_successes = 0
    valid_format_count = 0
    total_evaluated = 0
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving logs to: {output_dir}")

    pbar = tqdm(total=len(outputs), desc="Evaluating")
    for i, output in enumerate(outputs):
        completions = [o.text for o in output.outputs]
        original_data = original_reports_data[i]
        sample = eval_samples_kept[i]

        if args.verbose:
            print(f"\n[Sample {i}] {sample['filename']}")

        has_valid_json = any(parse_diff(c)[0] is not None for c in completions)
        if has_valid_json:
            valid_format_count += 1

        is_evaded, evaded_data, best_prob, winning_completion = evaluate_candidates(
            original_data, completions, classifier, device, embedder,
            verbose=args.verbose
        )

        if is_evaded:
            effective_successes += 1

        
        winning_thinking = None
        if winning_completion:
            think_match = re.search(r'<think>(.*?)</think>', winning_completion, re.DOTALL)
            if think_match:
                winning_thinking = think_match.group(1).strip()

        
        
        
        prefix = "evaded" if is_evaded else "failed"
        save_path = os.path.join(output_dir, f"{prefix}_{sample['filename']}")
        try:
            with open(save_path, "w") as f:
                json.dump({
                    "evaded": is_evaded,
                    "best_goodware_prob": best_prob,
                    "original": original_data,
                    "evaded_data": evaded_data,
                    "winning_thinking": winning_thinking,
                    "winning_completion": winning_completion,
                    "completions": completions,
                }, f, indent=2, default=str)
        except Exception as e:
            print(f"\n[ERROR] Could not write log for {sample['filename']}: {e}")

        total_evaluated += 1
        pbar.update(1)

        ev_rate = (effective_successes / total_evaluated) * 100
        syn_rate = (valid_format_count / total_evaluated) * 100
        pbar.set_postfix({
            "Ev Rate": f"{ev_rate:.1f}%",
            "Syn Rate": f"{syn_rate:.1f}%",
            "Best P(gw)": f"{best_prob:.3f}",
        })

        if total_evaluated % args.log_every == 0:
            print(f"\n[Log at {total_evaluated}] Evaded: {effective_successes}/{total_evaluated} "
                  f"({ev_rate:.2f}%) | Valid Syntax: {syn_rate:.2f}%")

    pbar.close()

    print("\n" + "=" * 50)
    print("FINAL EVALUATION SUMMARY")
    print(f"Total Samples    : {total_evaluated}")
    print(f"Total Evaded     : {effective_successes}")
    print(f"Valid Syntax >=1 : {valid_format_count}")
    print(f"Evasion Rate     : {(effective_successes / total_evaluated) * 100:.2f}%")
    print(f"Syntax Rate      : {(valid_format_count / total_evaluated) * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
