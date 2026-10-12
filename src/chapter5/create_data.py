import os
import json
import random
import argparse
import time
import requests
import torch
from tqdm import tqdm
import augmented_dataset
import ast
from common import MLPClassifier, ReportEmbedder, load_json_report



CLASSES = ['goodware', 'malware']

VALID_CATEGORIES = [
    'files', 'read_files', 'write_files', 'delete_files',
    'keys', 'read_keys', 'write_keys', 'delete_keys',
    'executed_commands', 'mutexes',
    'created_services', 'started_services'
]


DEFAULT_CATEGORY_IMPORTANCE = {
    "created_services": 0.0703,
    "resolved_apis":    0.0598,
    "delete_keys":      0.0487,
    "files":            0.0265,
    "keys":             0.0263,
    "write_keys":       0.0224,
    "write_files":      0.0192,
    "read_files":       0.0181,
    "read_keys":        0.0165,
    "executed_commands": 0.0148,
    "delete_files":     0.0094,
    "mutexes":          0.0057,
}

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

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": '{"summary": {"files": ["C:\\\\Users\\\\John\\\\AppData\\\\Local\\\\Temp\\\\malware.exe", "C:\\\\Windows\\\\System32\\\\config.bin"], "read_files": ["C:\\\\Users\\\\John\\\\AppData\\\\Local\\\\Temp\\\\malware.exe"], "write_files": ["C:\\\\Windows\\\\System32\\\\config.bin"], "keys": ["HKEY_LOCAL_MACHINE\\\\Software\\\\Malware\\\\Config", "HKEY_LOCAL_MACHINE\\\\SOFTWARE\\\\Microsoft\\\\OLEAUT"], "executed_commands": ["C:\\\\Users\\\\John\\\\AppData\\\\Local\\\\Temp\\\\malware.exe", "cmd.exe /c sc stop WinDefend"]}}'
    },
    {
        "role": "assistant",
        "content": """```json
{
  "add": {
    "read_files": [
      "C:\\\\Windows\\\\Fonts\\\\staticcache.dat",
      "C:\\\\Windows\\\\System32\\\\locale.nls",
      "C:\\\\Windows\\\\System32\\\\uxtheme.dll"
    ],
    "keys": [
      "HKEY_LOCAL_MACHINE\\\\System\\\\CurrentControlSet\\\\Control\\\\Nls\\\\Locale",
      "HKEY_LOCAL_MACHINE\\\\SOFTWARE\\\\Microsoft\\\\Windows NT\\\\CurrentVersion\\\\FontLink\\\\SystemLink"
    ],
    "read_keys": [
      "HKEY_LOCAL_MACHINE\\\\Software\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Policies\\\\System",
      "DisableUserModeCallbackFilter"
    ]
  },
  "edit": {
    "keys": [
      {"old": "HKEY_LOCAL_MACHINE\\\\Software\\\\Malware\\\\Config", "new": "HKEY_LOCAL_MACHINE\\\\Software\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Policies\\\\System"}
    ],
    "executed_commands": [
      {"old": "cmd.exe /c sc stop WinDefend", "new": "C:\\\\Windows\\\\splwow64.exe 8192"}
    ]
  }
}
```"""
    }
]


def load_references(sum_path, corpus_path):
    
    sum_entries = {}
    corpus_terms = []
    
    
    if os.path.exists(sum_path):
        with open(sum_path, 'r') as f:
            sum_data = json.load(f)
        for cat in VALID_CATEGORIES:
            if cat in sum_data and sum_data[cat]:
                
                sum_entries[cat] = sum_data[cat][:50]
    
    
    if os.path.exists(corpus_path):
        with open(corpus_path, 'r') as f:
            corpus_data = json.load(f)
        all_terms = list(corpus_data.keys())
        
        corpus_terms = [t for t in all_terms if len(t) > 3 and not all(c.isalpha() and len(t) <= 5 for c in t)]
    
    print(f"  Loaded {sum(len(v) for v in sum_entries.values())} sum entries across {len(sum_entries)} categories")
    print(f"  Loaded {len(corpus_terms)} meaningful corpus terms (from {len(corpus_data) if corpus_data else 0} total)")
    return sum_entries, corpus_terms


def sample_references(sum_entries, corpus_terms, corpus_sample_size=300):
    
    reference = {"goodware_entries_by_category": sum_entries}
    if corpus_terms:
        if len(corpus_terms) > corpus_sample_size:
            reference["goodware_terms_sample"] = random.sample(corpus_terms, corpus_sample_size)
        else:
            reference["goodware_terms_sample"] = corpus_terms
    return reference


def format_references(reference):
    
    parts = []
    
    if "goodware_entries_by_category" in reference:
        parts.append("=== Common Goodware Entries (by category, ordered by frequency) ===")
        for cat, entries in reference["goodware_entries_by_category"].items():
            entries_str = json.dumps(entries[:30], indent=None)
            parts.append(f"{cat}: {entries_str}")
    
    if "goodware_terms_sample" in reference:
        parts.append("\n=== Sample Goodware Terms (for constructing entries) ===")
        parts.append(json.dumps(reference["goodware_terms_sample"][:200], indent=None))
    
    return "\n".join(parts)


def load_samples(split_file):
    with open(split_file, 'r') as f:
        splits = json.load(f)
    
    adv_paths = splits.get("adv_gen", [])
    samples = []
    print(f"Loading {len(adv_paths)} adv_gen samples from {split_file}...")
    for path in tqdm(adv_paths, desc="Loading JSONs"):
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

def query_model(messages, model_name, api_url, temperature, max_tokens):
    endpoint = f"{api_url.rstrip('/')}/api/chat"
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": 48000,
            "num_predict": max_tokens,
        }
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "ERROR: No content in Ollama output")
        thinking = data.get("message", {}).get("thinking", None)
        return content, thinking
    except Exception as e:
        return f"ERROR: {e}", None

def parse_diff(text):
    last_error = None
    
    if '<think>' in text and '</think>' in text:
        think_end = text.find('</think>') + len('</think>')
        text = text[think_end:].strip()

    if '```json' in text:
        try:
            json_block = text.split('```json')[-1].split('```')[0].strip()
            return json.loads(json_block), None
        except Exception as e:
            last_error = str(e)

    brace_positions = [i for i, c in enumerate(text) if c == '{']
    for pos in reversed(brace_positions):
        candidate = text[pos:]
        try:
            return json.loads(candidate), None
        except Exception as e:
            last_error = str(e)
            continue

    return None, last_error or "No valid JSON found in completion"

def get_reasoning(raw_text):
    if '<think>' in raw_text and '</think>' in raw_text:
        start = raw_text.find('<think>') + len('<think>')
        end = raw_text.find('</think>')
        return raw_text[start:end].strip()
    if '<|channel|>thought' in raw_text and '<channel|>' in raw_text:
        start = raw_text.find('<|channel|>thought') + len('<|channel|>thought')
        end = raw_text.find('<channel|>')
        return raw_text[start:end].strip()
    return None

def predict_label(report_text, embedder, classifier):
    embedding = embedder.embed(report_text)
    device = next(classifier.parameters()).device
    with torch.no_grad():
        embedding_tensor = embedding.unsqueeze(0).to(device)
        logits = classifier(embedding_tensor)
        probs = torch.softmax(logits, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
        return CLASSES[predicted_idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--model", default="qwen3.5:35b")
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failed-output", required=True)
    parser.add_argument("--sum", default="sum.json")
    parser.add_argument("--corpus", default="corpus.json")
    parser.add_argument("--importance")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-device", default="cuda:0")
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--reference-size", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    
    cat_importance = DEFAULT_CATEGORY_IMPORTANCE
    if args.importance and os.path.exists(args.importance):
        with open(args.importance, 'r') as f:
                cat_importance = json.load(f)
        print(f"Loaded category importance from {args.importance}")
    else:
        print("Using default category importance (hardcoded from ablation run)")

    
    sorted_cats = sorted(cat_importance.items(), key=lambda x: x[1], reverse=True)
    importance_lines = []
    for rank, (cat, score) in enumerate(sorted_cats, 1):
        importance_lines.append(f"  {rank}. {cat} (importance: {score:.4f})")
    importance_ranking_str = "\n".join(importance_lines)

    print("Loading reference data (sum.json + corpus.json)...")
    sum_entries, corpus_terms = load_references(args.sum, args.corpus)

    print("Loading classifier...")
    device = torch.device(args.device)
    classifier = MLPClassifier(1024).to(device)
    if not os.path.exists(args.classifier):
        raise FileNotFoundError(args.classifier)
    classifier.load_state_dict(torch.load(args.classifier, map_location=device))
    classifier.eval()
    embedder = ReportEmbedder(args.embedding_model, args.embedding_device)

    train_samples = load_samples(args.splits)
    print(f"Loaded {len(train_samples)} valid samples.")
    if not train_samples:
        print("Error: No samples found. Is training done?")
        return

    existing_count = 0
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            for line in f:
                if line.strip():
                    existing_count += 1
    print(f"Existing successful bypasses in {args.output}: {existing_count}")

    if existing_count >= args.target:
        print("Target already reached. Exiting.")
        return

    pbar = tqdm(total=args.target, initial=existing_count, desc="Collecting adversarial logs")
    MAX_PROMPT_CHARS = 30000 * 3.5 

    with open(args.output, 'a') as out_f, open(args.failed_output, 'a') as fail_f:
        while existing_count < args.target:
            sample = random.choice(train_samples)
            report = sample['report']
            report_json = json.dumps(report)
            
            if len(report_json) > MAX_PROMPT_CHARS:
                print(f"\n[SKIP] Sample {sample['filename']} too large ({len(report_json)} chars).")
                continue

            print(f"\n[PICK] Sample: {sample['filename']} - {len(report_json)} chars")

            
            ref_data = sample_references(sum_entries, corpus_terms, args.reference_size)
            reference_text = format_references(ref_data)
            current_system_prompt = SYSTEM_PROMPT.replace("{reference_data}", reference_text).replace("{importance_ranking}", importance_ranking_str)

            success = False
            for attempt in range(1, 4):
                print(f"  [ATTEMPT {attempt}/3] Querying LLM...")
                
                messages = [{"role": "system", "content": current_system_prompt}]
                messages += FEW_SHOT_EXAMPLES
                messages.append({"role": "user", "content": report_json})

                start_time = time.time()
                raw_completion, thinking = query_model(
                    messages, args.model, args.api_url, args.temperature, args.max_tokens
                )
                duration = time.time() - start_time

                if raw_completion.startswith("ERROR:"):
                    print(f"    [FAIL] Ollama error: {raw_completion}")
                    time.sleep(2)
                    continue

                print(f"    [INFO] LLM responded in {duration:.1f}s. Parsing...")

                parsed_diff, parse_error = parse_diff(raw_completion)
                
                if parsed_diff is None:
                    print(f"    [FAIL] Parse error: {parse_error}")
                    fail_f.write(f"[FAIL]{raw_completion.replace('\n', '')}\n")
                    continue

                try:
                    diff_to_apply = parsed_diff
                    
                    if isinstance(diff_to_apply, dict) and 'summary' in diff_to_apply:
                        diff_to_apply = diff_to_apply['summary']

                    original_data = report.get('summary', report)
                    modified_data = augmented_dataset.apply_perturbation_diff(original_data, diff_to_apply)
                    
                    
                    add_count = sum(len(v) for v in diff_to_apply.get("add", {}).values())
                    edit_count = sum(len(v) for v in diff_to_apply.get("edit", {}).values())
                    print(f"    [INFO] Diff: {add_count} additions, {edit_count} edits")

                    predicted_class = predict_label(modified_data, embedder, classifier)
                    
                    final_thinking = thinking or get_reasoning(raw_completion)
                    log_entry = {
                        "full_input": report,
                        "thinking": final_thinking,
                        "raw_completion": raw_completion,
                        "parsed_diff": parsed_diff,
                        "original_family": sample['family'],
                        "filename": sample['filename'],
                        "bypassed": predicted_class == 'goodware'
                    }

                    if predicted_class == 'goodware':
                        print(f"    [SUCCESS] Bypass achieved!")
                        out_f.write(json.dumps(log_entry) + "\n")
                        out_f.flush()
                        
                        existing_count += 1
                        pbar.update(1)
                        success = True
                        break
                    elif predicted_class == 'malware':
                        print(f"    [FAIL] Classifier still detected malware. Logging to failed file.")
                        fail_f.write(json.dumps(log_entry) + "\n")
                        fail_f.flush()
                    else:
                        print("    [FAIL] Embedding generation for classifier failed.")
                except Exception as e:
                    print(f"    [ERROR] Processing error: {e}")
                    pass
            
            if not success:
                print(f"  [GIVE UP] Failed to generate adversarial for {sample['filename']}.")

    pbar.close()
    print(f"Finished. Collected {existing_count} bypass samples in {args.output}")

if __name__ == "__main__":
    main()
