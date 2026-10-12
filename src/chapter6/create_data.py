import os
import copy
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

SYSTEM_PROMPT_TEMPLATE = """You are a cybersecurity expert specializing in malware evasion analysis.
Your task is to generate an adversarial perturbation (a "diff" in JSON format) that, when applied to a malware behavioral log, makes a classifier think it is "goodware".

You have TWO types of operations available:
1. **add**: Insert new entries into categories. These should be characteristic of legitimate software behavior.
2. **edit**: Replace existing suspicious/malicious entries with benign-looking alternatives that a goodware program would produce.
You are not allowed to edit api calls (resolved_apis), but you CAN add new api entries.

Rules:
1. Output only the JSON diff and nothing else.
2. Use this exact format:
{{
  "add": {{
    "category": ["new_entry_1", "new_entry_2"]
  }},
  "edit": {{
    "category": [
      {{"old": "exact_existing_entry", "new": "replacement_entry"}}
    ]
  }}
}}
3. Valid categories: {valid_categories}
4. Each "old" value in edits MUST exactly match an entry in the current state of the log.
5. Replacement entries should look like legitimate software behavior.
6. Added entries must be realistic and not duplicate existing entries.
7. Do not remove entries — only add or replace.
8. Make sure to make between 10 and 100 additions and edits to each log file.
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
        "content": """## Initial Input
{"files": ["C:\\\\Users\\\\John\\\\AppData\\\\Local\\\\Temp\\\\malware.exe", "C:\\\\Windows\\\\System32\\\\config.bin"], "read_files": ["C:\\\\Users\\\\John\\\\AppData\\\\Local\\\\Temp\\\\malware.exe"], "write_files": ["C:\\\\Windows\\\\System32\\\\config.bin"], "keys": ["HKEY_LOCAL_MACHINE\\\\Software\\\\Malware\\\\Config", "HKEY_LOCAL_MACHINE\\\\SOFTWARE\\\\Microsoft\\\\OLEAUT"], "executed_commands": ["C:\\\\Users\\\\John\\\\AppData\\\\Local\\\\Temp\\\\malware.exe", "cmd.exe /c sc stop WinDefend"]}

## Initial Classifier Score: P(goodware) = 0.0312
## Working Diff: {}
## Current Classifier Score: n/a"""
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
    ],
    "resolved_apis": [
      "kernel32.GetModuleHandleW",
      "advapi32.RegQueryValueExW",
      "user32.GetSystemMetrics"
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


def format_importance_ranking(importance_dict):
    sorted_cats = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    lines = []
    for rank, (cat, score) in enumerate(sorted_cats, 1):
        note = ""
        if cat == "resolved_apis":
            note = " (add only, no edits)"
        lines.append(f"  {rank}. {cat}: importance={score:.4f}{note}")
    return "\n".join(lines)


def load_importance_data(importance_file):
    if not importance_file or not os.path.exists(importance_file):
        print(f"  Importance file '{importance_file}' not found, using defaults.")
        return DEFAULT_CATEGORY_IMPORTANCE
    with open(importance_file, 'r') as f:
        data = json.load(f)
    importance = {}
    cat_data = data.get("category_importance", {})
    for cat, stats in cat_data.items():
        importance[cat] = stats.get("mean_abs", 0.0)
    print(f"  Loaded importance for {len(importance)} categories from '{importance_file}'.")
    return importance


def load_references(sum_path, corpus_path):
    sum_entries = {}
    corpus_terms = []

    if os.path.exists(sum_path):
        with open(sum_path, 'r') as f:
            sum_data = json.load(f)
        for cat in VALID_CATEGORIES:
            if cat in sum_data and sum_data[cat]:
                sum_entries[cat] = sum_data[cat][:50]

    corpus_data = {}
    if os.path.exists(corpus_path):
        with open(corpus_path, 'r') as f:
            corpus_data = json.load(f)
        all_terms = list(corpus_data.keys())
        corpus_terms = [t for t in all_terms if len(t) > 3 and not all(c.isalpha() and len(t) <= 5 for c in t)]

    print(f"  Loaded {sum(len(v) for v in sum_entries.values())} sum entries across {len(sum_entries)} categories")
    print(f"  Loaded {len(corpus_terms)} meaningful corpus terms (from {len(corpus_data)} total)")
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
        resp = requests.post(endpoint, json=payload, timeout=60*5)
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




def score_report(report, embedder, classifier):
    embedding = embedder.embed(report)
    with torch.no_grad():
        device = next(classifier.parameters()).device
        tensor = embedding.unsqueeze(0).to(device)
        logits = classifier(tensor)
        probs = torch.softmax(logits, dim=1)
        return probs[0, 0].item()




def merge_diff(accumulated, new_diff):
    
    merged = copy.deepcopy(accumulated) if accumulated else {}


    new_adds = new_diff.get("add", {})
    if new_adds:
        if "add" not in merged:
            merged["add"] = {}
        for cat, items in new_adds.items():
            if cat not in merged["add"]:
                merged["add"][cat] = []
            if isinstance(items, list):
                merged["add"][cat].extend(items)
            else:
                merged["add"][cat].append(items)


    new_edits = new_diff.get("edit", {})
    if new_edits:
        if "edit" not in merged:
            merged["edit"] = {}
        for cat, replacements in new_edits.items():
            if cat not in merged["edit"]:
                merged["edit"][cat] = []
            if isinstance(replacements, list):
                merged["edit"][cat].extend(replacements)

    return merged


def count_changes(diff):
    
    add_count = sum(len(v) if isinstance(v, list) else 1 for v in diff.get("add", {}).values())
    edit_count = sum(len(v) if isinstance(v, list) else 1 for v in diff.get("edit", {}).values())
    return add_count, edit_count




def build_message(original_report, initial_score, accumulated_diff, current_score, step):
    
    report_json = json.dumps(original_report, indent=None)


    if accumulated_diff and (accumulated_diff.get("add") or accumulated_diff.get("edit")):
        acc_str = json.dumps(accumulated_diff, indent=None)
        
        if len(acc_str) > 5000:
            a, e = count_changes(accumulated_diff)
            acc_str = f"[{a} additions, {e} edits across categories — too large to display. Focus on NEW changes.]"
    else:
        acc_str = "{}"

    if step == 0:
        current_score_str = "n/a"
        guidance = "Generate a JSON diff to push P(goodware) above 0.50."
    else:
        current_score_str = f"P(goodware) = {current_score:.4f}"
        delta = current_score - initial_score
        a, e = count_changes(accumulated_diff)
        guidance = (
            f"The previous {step} step(s) moved the score from {initial_score:.4f} → {current_score:.4f} "
            f"(delta: {delta:+.4f}) with {a} additions and {e} edits so far, but did NOT evade.\n"
            f"Generate a NEW diff (applied on top of the current state) to push P(goodware) above 0.50.\n"
            f"Focus on categories you haven't modified much yet, or try different strategies."
        )

    parts = [
        f"## Initial Input\n{report_json}",
        f"\n## Initial Classifier Score: P(goodware) = {initial_score:.4f}",
        f"## Working Diff (accumulated changes so far): {acc_str}",
        f"## Current Classifier Score: {current_score_str}",
        f"\n{guidance}"
    ]
    return "\n".join(parts)




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
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--reference-size", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()


    print("Loading category importance data...")
    importance = load_importance_data(args.importance)

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
        print("Error: No samples found.")
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


    importance_ranking_str = format_importance_ranking(importance)
    ref_data = sample_references(sum_entries, corpus_terms, args.reference_size)
    reference_text = format_references(ref_data)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        valid_categories=str(VALID_CATEGORIES),
        importance_ranking=importance_ranking_str,
        reference_data=reference_text
    )

    MAX_PROMPT_CHARS = 30000 * 3.5


    total_attempts = 0
    step_evasion_counts = [0] * args.max_steps
    score_progressions = []

    pbar = tqdm(total=args.target, initial=existing_count, desc="Collecting adversarial logs")

    with open(args.output, 'a') as out_f, open(args.failed_output, 'a') as fail_f:
        while existing_count < args.target:
            sample = random.choice(train_samples)
            report = sample['report']
            original_data = report.get('summary', report)
            report_json = json.dumps(original_data)

            if len(report_json) > MAX_PROMPT_CHARS:
                print(f"\n[SKIP] Sample {sample['filename']} too large ({len(report_json)} chars).")
                continue

            total_attempts += 1


            initial_score = score_report(original_data, embedder, classifier)
            if initial_score is None:
                print(f"\n[SKIP] Embedding failed for {sample['filename']}.")
                continue

            if initial_score > 0.5:
                print(f"\n[SKIP] {sample['filename']} already classified as goodware (P={initial_score:.4f}).")
                continue

            print(f"\n[PICK] Sample: {sample['filename']} — P(goodware)={initial_score:.4f}")


            accumulated_diff = {}
            current_report = copy.deepcopy(original_data)
            current_score = initial_score
            success = False
            step_scores = [initial_score]
            all_raw_completions = []
            all_thinkings = []

            for step in range(args.max_steps):
                print(f"  [STEP {step+1}/{args.max_steps}] P(gw)={current_score:.4f} — Querying LLM...")

                user_message = build_message(
                    original_data, initial_score,
                    accumulated_diff, current_score, step
                )

                messages = [{"role": "system", "content": system_prompt}]
                messages += FEW_SHOT_EXAMPLES
                messages.append({"role": "user", "content": user_message})

                start_time = time.time()
                raw_completion, thinking = query_model(
                    messages, args.model, args.api_url, args.temperature, args.max_tokens
                )
                duration = time.time() - start_time

                all_raw_completions.append(raw_completion)
                all_thinkings.append(thinking or get_reasoning(raw_completion))

                if raw_completion.startswith("ERROR:"):
                    print(f"    [FAIL] Ollama error: {raw_completion}. Aborting sample.")
                    break

                print(f"    [INFO] LLM responded in {duration:.1f}s. Parsing...")

                parsed_diff, parse_error = parse_diff(raw_completion)
                if parsed_diff is None:
                    print(f"    [FAIL] Parse error: {parse_error}")
                    continue

                try:
                    diff_to_apply = parsed_diff
                    if isinstance(diff_to_apply, dict) and 'summary' in diff_to_apply:
                        diff_to_apply = diff_to_apply['summary']


                    saved_report = copy.deepcopy(current_report)
                    saved_diff = copy.deepcopy(accumulated_diff)


                    current_report = augmented_dataset.apply_perturbation_diff(current_report, diff_to_apply)
                    accumulated_diff = merge_diff(accumulated_diff, diff_to_apply)

                    a, e = count_changes(diff_to_apply)
                    total_a, total_e = count_changes(accumulated_diff)
                    print(f"    [INFO] This step: +{a} adds, +{e} edits | Total: {total_a} adds, {total_e} edits")


                    new_score = score_report(current_report, embedder, classifier)
                    if new_score is None:
                        print(f"    [FAIL] Embedding failed after applying diff.")
                        current_report = saved_report
                        accumulated_diff = saved_diff
                        continue

                    delta = new_score - current_score


                    if delta < 0.2 and new_score <= 0.5:
                        print(f"    [NEG-DELTA] P(gw) went from {current_score:.4f} → {new_score:.4f} (delta: {delta:+.4f}). Reverting & retrying...")
                        current_report = saved_report
                        accumulated_diff = saved_diff


                        retry_completion, retry_thinking = query_model(
                            messages, args.model, args.api_url, args.temperature, args.max_tokens
                        )
                        all_raw_completions.append(retry_completion)
                        all_thinkings.append(retry_thinking or get_reasoning(retry_completion))

                        if retry_completion.startswith("ERROR:"):
                            print(f"    [NEG-DELTA RETRY FAIL] Ollama error. Aborting sample.")
                            break

                        retry_diff, retry_error = parse_diff(retry_completion)
                        if retry_diff is None:
                            print(f"    [NEG-DELTA RETRY FAIL] Parse error: {retry_error}. Aborting sample.")
                            break

                        retry_to_apply = retry_diff
                        if isinstance(retry_to_apply, dict) and 'summary' in retry_to_apply:
                            retry_to_apply = retry_to_apply['summary']

                        retry_report = augmented_dataset.apply_perturbation_diff(current_report, retry_to_apply)
                        retry_score = score_report(retry_report, embedder, classifier)

                        if retry_score is None:
                            print(f"    [NEG-DELTA RETRY FAIL] Embedding failed. Aborting sample.")
                            break

                        retry_delta = retry_score - current_score
                        if retry_delta < 0 and retry_score <= 0.5:
                            print(f"    [NEG-DELTA RETRY FAIL] Retry also negative ({current_score:.4f} → {retry_score:.4f}). Aborting sample.")
                            step_scores.append(retry_score)
                            current_score = retry_score
                            break
                        else:

                            print(f"    [NEG-DELTA RETRY OK] Retry positive ({current_score:.4f} → {retry_score:.4f})")
                            current_report = retry_report
                            accumulated_diff = merge_diff(accumulated_diff, retry_to_apply)
                            current_score = retry_score
                            step_scores.append(current_score)
                            delta = retry_delta

                            if current_score > 0.5:
                                step_evasion_counts[step] += 1
                                print(f"    [SUCCESS] Evasion achieved at step {step+1} (via retry)!")
                                success = True
                                log_entry = {
                                    "full_input": report,
                                    "original_data": original_data,
                                    "accumulated_diff": accumulated_diff,
                                    "initial_score": initial_score,
                                    "final_score": current_score,
                                    "score_progression": step_scores,
                                    "steps_taken": step + 1,
                                    "all_completions": all_raw_completions,
                                    "all_thinkings": all_thinkings,
                                    "original_family": sample['family'],
                                    "filename": sample['filename'],
                                    "bypassed": True
                                }
                                out_f.write(json.dumps(log_entry) + "\n")
                                out_f.flush()
                                existing_count += 1
                                pbar.update(1)
                                break
                            print(f"    [SCORE] P(goodware) = {current_score:.4f} (delta: {delta:+.4f})")
                            continue


                    current_score = new_score
                    step_scores.append(current_score)
                    print(f"    [SCORE] P(goodware) = {current_score:.4f} (delta: {delta:+.4f})")

                    if current_score > 0.5:
                        step_evasion_counts[step] += 1
                        print(f"    [SUCCESS] Evasion achieved at step {step+1}!")
                        success = True

                        log_entry = {
                            "full_input": report,
                            "original_data": original_data,
                            "accumulated_diff": accumulated_diff,
                            "initial_score": initial_score,
                            "final_score": current_score,
                            "score_progression": step_scores,
                            "steps_taken": step + 1,
                            "all_completions": all_raw_completions,
                            "all_thinkings": all_thinkings,
                            "original_family": sample['family'],
                            "filename": sample['filename'],
                            "bypassed": True
                        }
                        out_f.write(json.dumps(log_entry) + "\n")
                        out_f.flush()
                        existing_count += 1
                        pbar.update(1)
                        break

                except Exception as e:
                    print(f"    [ERROR] Processing error: {e}")
                    continue

            if not success:
                print(f"  [GIVE UP] {sample['filename']} — final P(gw)={current_score:.4f} after {args.max_steps} steps")
                score_progressions.append(step_scores)

                log_entry = {
                    "full_input": report,
                    "original_data": original_data,
                    "accumulated_diff": accumulated_diff,
                    "initial_score": initial_score,
                    "final_score": current_score,
                    "score_progression": step_scores,
                    "steps_taken": args.max_steps,
                    "all_completions": all_raw_completions,
                    "all_thinkings": all_thinkings,
                    "original_family": sample['family'],
                    "filename": sample['filename'],
                    "bypassed": False
                }
                fail_f.write(json.dumps(log_entry) + "\n")
                fail_f.flush()

    pbar.close()


    print("\n" + "=" * 60)
    print("MULTI-TURN GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total samples attempted : {total_attempts}")
    print(f"Successful evasions     : {existing_count}")
    print(f"Evasion rate            : {existing_count/max(total_attempts,1)*100:.1f}%")
    print(f"\nEvasions by step:")
    for i, count in enumerate(step_evasion_counts):
        if count > 0:
            print(f"  Step {i+1}: {count} evasions ({count/max(existing_count,1)*100:.1f}% of all evasions)")
    print("=" * 60)


if __name__ == "__main__":
    main()
