
import os



import copy
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
    merge_diff,
    count_changes,
)


CLASSES = ['goodware', 'malware']


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


def load_samples(split_file, split_name):
    with open(split_file, 'r') as f:
        splits = json.load(f)

    adv_paths = splits.get(split_name, [])
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


def score_report(report_data, embedder, classifier, device):
    try:
        embedding_tensor = embedder.embed(report_data).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = classifier(embedding_tensor)
            probs = torch.softmax(logits, dim=1)
            return probs[0, CLASSES.index('goodware')].item()
    except Exception:
        return None


def best_candidate(completions, current_report, classifier, device, embedder, verbose=False):
    
    best_result = None
    best_score = -1.0

    for i, completion in enumerate(completions):
        diff, parse_err = parse_diff(completion)
        if diff is None:
            if verbose:
                print(f"      [candidate {i}] parse failed: {parse_err}")
            continue

        if isinstance(diff, dict) and 'summary' in diff and isinstance(diff['summary'], dict):
            diff = diff['summary']

        try:
            modified = augmented_dataset.apply_perturbation_diff(copy.deepcopy(current_report), diff)
            score = score_report(modified, embedder, classifier, device)
            if score is None:
                if verbose:
                    print(f"      [candidate {i}] embedding failed")
                continue

            if verbose:
                print(f"      [candidate {i}] P(gw)={score:.4f}")

            if score > best_score:
                best_score = score
                best_result = (diff, modified, score, completion)

        except Exception as e:
            if verbose:
                print(f"      [candidate {i}] error: {e}")
            continue

    return best_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--split-name", required=True, choices=["train", "adv_gen"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sum", default="sum.json")
    parser.add_argument("--corpus", default="corpus.json")
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    
    print("Loading malware validation splits...")
    samples = load_samples(args.splits, args.split_name)
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
            max_model_len=40960,
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

    from vllm.lora.request import LoRARequest
    lora_request = LoRARequest("lora_adapter", 1, args.adapter) if use_lora else None

    sampling_params = SamplingParams(
        n=args.attempts,
        temperature=0.9,
        max_tokens=16000,
        stop=["<|end_of_text|>", "<|im_end|>"]
    )

    
    print("Loading reference data...")
    sum_entries, corpus_terms = load_references(args.sum, args.corpus)
    importance_ranking_str = format_importance_ranking(DEFAULT_CATEGORY_IMPORTANCE)

    ref_data = sample_references(sum_entries, corpus_terms, 300)
    reference_text = format_references(ref_data)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        valid_categories=str(VALID_CATEGORIES),
        importance_ranking=importance_ranking_str,
        reference_data=reference_text,
    )

    MAX_PROMPT_CHARS = 30000 * 3.5

    
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving logs to: {output_dir}")

    effective_successes = 0
    total_evaluated = 0
    step_evasion_counts = [0] * args.max_steps

    pbar = tqdm(total=len(reports), desc="Multi-turn Evasion Test")

    for sample in reports:
        report_data = sample['report'].get('summary', sample['report'])
        report_json = json.dumps(report_data)

        if len(report_json) > MAX_PROMPT_CHARS:
            if args.verbose:
                print(f"\n[SKIP] {sample['filename']} too large ({len(report_json)} chars)")
            pbar.update(1)
            continue

        
        initial_score = score_report(report_data, embedder, classifier, device)
        if initial_score is None:
            if args.verbose:
                print(f"\n[SKIP] Embedding failed for {sample['filename']}")
            pbar.update(1)
            continue

        if initial_score > 0.5:
            if args.verbose:
                print(f"\n[SKIP] {sample['filename']} already goodware (P={initial_score:.4f})")
            pbar.update(1)
            continue

        total_evaluated += 1

        if args.verbose:
            print(f"\n[SAMPLE {total_evaluated}] {sample['filename']} — P(gw)={initial_score:.4f}")

        
        accumulated_diff = {}
        current_report = copy.deepcopy(report_data)
        current_score = initial_score
        success = False
        score_progression = [initial_score]
        all_completions = []

        for step in range(args.max_steps):
            if args.verbose:
                print(f"  [STEP {step+1}/{args.max_steps}] P(gw)={current_score:.4f}")

            
            user_msg = build_message(
                report_data, initial_score,
                accumulated_diff, current_score, step
            )

            messages = [{"role": "system", "content": system_prompt}]
            messages += FEW_SHOT_EXAMPLES
            messages.append({"role": "user", "content": user_msg})

            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
            if token_count > 32000:
                if args.verbose:
                    print(f"    [SKIP STEP] Prompt too long ({token_count} tokens)")
                break

            
            outputs = llm.generate([prompt], sampling_params, lora_request=lora_request)
            completions = [o.text for o in outputs[0].outputs]
            all_completions.extend(completions)

            
            saved_report = copy.deepcopy(current_report)
            saved_diff = copy.deepcopy(accumulated_diff)

            
            result = best_candidate(
                completions, current_report, classifier, device, embedder,
                verbose=args.verbose
            )

            if result is None:
                if args.verbose:
                    print(f"    [FAIL] No valid candidates at step {step+1}")
                continue

            best_diff, best_modified, best_score, best_completion = result
            delta = best_score - current_score

            
            if delta < 0 and best_score <= 0.5:
                if args.verbose:
                    print(f"    [NEG-DELTA] {current_score:.4f} → {best_score:.4f} (delta: {delta:+.4f}). Retrying...")

                
                retry_outputs = llm.generate([prompt], sampling_params, lora_request=lora_request)
                rebest_candidate = [o.text for o in retry_outputs[0].outputs]
                all_completions.extend(rebest_candidate)

                retry_result = best_candidate(
                    rebest_candidate, current_report, classifier, device, embedder,
                    verbose=args.verbose
                )

                if retry_result is None:
                    if args.verbose:
                        print(f"    [NEG-DELTA RETRY FAIL] No valid candidates. Aborting sample.")
                    break

                retry_diff, retry_modified, retry_score, retry_completion = retry_result
                retry_delta = retry_score - current_score

                if retry_delta < 0 and retry_score <= 0.5:
                    if args.verbose:
                        print(f"    [NEG-DELTA RETRY FAIL] Still negative ({current_score:.4f} → {retry_score:.4f}). Aborting.")
                    score_progression.append(retry_score)
                    break
                else:
                    
                    if args.verbose:
                        print(f"    [NEG-DELTA RETRY OK] {current_score:.4f} → {retry_score:.4f}")
                    current_report = retry_modified
                    accumulated_diff = merge_diff(accumulated_diff, retry_diff)
                    current_score = retry_score
                    score_progression.append(current_score)
                    delta = retry_delta

                    if current_score > 0.5:
                        step_evasion_counts[step] += 1
                        success = True
                        if args.verbose:
                            print(f"    [SUCCESS] Evasion at step {step+1} (via retry)!")
                        break
                    continue

            
            current_report = best_modified
            accumulated_diff = merge_diff(accumulated_diff, best_diff)
            current_score = best_score
            score_progression.append(current_score)

            if args.verbose:
                a, e = count_changes(accumulated_diff)
                print(f"    [SCORE] P(gw)={current_score:.4f} (delta: {delta:+.4f}) | Total: {a} adds, {e} edits")

            if current_score > 0.5:
                step_evasion_counts[step] += 1
                success = True
                if args.verbose:
                    print(f"    [SUCCESS] Evasion at step {step+1}!")
                break

        
        if success:
            effective_successes += 1

        prefix = "evaded" if success else "failed"
        save_path = os.path.join(output_dir, f"{prefix}_{sample['filename']}")
        try:
            with open(save_path, "w") as f:
                json.dump({
                    "evaded": success,
                    "initial_score": initial_score,
                    "final_score": current_score,
                    "score_progression": score_progression,
                    "steps_taken": len(score_progression) - 1,
                    "original": report_data,
                    "accumulated_diff": accumulated_diff,
                    "evaded_data": current_report if success else None,
                    "all_completions": all_completions,
                }, f, indent=2, default=str)
        except Exception as e:
            print(f"\n[ERROR] Could not write log for {sample['filename']}: {e}")

        pbar.update(1)

        if total_evaluated > 0:
            ev_rate = (effective_successes / total_evaluated) * 100
            pbar.set_postfix({
                "Ev Rate": f"{ev_rate:.1f}%",
                "P(gw)": f"{current_score:.3f}",
                "Steps": len(score_progression) - 1,
            })

        if total_evaluated % args.log_every == 0 and total_evaluated > 0:
            print(f"\n[Log at {total_evaluated}] Evaded: {effective_successes}/{total_evaluated} "
                  f"({effective_successes/total_evaluated*100:.2f}%)")

    pbar.close()

    
    print("\n" + "=" * 60)
    print("MULTI-TURN EVASION TEST SUMMARY")
    print("=" * 60)
    print(f"Total Samples Evaluated : {total_evaluated}")
    print(f"Total Evaded            : {effective_successes}")
    if total_evaluated > 0:
        print(f"Evasion Rate            : {effective_successes/total_evaluated*100:.2f}%")
    print(f"\nEvasions by step:")
    for i, count in enumerate(step_evasion_counts):
        if count > 0:
            print(f"  Step {i+1}: {count} evasions ({count/max(effective_successes,1)*100:.1f}% of all evasions)")
    print("=" * 60)


if __name__ == "__main__":
    main()
