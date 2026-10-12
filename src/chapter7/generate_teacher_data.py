
import json
import os
import argparse
import re
import sys
import time
from pathlib import Path
import httpx




BASE_URL = "https://api.fireworks.ai/v1"

def read_api_key(environment_variable):
    key = os.environ.get(environment_variable, "").strip()
    if not key:
        raise RuntimeError(f"Missing API key in {environment_variable}")
    return key


def get_account_id(api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = httpx.get(f"{BASE_URL}/accounts", headers=headers)
        if resp.status_code == 200:
            accounts = resp.json().get("accounts", [])
            if accounts:
                
                name_field = accounts[0].get("name", "")
                if "/" in name_field:
                    return name_field.split("/")[-1]
        print(f"Error fetching account info: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Failed to fetch account ID: {e}")
    return None


def get_confusion_matrix(y_true, y_pred, labels=["malware", "benign", "uncertain"]):
    matrix = {t: {p: 0 for p in labels + ["error"]} for t in labels}
    for t, p in zip(y_true, y_pred):
        if p not in matrix[t]:
            matrix[t][p] = 0
        matrix[t][p] += 1
        
    print("\n" + "="*40)
    print("CONFUSION MATRIX")
    print("="*40)
    print(f"{'True \\ Predicted':>18} | " + " | ".join([f"{l[:9]:>9}" for l in labels + ["error"]]))
    print("-" * 65)
    for t in labels:
        if t == "uncertain": continue 
        row_str = f"{t:>18} | "
        for p in labels + ["error"]:
            row_str += f"{matrix[t][p]:>9} | "
        print(row_str)
    print("="*40 + "\n")


def build_user_prompt(log_text):
    return f"""You are an expert malware analyst labeling dynamic sandbox behavioral-summary logs.

Your task is to classify the sample as malware, benign, or uncertain based only on behaviors explicitly present in the provided log.

Output a single valid JSON object with the label and a detailed explanation where you explain your reasoning.
Do not output markdown, comments, or extra text outside those two parts.

Output format:

{{
  "suspicious_behaviors": [],
  "benign_behaviors": [],
  "missing_or_weak_evidence": [],
  "explanation": "",
  "label": "malware | benign | uncertain",
  "confidence": 0.0
}}

Log:
{log_text}"""


def create_batch_input_file(sample_files, batch_filepath, completed_paths, max_requests=500):
    count = 0
    with open(batch_filepath, "w") as f:
        for idx, item in enumerate(sample_files):
            fpath = item["path"]
            if fpath in completed_paths:
                continue
                
            if count >= max_requests:
                break
                
            try:
                with open(fpath, "r") as log_f:
                    log_data = json.load(log_f)
                
                if isinstance(log_data, dict) and "summary" in log_data:
                    log_text = json.dumps(log_data["summary"])
                else:
                    log_text = json.dumps(log_data)
                
                prompt = build_user_prompt(log_text)
                
                request_line = {
                    "custom_id": f"req_{idx}",
                    "body": {
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 4000,
                        "temperature": 0.1
                    }
                }
                f.write(json.dumps(request_line) + "\n")
                count += 1
            except Exception as e:
                print(f"Skipping {fpath} due to error: {e}")
    return count


def load_completed_results(output_log_path):
    completed_paths = set()
    completed_records = []
    if os.path.exists(output_log_path):
        with open(output_log_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if "path" in record:
                        completed_paths.add(record["path"])
                        completed_records.append(record)
                except Exception:
                    pass
    return completed_paths, completed_records


def process_response_text(response_text, true_label):
    
    cleaned = response_text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    
    
    json_match = re.search(r'(\{.*\})', cleaned.replace('\n', ' '), re.DOTALL)
    
    pred = "error"
    parsed_response = {}
    if json_match:
        try:
            parsed_response = json.loads(json_match.group(1))
            pred = parsed_response.get("label", "error").lower().strip()
            if "malware" in pred: pred = "malware"
            elif "benign" in pred: pred = "benign"
            elif "uncertain" in pred: pred = "uncertain"
            else: pred = "error"
        except json.JSONDecodeError:
            pred = "error"
            
    is_correct = (pred == true_label)
    return pred, parsed_response, is_correct


def start_batch_run(api_key, account_id, file_list_path, output_log_path, model_name,
                    max_requests=500, resume_job_id=None):
    if not api_key:
        print("Error: api_key is not set.")
        sys.exit(1)
        
    with open(file_list_path, "r") as f:
        manifest = json.load(f)
    sample_files = manifest.get("samples", manifest) if isinstance(manifest, dict) else manifest
    if not isinstance(sample_files, list):
        raise ValueError("Teacher sample manifest must contain a samples list")
        
    output_path = Path(output_log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_input_path = f"{output_log_path}.batch-input.jsonl"
    completed_paths, completed_records = load_completed_results(output_log_path)
    print(f"Loaded {len(sample_files)} total files.")
    print(f"Already completed: {len(completed_paths)} files.")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    client = httpx.Client(headers=headers, timeout=120.0)
    
    job_id = resume_job_id
    output_dataset_id = None
    
    if job_id:
        print(f"Resuming existing batch job: {job_id}...")
        status_resp = client.get(
            f"{BASE_URL}/accounts/{account_id}/batchInferenceJobs/{job_id}"
        )
        if status_resp.status_code != 200:
            print(f"Failed to fetch job info for {job_id}: {status_resp.status_code} - {status_resp.text}")
            return
            
        status_info = status_resp.json()
        full_out_ds = status_info.get("outputDatasetId", "")
        if "/" in full_out_ds:
            output_dataset_id = full_out_ds.split("/")[-1]
            print(f"Found output dataset: {output_dataset_id}")
        else:
            print(f"Warning: Could not parse output dataset ID from job detail. detail: {status_info}")
            
    else:
        remaining_count = len(sample_files) - len(completed_paths)
        if remaining_count == 0:
            print("All files are already processed!")
            y_true = [r["target_label"] for r in completed_records]
            y_pred = [r["predicted_label"] for r in completed_records]
            get_confusion_matrix(y_true, y_pred)
            return
            
        print(f"Remaining to process: {remaining_count}")
        
        
        created_count = create_batch_input_file(sample_files, batch_input_path, completed_paths, max_requests)
        if created_count == 0:
            print("No requests generated for batch.")
            return
            
        print(f"Submitting batch of size: {created_count}")
        
        
        timestamp = int(time.time())
        dataset_id = f"batch-input-{timestamp}"
        output_dataset_id = f"batch-output-{timestamp}"
        job_id = f"batch-job-{timestamp}"
        
        print(f"1. Creating dataset entry: {dataset_id}...")
        dataset_payload = {
            "datasetId": dataset_id,
            "dataset": {
                "userUploaded": {},
                "exampleCount": created_count
            }
        }
        
        create_ds_resp = client.post(
            f"{BASE_URL}/accounts/{account_id}/datasets",
            json=dataset_payload
        )
        
        if create_ds_resp.status_code != 200:
            print(f"Dataset creation failed: {create_ds_resp.status_code} - {create_ds_resp.text}")
            return
            
        
        print(f"2. Uploading JSONL file: {batch_input_path}...")
        upload_headers = {"Authorization": f"Bearer {api_key}"}
        with open(batch_input_path, "rb") as f:
            upload_resp = httpx.post(
                f"{BASE_URL}/accounts/{account_id}/datasets/{dataset_id}:upload",
                headers=upload_headers,
                files={"file": (batch_input_path, f, "application/octet-stream")},
                timeout=600.0
            )
            
        if upload_resp.status_code != 200:
            print(f"Dataset upload failed: {upload_resp.status_code} - {upload_resp.text}")
            return
            
        print("Upload completed successfully.")
        
        
        print(f"3. Submitting Batch Inference Job: {job_id} using model {model_name}...")
        job_payload = {
            "model": model_name,
            "inputDatasetId": f"accounts/{account_id}/datasets/{dataset_id}",
            "outputDatasetId": f"accounts/{account_id}/datasets/{output_dataset_id}",
            "inferenceParameters": {
                "maxTokens": 4000,
                "temperature": 0.1
            }
        }
        
        job_resp = client.post(
            f"{BASE_URL}/accounts/{account_id}/batchInferenceJobs?batchInferenceJobId={job_id}",
            json=job_payload
        )
        
        if job_resp.status_code != 200:
            print(f"Batch job creation failed: {job_resp.status_code} - {job_resp.text}")
            return
            
        print(f"Batch job created. Job ID: {job_id}")
        
    
    print("Waiting for batch job to finish. Polling every 20 seconds...")
    completed = False
    status_info = {}
    
    try:
        while not completed:
            status_resp = client.get(
                f"{BASE_URL}/accounts/{account_id}/batchInferenceJobs/{job_id}"
            )
            if status_resp.status_code != 200:
                print(f"Failed to fetch job status: {status_resp.status_code}. Retrying in 20s...")
                time.sleep(20)
                continue
                
            status_info = status_resp.json()
            state = status_info.get("state", "UNKNOWN")
            print(f"Current Job State: {state}")
            
            if state in ["JOB_STATE_COMPLETED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
                completed = True
            else:
                time.sleep(20)
    except KeyboardInterrupt:
        print(f"\nPolling interrupted by user. The Batch job '{job_id}' is running on Fireworks.ai.")
        print("Run this command again with --resume-job-id " + job_id)
        sys.exit(0)
        
    final_state = status_info.get("state", "UNKNOWN")
    if final_state != "JOB_STATE_COMPLETED":
        print(f"Batch job failed or cancelled with state: {final_state}")
        print(status_info)
        return
        
    
    if not output_dataset_id:
        full_out_ds = status_info.get("outputDatasetId", "")
        if "/" in full_out_ds:
            output_dataset_id = full_out_ds.split("/")[-1]
            
    if not output_dataset_id:
        print("Error: Output dataset identifier not found.")
        return
        
    
    print("5. Retrieving signed download URL for batch results...")
    dl_resp = client.get(
        f"{BASE_URL}/accounts/{account_id}/datasets/{output_dataset_id}:getDownloadEndpoint"
    )
    
    if dl_resp.status_code != 200:
        print(f"Failed to retrieve download endpoint: {dl_resp.status_code} - {dl_resp.text}")
        return
        
    dl_info = dl_resp.json()
    filename_to_signed_urls = dl_info.get("filenameToSignedUrls", {})
    if not filename_to_signed_urls:
        print("Completed successfully but output dataset is empty or not found.")
        return
        
    
    results_map = {}
    
    for filename, signed_url in filename_to_signed_urls.items():
        print(f"Downloading results fraction: {filename}...")
        try:
            content_resp = httpx.get(signed_url, timeout=120.0)
            if content_resp.status_code == 200:
                lines = content_resp.text.strip().split("\n")
                for line in lines:
                    if not line.strip(): continue
                    res_obj = json.loads(line)
                    results_map[res_obj["custom_id"]] = res_obj
            else:
                print(f"Failed to download fraction {filename}: {content_resp.status_code}")
        except Exception as e:
            print(f"Error downloading fraction {filename}: {e}")
            
    
    new_records = []
    with open(output_log_path, "a") as out_txt:
        for idx, item in enumerate(sample_files):
            fpath = item["path"]
            if fpath in completed_paths:
                continue
                
            custom_id = f"req_{idx}"
            if custom_id in results_map:
                res_obj = results_map[custom_id]
                true_label = "benign" if item["label"] == "goodware" else "malware"
                
                
                err = res_obj.get("error")
                if err:
                    pred = "error"
                    response_text = f"Batch Error: {json.dumps(err)}"
                    parsed_response = {}
                    is_correct = False
                else:
                    resp_dict = res_obj.get("response", {})
                    
                    if "choices" in resp_dict:
                        choices_list = resp_dict.get("choices", [{}])
                    else:
                        choices_list = resp_dict.get("body", {}).get("choices", [{}])
                    response_text = choices_list[0].get("message", {}).get("content", "") if choices_list else ""
                    pred, parsed_response, is_correct = process_response_text(response_text, true_label)
                
                result_log = {
                    "path": fpath,
                    "target_label": true_label,
                    "predicted_label": pred,
                    "is_correct": is_correct,
                    "json_output": parsed_response,
                    "raw_response_cot": response_text,
                    "split": item.get("split"),
                    "length_bucket": item.get("length_bucket"),
                    "source_tokens": item.get("tokens"),
                }
                out_txt.write(json.dumps(result_log) + "\n")
                new_records.append(result_log)
                
    
    if os.path.exists(batch_input_path):
        os.remove(batch_input_path)
        
    print("Done! Evaluation completed.")
    
    
    all_paths, all_records = load_completed_results(output_log_path)
    y_true = [r["target_label"] for r in all_records]
    y_pred = [r["predicted_label"] for r in all_records]
    
    correct = sum(1 for r in all_records if r["is_correct"])
    if len(all_records) > 0:
        print(f"Accuracy: {correct / len(all_records) * 100:.2f}% ({correct}/{len(all_records)})")
    get_confusion_matrix(y_true, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-requests", type=int, default=500)
    parser.add_argument("--resume-job-id")
    parser.add_argument("--api-key-env", default="FIREWORKS_API_KEY")
    parser.add_argument("--account-id")
    args = parser.parse_args()

    api_key = read_api_key(args.api_key_env)
    account_id = args.account_id or get_account_id(api_key)
    if not account_id:
        raise RuntimeError("Could not determine the Fireworks account ID")

    start_batch_run(
        api_key,
        account_id,
        args.input,
        args.output,
        args.model,
        max_requests=args.max_requests,
        resume_job_id=args.resume_job_id,
    )
