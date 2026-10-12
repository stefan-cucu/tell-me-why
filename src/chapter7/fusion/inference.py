import json
import json5
import os
import re

import torch
from datasets import DatasetDict
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, BitsAndBytesConfig

from fusion.model import (
    FUSION_CONFIG_FILE,
    LLM_ADAPTER_DIR,
    FusionConfig,
    build_fusion_model,
    load_fusion_state,
)

BINARY_LABELS = {0: "benign", 1: "malware"}

def normalize_label(label):
    label = str(label or "").lower().strip()
    if "malware" in label:
        return "malware"
    if "benign" in label or "goodware" in label:
        return "benign"
    if "uncertain" in label:
        return "uncertain"
    return "error"

def label_from_payload(parsed):
    if not isinstance(parsed, dict):
        return "error"
    for key in ("label", "decision", "classification", "prediction", "verdict"):
        label = normalize_label(parsed.get(key))
        if label != "error":
            return label
    classifier_signal = parsed.get("classifier_signal")
    if isinstance(classifier_signal, dict):
        for key in ("label", "decision", "classification", "prediction", "verdict", "classifier_predicts"):
            label = normalize_label(classifier_signal.get(key))
            if label != "error":
                return label
    return "error"

def regex_label_from_text(text):
    for key in ("label", "decision", "classification", "prediction", "verdict"):
        match = re.search(rf'"{key}"\s*:\s*"([^"\n]+)"', text, flags=re.IGNORECASE)
        if match:
            label = normalize_label(match.group(1))
            if label != "error":
                return label, {key: match.group(1), "parse_source": "json_key_regex"}
    free_text_patterns = (
        r"Classifier signal:\s*classifier predicts\s+(malware|benign)",
        r"Final classification:\s*(?:classify(?:\s+as)?\s+)?(malware|benign)",
        r"Final Label:\s*(malware|benign)",
        r"Decision:\s*(?:classify(?:\s+as)?|classification is)?\s*(malware|benign)",
    )
    for pattern in free_text_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            label = normalize_label(match.group(1))
            if label != "error":
                return label, {"label": match.group(1), "parse_source": "free_text_regex"}
    return "error", {}

def extract_json_object(text):
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start == -1:
        return None, "No JSON object start found"
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start:idx + 1], None
    return None, "No complete JSON object found"

def parse_prediction(text):
    json_text, err = extract_json_object(text)
    if err:
        label, payload = regex_label_from_text(text)
        if label != "error":
            return label, payload, err
        return "error", {}, err
    try:
        parsed = json.loads(json_text)
        return label_from_payload(parsed), parsed, None
    except json.JSONDecodeError as json_exc:
        try:
            parsed = json5.loads(json_text)
            return label_from_payload(parsed), parsed, f"Recovered with json5 after JSON parse error: {json_exc}"
        except Exception as json5_exc:
            label, payload = regex_label_from_text(json_text)
            if label == "error":
                label, payload = regex_label_from_text(text)
            if label != "error":
                return label, payload, f"JSON/json5 parse error: {json_exc}; {json5_exc}"
            return "error", {}, f"JSON/json5 parse error: {json_exc}; {json5_exc}"

def _decision_scalar(decision, key, default=None):
    value = decision.get(key) if isinstance(decision, dict) else None
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return value.detach().cpu().view(-1)[0].item()
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value

def _decision_probs(decision):
    probs = decision.get("classifier_probs") if isinstance(decision, dict) else None
    if probs is None:
        return None
    if torch.is_tensor(probs):
        if probs.numel() == 0:
            return None
        return [float(x) for x in probs.detach().cpu().view(-1)[:2].tolist()]
    if isinstance(probs, (list, tuple)) and probs:
        first = probs[0] if isinstance(probs[0], (list, tuple)) else probs
        return [float(x) for x in first[:2]]
    return None

def classifier_output_from_decision(decision):
    label_id = _decision_scalar(decision, "classifier_label")
    confidence = _decision_scalar(decision, "classifier_confidence")
    label = BINARY_LABELS.get(int(label_id)) if label_id is not None else "error"
    probs = _decision_probs(decision)
    return label or "error", float(confidence) if confidence is not None else None, probs

def classifier_authoritative_fallback_json(label, confidence, raw_json):
    payload = dict(raw_json) if isinstance(raw_json, dict) else {}
    payload["label"] = label
    if confidence is not None:
        payload["confidence"] = confidence
    payload.setdefault("suspicious_behaviors", [])
    payload.setdefault("benign_behaviors", [])
    payload.setdefault("missing_or_weak_evidence", [])
    payload.setdefault("explanation", "")
    payload["classifier_authoritative"] = True
    return payload

def get_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32

def load_dataset(dataset_path, split, num_samples):
    ds = DatasetDict.load_from_disk(dataset_path)
    dataset = ds[split]
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    return dataset

def load_model(args):
    dtype = get_dtype()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(args.fusion_dir, FUSION_CONFIG_FILE)) as f:
        fusion_config = FusionConfig.from_dict(json.load(f))

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ) if args.load_llm_in_4bit else None
    device_map = {"": 0} if torch.cuda.is_available() else None

    llm = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        quantization_config=quant_config,
        device_map=device_map,
    )
    llm = PeftModel.from_pretrained(llm, os.path.join(args.fusion_dir, LLM_ADAPTER_DIR), is_trainable=False)
    llm.eval()

    embedding_model = AutoModel.from_pretrained(
        fusion_config.embedding_model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
    )
    embedding_model.eval()

    model = build_fusion_model(
        llm=llm,
        embedding_model=embedding_model,
        classifier_path=fusion_config.classifier_path,
        num_memory_tokens=fusion_config.num_memory_tokens,
        cross_attn_hidden_size=fusion_config.cross_attn_hidden_size,
        num_cross_attention_heads=fusion_config.num_cross_attention_heads,
        cross_attn_layers=fusion_config.cross_attn_layers,
        aux_ce_weight=0.0,
        embedding_model_name=fusion_config.embedding_model_name,
        classifier_threshold=args.classifier_threshold,
    )
    load_fusion_state(model, args.fusion_dir)
    model.move_fusion_modules(device=device, dtype=dtype)
    model.eval()
    return model, fusion_config
