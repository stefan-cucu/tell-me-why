import json
import re
from pathlib import Path

import json5


PREDICTION_LABELS = ("malware", "benign", "uncertain", "error")


def normalize_label(value):
    text = str(value or "").strip().lower()
    if re.search(r"\bmalware\b|\bmalicious\b", text):
        return "malware"
    if re.search(r"\bbenign\b|\bgoodware\b", text):
        return "benign"
    if re.search(r"\buncertain\b", text):
        return "uncertain"
    return "error"


def resolved(path):
    return str(Path(path).expanduser().resolve())


def load_jsonl(path, successful_only=False):
    records = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not record.get("path"):
                continue
            if successful_only and record.get("status", "completed") not in {
                "completed",
                "ok",
            }:
                continue
            records[resolved(record["path"])] = record
    return records


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_report(path):
    with Path(path).open(encoding="utf-8") as handle:
        report = json.load(handle)
    if isinstance(report, dict):
        report = report.get("summary", report)
    if not isinstance(report, dict):
        raise ValueError(f"Report is not a JSON object: {path}")
    return report


def prediction(record):
    if record.get("status", "completed") not in {"completed", "ok"}:
        return "error"
    value = record.get("predicted_label")
    if value is None and isinstance(record.get("json_output"), dict):
        value = record["json_output"].get("label")
    return normalize_label(value)


def target(record):
    return normalize_label(record.get("target_label") or record.get("teacher_label"))


def explanation(record, include_behavior_lists=False):
    payload = record.get("json_output")
    if not isinstance(payload, dict):
        payload = {}
    parts = []
    if include_behavior_lists:
        suspicious = payload.get("suspicious_behaviors") or []
        benign = payload.get("benign_behaviors") or []
        if suspicious:
            parts.append("Suspicious behaviors:\n- " + "\n- ".join(map(str, suspicious)))
        if benign:
            parts.append("Benign behaviors:\n- " + "\n- ".join(map(str, benign)))
    text = str(payload.get("explanation") or "").strip()
    if text:
        parts.append("Explanation:\n" + text if include_behavior_lists else text)
    if not parts and include_behavior_lists:
        reasoning = str(record.get("generated_reasoning") or "").strip()
        if reasoning:
            parts.append("Reasoning:\n" + reasoning)
    return "\n\n".join(parts)


def response_prompt(text):
    marker = "### Response:\n"
    if marker not in text:
        raise ValueError("Dataset text lacks the response marker")
    return text.split(marker, 1)[0] + marker


def prompt_with_report(text, report):
    input_marker = "### Input:\n"
    response_marker = "### Response:\n"
    if input_marker not in text or response_marker not in text:
        raise ValueError("Dataset text lacks input and response markers")
    prefix = text.split(input_marker, 1)[0]
    report_json = json.dumps(report, ensure_ascii=False, sort_keys=True)
    return prefix + input_marker + report_json + "\n\n" + response_marker


def balanced_object(text):
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_reasoning(text):
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def parse_completion(text):
    without_reasoning = re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
    ).strip()
    candidate = balanced_object(without_reasoning)
    errors = []
    if candidate:
        for parser_name, parser in (("json", json.loads), ("json5", json5.loads)):
            try:
                payload = parser(candidate)
                if isinstance(payload, dict):
                    label = normalize_label(payload.get("label"))
                    if label != "error":
                        return label, payload, parser_name, None
            except Exception as error:
                errors.append(f"{parser_name}: {error}")
    else:
        errors.append("No complete JSON object")
    match = re.search(
        r'''["']?label["']?\s*:\s*["']?(malware|malicious|benign|goodware|uncertain)''',
        text,
        flags=re.IGNORECASE,
    )
    if match:
        value = match.group(1)
        return normalize_label(value), {"label": value}, "label_regex", "; ".join(errors)
    return "error", {}, "error", "; ".join(errors)


def malware_score(label, payload):
    label = normalize_label(label)
    try:
        confidence = float((payload or {}).get("confidence"))
    except (TypeError, ValueError):
        return 0.5 if label == "uncertain" else None
    confidence = min(max(confidence, 0.0), 1.0)
    if label == "malware":
        return confidence
    if label == "benign":
        return 1.0 - confidence
    return 0.5 if label == "uncertain" else None


def classification_metrics(records):
    matrix = {
        truth: {predicted: 0 for predicted in PREDICTION_LABELS}
        for truth in ("malware", "benign")
    }
    for record in records:
        truth = target(record)
        if truth in matrix:
            matrix[truth][prediction(record)] += 1
    total = sum(sum(row.values()) for row in matrix.values())
    correct = matrix["malware"]["malware"] + matrix["benign"]["benign"]
    per_class = {}
    for label in ("malware", "benign"):
        true_positive = matrix[label][label]
        false_negative = sum(matrix[label].values()) - true_positive
        false_positive = sum(matrix[other][label] for other in matrix if other != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(values["f1"] for values in per_class.values()) / 2,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def parse_named_paths(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, received {value!r}")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name or name in result:
            raise ValueError(f"Invalid or duplicate name: {name!r}")
        result[name] = Path(path)
    return result
