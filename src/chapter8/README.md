# Chapter 8: Evaluation

Evaluates all four model variants, scores explanation quality and faithfulness, and creates the final plots and examples.

## Files

- `common.py`: Contains shared prediction parsing, report loading, and metric helpers.
- `evaluate_language_models.py`: Evaluates the base Qwen model or the SFT-only model on a shared cohort.
- `judge_explanations.py`: Uses an anonymous Ollama judge to score four explanation systems.
- `evaluate_faithfulness.py`: Selects a shared malware cohort, maps explanations to report entries, and scores the modified reports.
- `analyze.py`: Calculates classification metrics and creates plots and selected Markdown examples.

## Required Arguments

- `evaluate_language_models.py`
  - `--backend`: `base` or `sft`
  - `--dataset`: evaluation dataset
  - `--reference`: shared cohort predictions
  - `--output`: prediction JSONL
  - `--host`: Ollama server URL
  - `--base-model`: base checkpoint
  - `--adapter`: SFT adapter
- `judge_explanations.py`
  - `--prediction`:  prediction files
  - `--host`: judge Ollama server URL
  - `--output`: raw judge JSONL
  - `--summary`: summary JSON
  - `--csv`: score CSV
- `evaluate_faithfulness.py prepare`
  - `--prediction`: input prediction files.
  - `--output`: cohort JSON
- `evaluate_faithfulness.py select`
  - `--predictions`: one model's predictions
  - `--cohort`: cohort JSON
  - `--host`: Ollama server URL
  - `--output`: deletion manifest JSONL
- `evaluate_faithfulness.py score`
  - `--backend`: `base`, `sft`, or `fusion`
  - `--manifest`: deletion manifest JSONL
  - `--dataset`: evaluation dataset
  - `--output`: faithfulness JSONL
  - `--host`: Ollama server URL
  - `--base-model`: base checkpoint
  - `--adapter`: SFT adapter
  - `--fusion-model`: fusion checkpoint
- `analyze.py`
  - `--prediction`: input prediction files
  - `--output`: metrics output directory
  - `--figures`: figure output directory.
- External requirements
  - Ollama for base evaluation, explanation judging, evidence selection
  - Base model: `qwen3:8b`
  - Judge and selector model: `qwen3.6:27b`
