# Chapter 5: Single-Turn Attacks

Generates single-turn adversarial reports, trains the SFT and GRPO adapters, evaluates them, and plots the results.

## Files

- `common.py`: Contains the MLP and local Transformer report embedder.
- `augmented_dataset.py`: Applies generated JSON additions and edits to a report.
- `create_data.py`: Queries an Ollama model and saves successful and failed attack attempts.
- `train_sft.py`: Trains the initial adversarial SFT adapter.
- `train_grpo.py`: Continues training with classifier-based GRPO rewards.
- `evaluate.py`: Evaluates a base model or adapter on the adversarial split.
- `analyze.py`: Creates the Chapter 5 figures from evaluation and trainer outputs.

## Required Arguments

- `create_data.py`
  - `--api-url`: running Ollama server URL
  - `--classifier`: MLP checkpoint
  - `--splits`: split JSON
  - `--embedding-model`: embedding checkpoint
  - `--output`: successful attacks JSONL
  - `--failed-output`: failed attacks JSONL
- `train_sft.py`
  - `--data`: successful attacks JSONL
  - `--output`: adapter output directory
- `train_grpo.py`
  - `--model`: SFT model or adapter
  - `--classifier`: MLP checkpoint
  - `--splits`: split JSON
  - `--embedding-model`: embedding checkpoint
  - `--output`: GRPO output directory
- `evaluate.py`
  - `--model`: base model
  - `--classifier`: MLP checkpoint
  - `--splits`: split JSON
  - `--embedding-model`: embedding checkpoint
  - `--output`: evaluation JSONL
- `analyze.py`
  - `--sft-results`: SFT evaluation JSONL
  - `--grpo-results`: GRPO evaluation JSONL
  - `--trainer-state`: GRPO trainer state JSON
  - `--output`: figure directory

- External services and files
  - Ollama server
  - Ollama model: `qwen3.5:35b`
  - `sum.json`: AutoRobust summary reference
  - `corpus.json`: AutoRobust behavior corpus
