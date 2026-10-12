# Chapter 6: Multi-Turn Hardening

Runs iterative attacks, trains the robust classifier, compares it with the baseline, and creates the Chapter 6 figures.

## Files

- `common.py`: Contains the MLP and local Transformer report embedder.
- `augmented_dataset.py`: Applies generated JSON additions and edits to a report.
- `create_data.py`: Collects multi-turn teacher attacks through Ollama.
- `train_sft.py`: Trains the multi-turn SFT adapter from successful and failed trajectories.
- `train_grpo.py`: Continues training with classifier-based GRPO rewards.
- `run_attack.py`: Runs filewise attacks on the training or adversarial split.
- `train_classifier.py`: Fine-tunes the baseline MLP using original and adversarial reports.
- `compare_classifiers.py`: Compares the baseline and robust classifiers on the saved embeddings.
- `analyze.py`: Creates the multi-turn attack figures.

## Required Arguments

- `create_data.py`
  - `--api-url`: Ollama server URL
  - `--classifier`: MLP checkpoint
  - `--splits`: split JSON
  - `--embedding-model`: embedding checkpoint
  - `--output`: successful trajectories JSONL
  - `--failed-output`: failed trajectories JSONL
- `train_sft.py`
  - `--success-data`: successful trajectories JSONL
  - `--failed-data`: failed trajectories JSONL
  - `--output`: adapter output directory
- `train_grpo.py`
  - `--model`: SFT model or adapter
  - `--classifier`: MLP checkpoint
  - `--splits`: split JSON
  - `--embedding-model`: embedding checkpoint
  - `--output`: GRPO output directory
- `run_attack.py`
  - `--model`: base model or checkpoint
  - `--classifier`: MLP checkpoint
  - `--splits`: split JSON
  - `--split-name`: `train` or `adv_gen`
  - `--embedding-model`: embedding checkpoint
  - `--output`: attack output directory
- `train_classifier.py`
  - `--classifier`: baseline MLP checkpoint
  - `--adversarial-dirs`: attack output directories
  - `--splits`: split JSON
  - `--embedding-model`: embedding checkpoint
  - `--cache`: embedding cache directory
  - `--output`: robust MLP checkpoint
- `compare_classifiers.py`
  - `--embeddings`: baseline embedding directory
  - `--baseline`: baseline MLP checkpoint
  - `--robust`: robust MLP checkpoint
  - `--output`: comparison output directory
- `analyze.py`
  - `--results`: attack evaluation JSONL
  - `--output`: figure directory

- External services and files
  - Ollama server
  - Ollama model: `qwen3.5:35b`
  - `sum.json`: AutoRobust summary reference
  - `corpus.json`: AutoRobust behavior corpus
