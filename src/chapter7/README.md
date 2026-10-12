# Chapter 7: Classifier-LLM Fusion

Creates teacher explanations, trains the explanation SFT, prepares classifier-conditioned data, and trains and validates the fused model.

## Files

- `sample_teacher_data.py`: Selects train and test reports for teacher generation.
- `generate_teacher_data.py`: Sends the selected reports to GLM-5.2 through the Fireworks batch API.
- `train_sft.py`: Builds the teacher dataset and trains the explanation SFT adapter.
- `validate_sft.py`: Evaluates the explanation-only SFT model.
- `prepare_fusion_data.py`: Adds classifier labels and confidence values and removes teacher-classifier disagreements.
- `train_fusion.py`: Trains the cross-attention fusion modules.
- `validate_fusion.py`: Generates classifier-conditioned reasoning and final JSON outputs.
- `analyze.py`: Creates the fusion training figure and validation summary.
- `fusion/classifier.py`: Defines the classifier and embedding wrapper used by fusion.
- `fusion/data.py`: Loads and batches fusion training examples.
- `fusion/model.py`: Defines the memory tokens, cross-attention layers, and fused model.
- `fusion/inference.py`: Contains shared generation and checkpoint-loading helpers.
- `fusion/__init__.py`: Exposes the fusion package.

## Required Arguments

- `sample_teacher_data.py`
  - `--splits`: split JSON
  - `--output`: teacher manifest JSON
- `generate_teacher_data.py`
  - `--input`: teacher manifest JSON
  - `--output`: teacher results JSONL
  - `--model`: Fireworks model ID
- `train_sft.py`
  - `--result-files`: teacher result JSONL files
  - `--splits`: split JSON
  - `--dataset`: processed dataset directory
  - `--output`: SFT adapter directory
- `validate_sft.py`
  - `--model`: SFT adapter directory
  - `--dataset`: processed dataset directory
  - `--output`: validation JSONL
- `prepare_fusion_data.py`
  - `--dataset`: SFT dataset directory
  - `--output`: fusion dataset directory
  - `--classifier`: MLP checkpoint
  - `--llm-tokenizer`: LLM tokenizer or adapter
- `train_fusion.py`
  - `--sft-adapter`: SFT adapter directory
  - `--classifier`: MLP checkpoint
  - `--dataset`: fusion dataset directory
  - `--output`: fusion checkpoint directory
- `validate_fusion.py`
  - `--fusion-model`: fusion checkpoint directory
  - `--dataset`: fusion dataset directory
  - `--output`: validation JSONL
- `analyze.py`
  - `--validation`: fusion validation JSONL
  - `--trainer-state`: trainer state JSON
  - `--fusion-history`: W&B history CSV
  - `--output`: analysis output directory

- External service
  - Fireworks URL: `https://api.fireworks.ai/v1`
  - Fireworks key: set env `FIREWORKS_API_KEY`
