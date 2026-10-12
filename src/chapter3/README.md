# Chapter 3: Baseline Classifiers

Creates the fixed data splits, report embeddings, baseline classifiers, metrics, and figures.

## Files

- `common.py`: Loads the Chapter 3 configuration and contains shared dataset, embedding, and MLP helpers.
- `create_splits.py`: Creates the train, validation, test, and adversarial splits.
- `embed_reports.py`: Embeds the reports in parallel and merges the embedding shards.
- `train_classifiers.py`: Trains and selects the baseline classifiers using validation performance.
- `evaluate_classifiers.py`: Evaluates the selected models on the test split and creates the result files and figures.
- `run_all.sh`: Runs the complete Chapter 3 pipeline with the fixed two-GPU setup.

## Required Arguments

- `embed_reports.py`
  - `command`: `prepare`, `shard`, or `merge`.
  - `--index`: shard number.
  - `--device`: CUDA device.
