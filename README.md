# Tell Me Why: Eliciting explanations from AI models

This repository contains the code used in the experiments for the thesis
"Tell Me Why: Eliciting explanations from AI models". Code is grouped
up by relevant chapter.

## Structure

    configs/
      chapter3.json
      chapter4.json
    src/
      chapter3/
      chapter4/
      chapter5/
      chapter6/
      chapter7/
      chapter8/
    requirements.txt


## Environment

Python 3.13.12 was used via micromamba virtual env - `requirements.txt` contains output of `pip freeze` from the environment used for the experiments. 

The initial machine on which the experiments were run featured two NVIDIA GeForce RTX 5090 GPUs with 32 GB VRAM each, Intel Xeon Gold 5418N CPU, and 503 GiB RAM. The code in this repository expects both access to sufficient VRAM and a dual-GPU setup. The environment and training scripts expect Blackwell architecture GPUs.

## External artifacts

The scripts expect paths supplied through configuration files or command-line
arguments:

- the original goodware and malware behavioral JSON reports: https://www.kaggle.com/datasets/greimas/malware-and-goodware-dynamic-analysis-reports
- Ollama inference server running with `qwen3.5:35b`, `qwen3:8b`, and `qwen3.6:27b` downloaded
- AutoRobust sum.json and corpus.json for benign behavior references
- Fireworks API key
