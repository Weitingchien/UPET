# MNLI RoBERTa Prompt P-tuning Pipeline

This directory contains all the files needed to run the MNLI (Multi-Genre Natural Language Inference) task using RoBERTa with prompt p-tuning, specifically organized from the main UPET repository.

## Main Script

- `run_script_fewshot/run_mnli_roberta.sh` - The main script that executes MNLI training with RoBERTa and prompt p-tuning

## Core Files

- `run.py` - Main entry point for training
- `arguments.py` - Command line argument parsing and configuration
- `requirements.txt` - Python package dependencies

## Task Implementation

- `tasks/utils.py` - Task definitions and utilities
- `tasks/glue/dataset.py` - GLUE dataset handling for MNLI
- `tasks/glue/get_trainer.py` - GLUE trainer setup and configuration

## Model Implementation

- `model/utils.py` - Model utilities and factory functions
- `model/prompt_for_sequence_classification.py` - Prompt-based sequence classification models
- `model/prefix_encoder.py` - Prefix encoder for prompt p-tuning
- `model/parameter_freeze.py` - Parameter freezing utilities
- `model/loss.py` - Loss functions
- `model/head_for_sequence_classification.py` - Classification heads
- `model/model_adaptation.py` - Model adaptation utilities
- `model/deberta.py` - DeBERTa model components
- `model/debertaV2.py` - DeBERTa V2 model components
- `model/head_for_question_answering.py` - Question answering heads
- `model/head_for_token_classification.py` - Token classification heads
- `model/multiple_choice.py` - Multiple choice model components

## Training Components

- `training/trainer_base.py` - Base trainer class
- `training/self_trainer.py` - Self-training implementation
- `training/sampler.py` - Data sampling utilities

## Usage

To run the MNLI task with RoBERTa and prompt p-tuning:

```bash
cd mnli_pipeline
bash run_script_fewshot/run_mnli_roberta.sh
```

## Configuration

The script is configured for:
- Task: GLUE MNLI (Multi-Genre Natural Language Inference)
- Model: RoBERTa-large
- Method: Prompt p-tuning with positional encoding
- Few-shot: 16 examples per label
- Batch size: 4
- Learning rate: 5e-5
- Epochs: 100
- Sequence length: 128

## Notes

This pipeline contains only the files necessary for running the MNLI RoBERTa prompt p-tuning experiment. All external dependencies (transformers, torch, etc.) need to be installed separately via the requirements.txt file.