# SmolLM2 - Finetuned for Sentence Completeness

<div align="center">
    <a href="https://huggingface.co/spaces/ZivK/smollm2-end-of-sentence-demo"><img alt="Static Badge" src="https://img.shields.io/badge/Demo-Hugging%20Face-000000?logo=huggingface&logoColor=000&labelColor=FFD21E&style=for-the-badge"></a>
    <a href="https://huggingface.co/ZivK/smollm2-end-of-sentence"><img alt="Static Badge" src="https://img.shields.io/badge/Model-Hugging%20Face-000000?style=for-the-badge&logo=huggingface&logoColor=000&labelColor=FFD21E"></a>
</div>

---

A fine-tuned version of the [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) language model. This repository contains all the training code, dataset, model modifications, and a CLI tool (`train.py`) to help you train the model on your own data.

## Overview

**Smollm2-End-of-Sentence** is a model specifically designed to determine whether an input sentence is complete or has been cut off. Using LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning, we adapt the original [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) model with modifications to its layers and adding a classifier head, improving its ability to recognize sentence boundaries.

<u>Key features:</u>
- **Sentence Completeness Task**: Identify if a sentence is complete or truncated.
- **LoRA (PEFT) Fine-Tuning**: Efficiently adapt the model using low-rank updates.
- **Custom Dataset**: Fine-tuned on a dataset curated for this specific task.
- **CLI Support**: Easy-to-use command-line interface via `train.py` for training and evaluation.

## Repository Structure

```plaintext
smollm2-end-of-sentence/
├── data/                   # Custom data for sentence completeness task
├── notebooks/              # Notebooks used for testing
├── src/                    # Utility scripts for preprocessing, model structure, training.
├──────model.py
├──────train_inner.py
├──────dataset.py
├── config.py               # Configuration file
├── train.py                # CLI script for training the model
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

* **data/**: Contains the training and evaluation datasets.
* **scripts/**: Includes model configurations, additional scripts for data preprocessing and utility functions.
* **train.py**: The main CLI script to kick off training and fine-tuning.

## Getting Started

### Prerequisites

* PyTorch

* Transformers

* PyTorch Lightning

* PEFT

* Additional packages listed in requirements.txt

### Installation

1. Clone the Repository
```bash
git clone https://github.com/your-username/smollm2-end-of-sentence.git
cd smollm2-end-of-sentence
```
2. Create and Activate a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```
3. Install Dependencies 
```bash
pip install -r requirements.txt
```

## Usage
### Training the Model
The repository includes a CLI tool to train the model. Use the following command:
```bash
python train.py --mode token --device cuda --epochs 2 --batch-size 24 --lr 0.0003
```
### Configuration File
A sample configuration file (`config.py`) is provided. This file contains key settings like:

```markdown
- Batch size
- Learning rate
- Number of epochs
- Base checkpoint
- Training checkpoints (resume training)
- Training device
```
