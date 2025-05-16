# Contrastive Learning for RAG

This directory contains implementation for contrastive learning in Retrieval-Augmented Generation (RAG) systems. The goal is to improve document retrieval by training a model that can better distinguish between relevant and irrelevant documents for a given query.

## Features

- Contrastive learning model that learns to encode queries and documents in the same semantic space
- Training with positive (containing the answer) and negative (not containing the answer) examples
- Integration with existing RAG pipeline for evaluation
- Support for both Chinese and English datasets

## Directory Structure

```
contrastive/
├── model.py           # Contrastive learning model implementation
├── train.py           # Training script for contrastive learning
├── retrieval.py       # Document retrieval using the trained model
├── main.py            # Main script to run the entire pipeline
└── README.md          # This file
```

## Requirements

- PyTorch
- Sentence Transformers
- FAISS
- tqdm

## Usage

### Full Pipeline

To run the entire pipeline (train, retrieve, evaluate):

```bash
python contrastive/main.py --data_path data/zh.json --dataset zh
```

### Training Only

To train the contrastive learning model:

```bash
python contrastive/main.py --train --data_path data/zh.json
```

### Retrieval Only

To use a trained model for document retrieval:

```bash
python contrastive/main.py --retrieve --data_path data/zh.json
```

### Evaluation Only

To evaluate the RAG system with retrieved documents:

```bash
python contrastive/main.py --evaluate --dataset zh --modelname Qwen2.5 --method TA_ARE
```

## Command Line Arguments

### Pipeline Control
- `--train`: Train the contrastive learning model
- `--retrieve`: Use the trained model for retrieval
- `--evaluate`: Evaluate using the retrieved documents

### Data Parameters
- `--data_path`: Path to the data file (default: "data/zh.json")
- `--dataset`: Dataset name for evaluation (default: "zh")
- `--train_ratio`: Ratio of data to use for training (default: 0.8)

### Model Parameters
- `--base_model`: Base model for sentence embeddings (default: "model/paraphrase-multilingual-MiniLM-L12-v2")
- `--embedding_dim`: Dimension of base embeddings (default: 384)
- `--projection_dim`: Dimension of projected embeddings (default: 128)
- `--temperature`: Temperature for contrastive loss (default: 0.07)

### Training Parameters
- `--num_epochs`: Number of training epochs (default: 5)
- `--batch_size`: Batch size for training (default: 16)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--weight_decay`: Weight decay (default: 0.01)

### Retrieval Parameters
- `--top_k`: Number of documents to retrieve (default: 15)

### Evaluation Parameters
- `--modelname`: Model name for evaluation (default: "Qwen2.5")
- `--plm`: Path to the model (default: "./model/Qwen2.5-7B-Instruct")
- `--method`: Method for evaluation (default: "TA_ARE")
- `--generation_temperature`: Temperature for generation (default: 0.2)

### Output Parameters
- `--model_dir`: Directory to save the model (default: "contrastive/saved_models")
- `--output_dir`: Directory to save the output (default: "contrastive/output")

## How It Works

1. **Training Phase**: The model learns to encode queries and documents in a shared semantic space where relevant query-document pairs are closer together and irrelevant pairs are farther apart.

2. **Retrieval Phase**: The trained model encodes all documents in the dataset and builds a FAISS index. For each query, the model retrieves the most similar documents based on the learned representations.

3. **Evaluation Phase**: The retrieved documents are used in place of the original documents in the RAG pipeline to evaluate the effectiveness of the contrastive learning approach.

## Implementation Details

### Contrastive Learning Model

The core of the implementation is a contrastive learning model that uses SentenceTransformer as a base encoder and adds projection layers for queries and documents. The model is trained with a contrastive loss function that encourages relevant query-document pairs to have similar representations while pushing irrelevant pairs apart.

### Data Processing

The implementation automatically identifies positive documents (containing the answer) and negative documents (not containing the answer) for each query. This allows the model to learn meaningful distinctions between relevant and irrelevant documents.

### Integration with RAG

The trained model can be seamlessly integrated with the existing RAG pipeline, allowing for direct comparison of retrieval quality between the baseline and contrastive learning approaches. 