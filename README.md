# ACE-RAG: Automated Context Engineering with Spatio-Temporal and Contrastive Optimization for Retrieval-Augmented Generation

This project implements a question answering system that combines Contrastive Learning and Retrieval-Augmented Generation (RAG) to enhance the accuracy of document retrieval and answer generation. The system supports both Chinese and English data and includes a complete workflow encompassing data acquisition, contrastive learning training, vector retrieval, RAG generation, and evaluation.

## Directory Structure

```
├── contrastive/
│   ├── main.py
│   ├── model.py
│   ├── train.py
│   ├── retrieval.py
│   ├── README.md
│   └── ...
├── RAG.py
├── vector_retrieval.py
├── models.py
├── utils.py
├── get_data.py
├── get_data_bbc.py
├── google_search.py
├── run_contrastive.sh
└── instruction.yaml
```

## Main Features

  - **Data Acquisition and Processing**: Supports collecting Chinese and English question-answering data from various channels such as news websites and Google Search, and automatically generates positive and negative samples.
  - **Contrastive Learning Training**: Trains a retrieval model using contrastive learning methods to encode queries and documents into the same semantic space, improving relevance discrimination.
  - **Vector Retrieval and BM25 Retrieval**: Supports vector retrieval based on SentenceTransformer and BM25 text retrieval, adaptable for both Chinese and English.
  - **RAG Generation and Evaluation**: Integrates various large language models (LLMs) such as Qwen, ChatGLM, Baichuan, etc., supports multiple Prompt templates, and automatically evaluates the accuracy metrics of generated results.

## Installation Dependencies

Please ensure the following main dependencies are installed (some dependencies may need to be added based on the specific models and environment):

  - Python 3.8+
  - torch
  - sentence-transformers
  - faiss
  - transformers
  - tqdm
  - spacy
  - jieba
  - nltk
  - requests
  - beautifulsoup4
  - pyyaml

Installation example:

```bash
pip install torch sentence-transformers faiss-cpu transformers tqdm spacy jieba nltk requests beautifulsoup4 pyyaml
python -m spacy download zh_core_web_sm
```

## Quick Start

### 1\. Data Acquisition and Processing

  - Run `get_data.py` or `get_data_bbc.py` to automatically crawl news web pages and generate question-answering data.
  - Use `utils.py` for data preprocessing such as positive and negative sample splitting and format conversion.

### 2\. Contrastive Learning Training and Retrieval

  - Train the contrastive learning model:

    ```bash
    python contrastive/main.py --train --data_path data/zh.json
    ```

  - Perform retrieval using the trained model:

    ```bash
    python contrastive/main.py --retrieve --data_path data/zh.json
    ```

### 3\. RAG Generation and Evaluation

  - Evaluate the performance of retrieval-augmented generation:

    ```bash
    python contrastive/main.py --evaluate --dataset zh --modelname Qwen2.5 --method TA_ARE
    ```

### 4\. Other Scripts

  - `vector_retrieval.py`: Independent implementation of vector retrieval and BM25 retrieval, which can be used for comparative experiments of retrieval algorithms.
  - `RAG.py`: RAG main process and evaluation script supporting various large models. See the script for detailed parameter descriptions.

## Parameter Description

The main parameters are as follows. For more parameters and default values, please refer to `contrastive/README.md` or the help information within each script:

  - `--train` / `--retrieve` / `--evaluate`: Controls the workflow stage
  - `--data_path`: Path to the data file
  - `--dataset`: Name of the evaluation dataset
  - `--base_model`: Path to the sentence embedding base model
  - `--modelname`: Name of the large model used for evaluation
  - `--method`: Evaluation method (Prompt template)
  - `--output_dir` / `--model_dir`: Output and model saving directories