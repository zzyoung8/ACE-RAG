#!/bin/bash

# Run contrastive learning pipeline with default settings
# Usage: ./run_contrastive.sh [dataset] [mode]
# dataset: zh, zh_google, en, en_google (default: zh)
# mode: full, train, retrieve, evaluate (default: full)

# Default values
DATASET=${1:-"zh"}
MODE=${2:-"full"}

# Set up directories
mkdir -p contrastive/saved_models
mkdir -p contrastive/output

# Base command - 显式指定方法为zzy
BASE_CMD="python contrastive/main.py --data_path data/${DATASET}.json --dataset ${DATASET} --method zzy"

# Run appropriate mode
case $MODE in
  "full")
    echo "Running full pipeline for dataset: ${DATASET}"
    $BASE_CMD --train --retrieve --evaluate
    ;;
  "train")
    echo "Training contrastive model for dataset: ${DATASET}"
    $BASE_CMD --train
    ;;
  "retrieve")
    echo "Retrieving documents for dataset: ${DATASET}"
    $BASE_CMD --retrieve
    ;;
  "evaluate")
    echo "Evaluating with retrieved documents for dataset: ${DATASET}"
    $BASE_CMD --evaluate
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo "Usage: ./run_contrastive.sh [dataset] [mode]"
    echo "dataset: zh, zh_google, en, en_google (default: zh)"
    echo "mode: full, train, retrieve, evaluate (default: full)"
    exit 1
    ;;
esac

echo "Done!" 