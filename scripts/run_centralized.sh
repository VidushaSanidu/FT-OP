#!/bin/bash

# Centralized Training Script
# This script runs centralized training using all datasets

echo "Starting Centralized Training for DO-TP Model"
echo "============================================"

# Default parameters
TRAIN_DATASETS="eth hotel zara1 zara2 univ"
VALIDATION_DATASET="zara1"
NUM_EPOCHS=200
BATCH_SIZE=64
LEARNING_RATE=0.001
OUTPUT_DIR="./results/centralized"
EXPERIMENT_NAME="centralized_training"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train_datasets)
            TRAIN_DATASETS="$2"
            shift 2
            ;;
        --validation_dataset)
            VALIDATION_DATASET="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --experiment_name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --train_datasets       Space-separated list of training datasets (default: 'eth hotel zara1 zara2 univ')"
            echo "  --validation_dataset   Validation dataset (default: 'zara1')"
            echo "  --num_epochs          Number of training epochs (default: 200)"
            echo "  --batch_size          Batch size (default: 64)"
            echo "  --learning_rate       Learning rate (default: 0.001)"
            echo "  --output_dir          Output directory (default: './results/centralized')"
            echo "  --experiment_name     Experiment name (default: 'centralized_training')"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Change to src directory (from scripts/ to src/)
cd "$(dirname "$0")/../src"

echo "Configuration:"
echo "  Training datasets: $TRAIN_DATASETS"
echo "  Validation dataset: $VALIDATION_DATASET"
echo "  Number of epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Experiment name: $EXPERIMENT_NAME"
echo ""

# Activate virtual environment if it exists
if [ -f "../.venv/bin/activate" ]; then
    source "../.venv/bin/activate"
    echo "Activated virtual environment"
fi

# Run centralized training
python train_centralized.py \
    --train_datasets $TRAIN_DATASETS \
    --validation_dataset "$VALIDATION_DATASET" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name "$EXPERIMENT_NAME" \
    --use_gpu

echo ""
echo "Centralized training completed!"
echo "Results saved to: $OUTPUT_DIR"
