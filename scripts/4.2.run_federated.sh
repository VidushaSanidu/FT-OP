#!/bin/bash

# Federated Training Script
# This script runs federated training using multiple datasets as clients

echo "Starting Federated Training for DO-TP Model"
echo "==========================================="

# Default parameters
CLIENT_DATASETS="eth hotel zara1 zara2 univ"
VALIDATION_DATASET="zara1"
GLOBAL_ROUNDS=10
LOCAL_EPOCHS=7
CLIENTS_PER_ROUND=5
BATCH_SIZE=16
LEARNING_RATE=0.001
OUTPUT_DIR="./results/federated"
EXPERIMENT_NAME="federated_training"
AGGREGATION_METHOD="weighted_avg"
CLEANUP=false
NO_TIMESTAMP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --client_datasets)
            CLIENT_DATASETS="$2"
            shift 2
            ;;
        --validation_dataset)
            VALIDATION_DATASET="$2"
            shift 2
            ;;
        --global_rounds)
            GLOBAL_ROUNDS="$2"
            shift 2
            ;;
        --local_epochs)
            LOCAL_EPOCHS="$2"
            shift 2
            ;;
        --clients_per_round)
            CLIENTS_PER_ROUND="$2"
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
        --aggregation_method)
            AGGREGATION_METHOD="$2"
            shift 2
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --no_timestamp)
            NO_TIMESTAMP=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --client_datasets      Space-separated list of client datasets (default: 'eth hotel zara1 zara2 univ')"
            echo "  --validation_dataset   Validation dataset (default: 'zara1')"
            echo "  --global_rounds        Number of global rounds (default: 10)"
            echo "  --local_epochs         Number of local epochs per round (default: 7)"
            echo "  --clients_per_round    Number of clients per round (default: 5)"
            echo "  --batch_size           Batch size (default: 32)"
            echo "  --learning_rate        Learning rate (default: 0.001)"
            echo "  --output_dir           Output directory (default: './results/federated')"
            echo "  --experiment_name      Experiment name (default: 'federated_training')"
            echo "  --aggregation_method   Aggregation method: weighted_avg or simple_avg (default: 'weighted_avg')"
            echo "  --cleanup              Clean up scattered files from previous runs before starting"
            echo "  --no_timestamp         Don't create timestamped subdirectory (for use with complete experiment)"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Convert to absolute path if relative
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$(dirname "$0")/../$OUTPUT_DIR"
fi

# Create timestamped subdirectory for this run (only if not called from complete experiment)
if [ "$NO_TIMESTAMP" = true ]; then
    FINAL_OUTPUT_DIR="$OUTPUT_DIR"
else
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    FINAL_OUTPUT_DIR="$OUTPUT_DIR/$TIMESTAMP"
fi

# Create output directory
mkdir -p "$FINAL_OUTPUT_DIR"

# Change to src directory (from scripts/ to src/)
cd "$(dirname "$0")/../src"

echo "Configuration:"
echo "  Client datasets: $CLIENT_DATASETS"
echo "  Validation dataset: $VALIDATION_DATASET"
echo "  Global rounds: $GLOBAL_ROUNDS"
echo "  Local epochs: $LOCAL_EPOCHS"
echo "  Clients per round: $CLIENTS_PER_ROUND"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Output directory: $FINAL_OUTPUT_DIR"
echo "  Experiment name: $EXPERIMENT_NAME"
echo "  Aggregation method: $AGGREGATION_METHOD"
echo "  Cleanup old files: $CLEANUP"
echo ""

# Activate virtual environment if it exists
if [ -f "../.venv/bin/activate" ]; then
    source "../.venv/bin/activate"
    echo "Activated virtual environment"
fi

# Run federated training
CLEANUP_FLAG=""
if [ "$CLEANUP" = true ]; then
    CLEANUP_FLAG="--cleanup"
fi

python train/train_federated.py \
    --client_datasets $CLIENT_DATASETS \
    --validation_dataset "$VALIDATION_DATASET" \
    --global_rounds "$GLOBAL_ROUNDS" \
    --local_epochs "$LOCAL_EPOCHS" \
    --clients_per_round "$CLIENTS_PER_ROUND" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --output_dir "$(realpath "$FINAL_OUTPUT_DIR")" \
    --experiment_name "$EXPERIMENT_NAME" \
    --aggregation_method "$AGGREGATION_METHOD" \
    $CLEANUP_FLAG \
    --use_gpu

echo ""
echo "Federated training completed!"
echo "Results saved to: $FINAL_OUTPUT_DIR"
