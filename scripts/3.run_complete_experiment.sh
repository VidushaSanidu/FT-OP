#!/bin/bash

# Complete Experiment Pipeline
# This script runs the complete experiment: centralized training, federated training, and comparison

echo "Running Complete DO-TP Experiment Pipeline"
echo "=========================================="

# Default parameters
TRAIN_DATASETS="eth hotel zara1 zara2 univ"
VALIDATION_DATASET="zara1"
BASE_OUTPUT_DIR="./results"
EXPERIMENT_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

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
        --base_output_dir)
            BASE_OUTPUT_DIR="$2"
            shift 2
            ;;
        --quick)
            # Quick mode: reduced epochs/rounds for testing
            QUICK_MODE=true
            shift 1
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --train_datasets       Space-separated list of training datasets (default: 'eth hotel zara1 zara2 univ')"
            echo "  --validation_dataset   Validation dataset (default: 'zara1')"
            echo "  --base_output_dir      Base output directory (default: './results')"
            echo "  --quick                Quick mode with reduced epochs/rounds for testing"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set experiment-specific directories
CENTRALIZED_DIR="$BASE_OUTPUT_DIR/experiment_$EXPERIMENT_TIMESTAMP/centralized"
FEDERATED_DIR="$BASE_OUTPUT_DIR/experiment_$EXPERIMENT_TIMESTAMP/federated"
COMPARISON_DIR="$BASE_OUTPUT_DIR/experiment_$EXPERIMENT_TIMESTAMP/comparison"

# Create directories
mkdir -p "$CENTRALIZED_DIR"
mkdir -p "$FEDERATED_DIR"
mkdir -p "$COMPARISON_DIR"

echo "Experiment Configuration:"
echo "  Training datasets: $TRAIN_DATASETS"
echo "  Validation dataset: $VALIDATION_DATASET"
echo "  Experiment timestamp: $EXPERIMENT_TIMESTAMP"
echo "  Centralized output: $CENTRALIZED_DIR"
echo "  Federated output: $FEDERATED_DIR"
echo "  Comparison output: $COMPARISON_DIR"

if [ "$QUICK_MODE" = true ]; then
    echo "  Mode: Quick (reduced epochs/rounds)"
    CENTRALIZED_EPOCHS=20
    FEDERATED_ROUNDS=3
else
    echo "  Mode: Full experiment"
    CENTRALIZED_EPOCHS=200
    FEDERATED_ROUNDS=10
fi

echo ""

# Change to script directory
SCRIPT_DIR="$(dirname "$0")"

echo "Step 1/3: Running Centralized Training"
echo "======================================"
"$SCRIPT_DIR/4.1.run_centralized.sh" \
    --train_datasets "$TRAIN_DATASETS" \
    --validation_dataset "$VALIDATION_DATASET" \
    --num_epochs "$CENTRALIZED_EPOCHS" \
    --output_dir "$CENTRALIZED_DIR" \
    --experiment_name "centralized_training_$EXPERIMENT_TIMESTAMP"

if [ $? -ne 0 ]; then
    echo "Error: Centralized training failed!"
    exit 1
fi

echo ""
echo "Step 2/3: Running Federated Training"
echo "===================================="
"$SCRIPT_DIR/4.2.run_federated.sh" \
    --client_datasets "$TRAIN_DATASETS" \
    --validation_dataset "$VALIDATION_DATASET" \
    --global_rounds "$FEDERATED_ROUNDS" \
    --output_dir "$FEDERATED_DIR" \
    --experiment_name "federated_training_$EXPERIMENT_TIMESTAMP"

if [ $? -ne 0 ]; then
    echo "Error: Federated training failed!"
    exit 1
fi

echo ""
echo "Step 3/3: Running Model Comparison"
echo "=================================="
"$SCRIPT_DIR/compare_models.sh" \
    --centralized_model "$CENTRALIZED_DIR/do_tp_centralized_best.pt" \
    --federated_model "$FEDERATED_DIR/do_tp_federated_final.pt" \
    --test_datasets "$TRAIN_DATASETS" \
    --validation_dataset "$VALIDATION_DATASET" \
    --output_dir "$COMPARISON_DIR" \
    --experiment_name "comparison_$EXPERIMENT_TIMESTAMP"

if [ $? -ne 0 ]; then
    echo "Error: Model comparison failed!"
    exit 1
fi

echo ""
echo "Complete Experiment Pipeline Finished!"
echo "======================================"
echo "Results saved to: $BASE_OUTPUT_DIR/experiment_$EXPERIMENT_TIMESTAMP/"
echo ""
echo "Summary:"
echo "  - Centralized results: $CENTRALIZED_DIR"
echo "  - Federated results: $FEDERATED_DIR"
echo "  - Comparison results: $COMPARISON_DIR"
echo ""
echo "Key files to check:"
echo "  - Comparison summary: $COMPARISON_DIR/comparison_summary.txt"
echo "  - Performance plots: $COMPARISON_DIR/performance_comparison.png"
echo "  - Detailed results: $COMPARISON_DIR/detailed_comparison_results.json"
