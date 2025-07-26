#!/bin/bash

# Model Comparison Script
# This script compares centralized and federated models

echo "Comparing Centralized vs Federated Models"
echo "========================================"

# Default parameters
CENTRALIZED_MODEL="../src/results/centralized/do_tp_centralized_best.pt"
FEDERATED_MODEL="../src/do_tp_federated_federated_final.pt"
TEST_DATASETS="eth hotel zara1 zara2 univ"
VALIDATION_DATASET="zara1"
OUTPUT_DIR="./results/comparison"
EXPERIMENT_NAME="centralized_vs_federated"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --centralized_model)
            CENTRALIZED_MODEL="$2"
            shift 2
            ;;
        --federated_model)
            FEDERATED_MODEL="$2"
            shift 2
            ;;
        --test_datasets)
            TEST_DATASETS="$2"
            shift 2
            ;;
        --validation_dataset)
            VALIDATION_DATASET="$2"
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
            echo "  --centralized_model    Path to centralized model checkpoint"
            echo "  --federated_model      Path to federated model checkpoint"
            echo "  --test_datasets        Space-separated list of test datasets (default: 'eth hotel zara1 zara2 univ')"
            echo "  --validation_dataset   Primary validation dataset (default: 'zara1')"
            echo "  --output_dir           Output directory (default: './results/comparison')"
            echo "  --experiment_name      Experiment name (default: 'centralized_vs_federated')"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if model files exist (relative to src directory where the script will run)
SCRIPT_DIR="$(dirname "$0")"
SRC_DIR="$SCRIPT_DIR/../src"

# Convert paths to be relative to src directory
if [[ "$CENTRALIZED_MODEL" =~ ^\.\./src/ ]]; then
    CENTRALIZED_CHECK="${CENTRALIZED_MODEL#../src/}"
else
    CENTRALIZED_CHECK="$CENTRALIZED_MODEL"
fi

if [[ "$FEDERATED_MODEL" =~ ^\.\./src/ ]]; then
    FEDERATED_CHECK="${FEDERATED_MODEL#../src/}"
else
    FEDERATED_CHECK="$FEDERATED_MODEL"
fi

if [ ! -f "$SRC_DIR/$CENTRALIZED_CHECK" ]; then
    echo "Error: Centralized model not found: $SRC_DIR/$CENTRALIZED_CHECK"
    echo "Please run centralized training first or specify correct path."
    exit 1
fi

if [ ! -f "$SRC_DIR/$FEDERATED_CHECK" ]; then
    echo "Error: Federated model not found: $SRC_DIR/$FEDERATED_CHECK"
    echo "Please run federated training first or specify correct path."
    exit 1
fi

# Convert to absolute path if relative
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$(dirname "$0")/../$OUTPUT_DIR"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Change to src directory (from scripts/ to src/)
cd "$(dirname "$0")/../src"

echo "Configuration:"
echo "  Centralized model: $CENTRALIZED_MODEL"
echo "  Federated model: $FEDERATED_MODEL"
echo "  Test datasets: $TEST_DATASETS"
echo "  Validation dataset: $VALIDATION_DATASET"
echo "  Output directory: $OUTPUT_DIR"
echo "  Experiment name: $EXPERIMENT_NAME"
echo ""

# Activate virtual environment if it exists
if [ -f "../.venv/bin/activate" ]; then
    source "../.venv/bin/activate"
    echo "Activated virtual environment"
fi

# Run comparison
python compare_models.py \
    --centralized_model "$CENTRALIZED_MODEL" \
    --federated_model "$FEDERATED_MODEL" \
    --test_datasets $TEST_DATASETS \
    --validation_dataset "$VALIDATION_DATASET" \
    --output_dir "$(realpath "$OUTPUT_DIR")" \
    --experiment_name "$EXPERIMENT_NAME" \
    --save_plots \
    --detailed_analysis \
    --use_gpu

echo ""
echo "Model comparison completed!"
echo "Results saved to: $OUTPUT_DIR"
