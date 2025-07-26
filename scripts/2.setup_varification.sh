#!/bin/bash

# Setup and Validation Script
# This script validates the environment and datasets

echo "DO-TP Framework Setup and Validation"
echo "==================================="

# Check Python and dependencies
echo "1. Checking Python environment..."
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python 3 not found. Please install Python 3."
    exit 1
fi

echo "2. Checking Python dependencies..."
cd "$(dirname "$0")"
cd ../
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies."
    exit 1
fi

echo "3. Validating dataset structure..."
DATASETS_DIR="./datasets"
REQUIRED_DATASETS=("eth" "hotel" "zara1" "zara2" "univ")
REQUIRED_SPLITS=("train" "val" "test")

missing_datasets=()
for dataset in "${REQUIRED_DATASETS[@]}"; do
    dataset_path="$DATASETS_DIR/$dataset"
    if [ ! -d "$dataset_path" ]; then
        missing_datasets+=("$dataset")
        continue
    fi
    
    for split in "${REQUIRED_SPLITS[@]}"; do
        split_path="$dataset_path/$split"
        if [ ! -d "$split_path" ]; then
            echo "Warning: Missing split $split for dataset $dataset"
        else
            # Check if there are any .txt files in the split directory
            if [ -z "$(ls -A $split_path/*.txt 2>/dev/null)" ]; then
                echo "Warning: No .txt files found in $split_path"
            fi
        fi
    done
done

if [ ${#missing_datasets[@]} -ne 0 ]; then
    echo "Error: Missing datasets: ${missing_datasets[*]}"
    echo "Please ensure all required datasets are in the datasets/ directory."
    exit 1
fi

echo "4. Validating Python imports..."
cd "$(dirname "$0")/../src"
python3 -c "
import sys
sys.path.append('.')
try:
    from config import ExperimentConfig
    from core.data_manager import create_data_manager
    from core.training import create_model
    from models.DO_TP_model import DO_TP
    print('✓ All imports successful')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "Error: Python import validation failed."
    exit 1
fi

cd "$(dirname "$0")/../"

echo "5. Creating necessary directories..."
mkdir -p results/centralized
mkdir -p results/federated
mkdir -p results/comparison
mkdir -p logs

echo "6. Making scripts executable..."
chmod +x scripts/*.sh

echo ""
echo "✓ Setup and validation completed successfully!"
echo ""
echo "Next steps:"
echo "1. Run quick test:"
echo "   ./scripts/run_complete_experiment.sh --quick"
echo ""
echo "2. Run full experiment:"
echo "   ./scripts/run_complete_experiment.sh"
echo ""
echo "3. Run individual components:"
echo "   ./scripts/run_centralized.sh --help"
echo "   ./scripts/run_federated.sh --help"
echo "   ./scripts/compare_models.sh --help"
