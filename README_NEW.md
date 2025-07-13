# DO-TP Federated Learning Framework

A comprehensive framework for training and comparing Destination-Oriented Trajectory Prediction (DO-TP) models using both centralized and federated learning approaches.

## Project Structure

```
├── src/
│   ├── config.py                    # Configuration management
│   ├── train_centralized.py         # Centralized training script
│   ├── train_federated_new.py       # Federated training script
│   ├── compare_models.py            # Model comparison script
│   ├── core/                        # Core modules
│   │   ├── __init__.py
│   │   ├── training.py              # Training utilities
│   │   ├── data_manager.py          # Data management
│   │   ├── federated.py             # Federated learning implementation
│   │   └── evaluation.py            # Evaluation and comparison
│   ├── models/
│   │   └── DO_TP_model.py           # DO-TP model implementation
│   ├── data/
│   │   ├── loader.py                # Data loading utilities
│   │   └── trajectories.py          # Trajectory dataset implementation
│   └── utils.py                     # Utility functions
├── scripts/
│   ├── run_centralized.sh           # Centralized training script
│   ├── run_federated.sh             # Federated training script
│   ├── compare_models.sh            # Model comparison script
│   └── run_complete_experiment.sh   # Complete experiment pipeline
├── datasets/                        # Dataset directory
│   ├── eth/
│   ├── hotel/
│   ├── zara1/
│   ├── zara2/
│   └── univ/
└── requirements.txt                 # Python dependencies
```

## Features

### ✅ Modular Architecture

- **Configuration Management**: Centralized configuration with dataclasses
- **Data Management**: Flexible data loading for both centralized and federated scenarios
- **Training Core**: Reusable training utilities with early stopping and metrics tracking
- **Federated Learning**: Complete federated learning implementation with weighted aggregation
- **Evaluation Framework**: Comprehensive model evaluation and comparison tools

### ✅ Training Modes

#### Centralized Training

- Trains on all specified datasets combined
- Validates on a single designated dataset
- Supports early stopping and best model saving
- Comprehensive metrics tracking and visualization

#### Federated Training

- Each dataset acts as a federated client
- Weighted aggregation based on client data sizes
- Configurable number of global rounds and local epochs
- Support for different aggregation methods

### ✅ Evaluation and Comparison

- Side-by-side performance comparison
- Multiple evaluation metrics (ADE, FDE, Loss)
- Training efficiency analysis
- Comprehensive visualization and reporting

## Installation

1. **Clone the repository**:

```bash
git clone <repository-url>
cd FT-OP
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Verify dataset structure**:
   Ensure your datasets are properly organized in the `datasets/` directory with train/val/test splits.

## Usage

### Quick Start - Complete Experiment

Run the complete experiment pipeline (centralized + federated + comparison):

```bash
# Full experiment
chmod +x scripts/*.sh
./scripts/run_complete_experiment.sh

# Quick test (reduced epochs/rounds)
./scripts/run_complete_experiment.sh --quick
```

### Individual Training Scripts

#### 1. Centralized Training

```bash
# Basic usage
./scripts/run_centralized.sh

# Custom configuration
./scripts/run_centralized.sh \
    --train_datasets "eth hotel zara1" \
    --validation_dataset "zara2" \
    --num_epochs 100 \
    --batch_size 64 \
    --output_dir "./results/centralized"
```

**Python script directly**:

```bash
cd src
python train_centralized.py \
    --train_datasets eth hotel zara1 zara2 univ \
    --validation_dataset zara1 \
    --num_epochs 200 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --output_dir ../results/centralized
```

#### 2. Federated Training

```bash
# Basic usage
./scripts/run_federated.sh

# Custom configuration
./scripts/run_federated.sh \
    --client_datasets "eth hotel zara1 zara2 univ" \
    --validation_dataset "zara1" \
    --global_rounds 10 \
    --local_epochs 7 \
    --aggregation_method "weighted_avg"
```

**Python script directly**:

```bash
cd src
python train_federated_new.py \
    --client_datasets eth hotel zara1 zara2 univ \
    --validation_dataset zara1 \
    --global_rounds 10 \
    --local_epochs 7 \
    --batch_size 32 \
    --output_dir ../results/federated
```

#### 3. Model Comparison

```bash
# Compare models
./scripts/compare_models.sh \
    --centralized_model "./results/centralized/do_tp_centralized_best.pt" \
    --federated_model "./results/federated/do_tp_federated_final.pt" \
    --test_datasets "eth hotel zara1 zara2 univ"
```

## Configuration Options

### Model Parameters

- `enc_hidden_dim`: Encoder hidden dimension (default: 32)
- `dest_dim`: Destination prediction dimension (default: 32)
- `kl_beta`: KL divergence loss weight (default: 0.1)
- `obs_len`: Observation sequence length (default: 8)
- `pred_len`: Prediction sequence length (default: 12)

### Training Parameters

- `learning_rate`: Learning rate (default: 0.001)
- `batch_size`: Batch size (centralized: 64, federated: 32)
- `num_epochs`: Number of epochs for centralized training (default: 200)
- `early_stopping_patience`: Early stopping patience (default: 20)

### Federated Parameters

- `global_rounds`: Number of global federation rounds (default: 10)
- `local_epochs`: Local training epochs per round (default: 7)
- `clients_per_round`: Number of clients selected per round (default: 5)
- `aggregation_method`: weighted_avg or simple_avg (default: weighted_avg)

## Output Structure

```
results/
├── centralized/
│   ├── do_tp_centralized_best.pt           # Best model checkpoint
│   ├── do_tp_centralized_final.pt          # Final model
│   ├── centralized_training_config.json    # Configuration
│   ├── centralized_training_metrics.json   # Training metrics
│   ├── centralized_training_training_curves.png
│   └── logs/
├── federated/
│   ├── do_tp_federated_final.pt            # Final federated model
│   ├── do_tp_federated_round_*.pt          # Round checkpoints
│   ├── federated_training_config.json
│   ├── federated_training_metrics.json
│   ├── federated_training_progress.png
│   └── logs/
└── comparison/
    ├── comparison_summary.txt               # Text summary
    ├── detailed_comparison_results.json     # Detailed results
    ├── performance_comparison.png           # Performance plots
    ├── training_progress_comparison.png     # Training comparison
    └── logs/
```

## Key Features and Improvements

### 1. **Proper Modularity**

- Separated concerns into distinct modules
- Reusable components for training, evaluation, and data management
- Configuration-driven approach for easy experimentation

### 2. **Centralized Training**

- Uses all specified datasets for training
- Validates on a designated dataset
- Comprehensive metrics tracking (ADE, FDE, Loss)
- Early stopping to prevent overfitting
- Best model checkpointing

### 3. **Federated Learning**

- Each dataset acts as a federated client
- Weighted aggregation based on client data sizes
- Support for different aggregation strategies
- Round-by-round validation and checkpointing

### 4. **Comprehensive Evaluation**

- Side-by-side model comparison
- Performance evaluation on multiple test datasets
- Training efficiency analysis
- Rich visualization and reporting

### 5. **Experiment Management**

- Automated experiment pipelines
- Comprehensive logging and result tracking
- Easy configuration management
- Reproducible experiments

## Model Performance Metrics

The framework tracks the following metrics:

- **ADE (Average Displacement Error)**: Average L2 distance between predicted and ground truth trajectories
- **FDE (Final Displacement Error)**: L2 distance between predicted and ground truth final positions
- **Loss**: Total training loss (trajectory loss + KL divergence loss)

## Advanced Usage

### Custom Configuration

Create custom configurations by modifying the config classes:

```python
from src.config import ExperimentConfig, get_centralized_config

# Create custom config
config = get_centralized_config()
config.model.enc_hidden_dim = 64
config.training.num_epochs = 300
config.data.batch_size = 128
```

### Programmatic Usage

```python
from src.core.data_manager import create_data_manager
from src.core.training import create_model, train_model
from src.config import get_centralized_config

# Create configuration
config = get_centralized_config()

# Setup data
data_manager = create_data_manager(config)
train_datasets, train_loader = data_manager.get_centralized_train_loader()
val_dataset, val_loader = data_manager.get_validation_loader()

# Train model
model = create_model(config)
metrics = train_model(model, train_loader, val_loader, config, device)
```

## Troubleshooting

### Common Issues

1. **Dataset not found**: Ensure datasets are properly organized in the `datasets/` directory
2. **CUDA out of memory**: Reduce batch size or use CPU training
3. **Import errors**: Ensure you're running scripts from the correct directory

### Debug Mode

Enable detailed logging:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python -u src/train_centralized.py --help
```

## Contributing

1. Follow the existing code structure and naming conventions
2. Add comprehensive logging for new features
3. Update configuration classes for new parameters
4. Add tests for new functionality

## License

[Add your license information here]

## Citation

If you use this framework, please cite:

```bibtex
@article{do_tp_federated,
    title={A federated pedestrian trajectory prediction model with data privacy protection},
    author={[Authors]},
    journal={[Journal]},
    year={[Year]}
}
```
