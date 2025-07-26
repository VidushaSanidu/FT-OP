# FT-OP: Organized Results Structure

## Directory Structure

The project now uses a clean, organized directory structure for all experiment results:

```
FT-OP/
├── results/                          # All experiment results
│   ├── centralized/                  # Centralized training results
│   │   └── YYYYMMDD_HHMMSS/         # Timestamped experiment runs
│   │       ├── models/              # Saved model files (.pt)
│   │       ├── logs/                # Training logs
│   │       ├── configs/             # Configuration and metrics
│   │       └── plots/               # Training plots (if generated)
│   └── federated/                   # Federated training results (same structure)
├── src/                             # Source code (no results files)
└── scripts/                         # Utility scripts
```

## Key Features

### 🧹 Automatic Cleanup

- The `--cleanup` flag removes scattered files from previous runs
- No more `.pt` files or configs cluttering the `src/` directory
- All results are properly organized in timestamped directories

### 📁 Organized Storage

- **Models**: `results/{exp_type}/{timestamp}/models/`
- **Logs**: `results/{exp_type}/{timestamp}/logs/`
- **Configs**: `results/{exp_type}/{timestamp}/configs/`
- **Plots**: `results/{exp_type}/{timestamp}/plots/`

### 🔍 Results Viewer

Use the `view_results.py` script to navigate experiments:

```bash
# List all experiments
python3 scripts/view_results.py

# View specific experiment details
python3 scripts/view_results.py --exp_type centralized --run_id 20250726_191302
```

## Running Experiments

### Centralized Training

```bash
# With automatic cleanup and default settings
python3 src/train/train_centralized.py --cleanup --num_epochs 200

# With custom output directory
python3 src/train/train_centralized.py --output_dir results/my_experiment --num_epochs 100

# Quick test run
python3 src/train/train_centralized.py --cleanup --num_epochs 2
```

### Key Arguments

- `--cleanup`: Clean up scattered files before starting
- `--output_dir`: Custom output directory (overrides timestamp naming)
- `--num_epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--learning_rate`: Learning rate
- `--train_datasets`: List of datasets for training
- `--validation_dataset`: Dataset for validation

## Benefits

1. **Clean Codebase**: No more scattered model files and logs in `src/`
2. **Organized Results**: Each experiment run is properly timestamped and organized
3. **Easy Navigation**: Use the results viewer to explore experiments
4. **Reproducibility**: All configurations and metrics are saved with each run
5. **Scalable**: Easy to extend for federated learning and other experiment types

## Migration

If you have old scattered files, use the `--cleanup` flag to automatically clean them up:

```bash
python3 src/train/train_centralized.py --cleanup --num_epochs 1
```

This will remove any `.pt`, `.json` files and `logs/` directories from the `src/` folder.
