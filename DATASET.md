# Dataset Documentation

## Overview

This repository contains pedestrian trajectory prediction datasets used for training and evaluating federated learning models. The datasets are derived from popular trajectory prediction benchmarks and are organized for both centralized and federated learning scenarios.

## Dataset Structure

The datasets are organized in the following hierarchy:

```
datasets/
├── raw/                    # Original dataset files
│   ├── all_data/          # Complete dataset files
│   ├── train/             # Training splits
│   └── val/               # Validation splits
├── eth/                   # ETH dataset splits
│   ├── train/
│   ├── val/
│   └── test/
├── hotel/                 # Hotel dataset splits
│   ├── train/
│   ├── val/
│   └── test/
├── univ/                  # University dataset splits
│   ├── train/
│   ├── val/
│   └── test/
├── zara1/                 # Zara1 dataset splits
│   ├── train/
│   ├── val/
│   └── test/
└── zara2/                 # Zara2 dataset splits
    ├── train/
    ├── val/
    └── test/
```

## Dataset Sources

The datasets are based on the following well-known pedestrian trajectory datasets:

### 1. ETH Dataset (biwi_eth)

- **Source**: ETH Zurich
- **Environment**: University campus
- **Total samples**: 5,492 trajectory points
- **Description**: Pedestrian trajectories recorded in outdoor university settings

### 2. Hotel Dataset (biwi_hotel)

- **Source**: ETH Zurich
- **Environment**: Hotel lobby
- **Total samples**: 6,543 trajectory points
- **Description**: Indoor pedestrian movements in a hotel environment

### 3. University Dataset (uni_examples)

- **Source**: University settings
- **Environment**: Academic campus
- **Total samples**: 2,747 trajectory points
- **Description**: Student and visitor movements in university areas

### 4. Zara Datasets

- **Zara01**: 5,153 trajectory points
- **Zara02**: 9,722 trajectory points
- **Zara03**: 5,005 trajectory points
- **Environment**: Shopping mall
- **Description**: Crowded pedestrian scenarios in retail environments

### 5. Students Datasets

- **Students001**: 21,813 trajectory points
- **Students003**: 17,953 trajectory points
- **Environment**: University campus
- **Description**: Large-scale student movement patterns

## Data Format

### File Format

- **File extension**: `.txt`
- **Delimiter**: Tab-separated values (`\t`)
- **Encoding**: UTF-8

### Data Schema

Each line in the dataset files contains the following fields:

```
<frame_id> <ped_id> <x> <y>
```

Where:

- `frame_id`: Timestamp/frame number (float)
- `ped_id`: Unique pedestrian identifier (float)
- `x`: X-coordinate position (float)
- `y`: Y-coordinate position (float)

### Example Data Points

```
780    1.0    8.46     3.59
790    1.0    9.57     3.79
800    1.0    10.67    3.99
800    2.0    13.64    5.8
810    1.0    11.73    4.32
```

## Dataset Statistics

### Overall Statistics

- **Total trajectory points**: 520,996
- **Number of datasets**: 8 distinct environments
- **File format**: Tab-separated text files
- **Coordinate system**: 2D Cartesian coordinates

### Split Distribution

#### Training Sets

| Dataset             | Training Samples |
| ------------------- | ---------------- |
| biwi_hotel_train    | 4,946            |
| biwi_eth_train      | 3,666            |
| crowds_zara01_train | 4,307            |
| crowds_zara02_train | 7,621            |
| crowds_zara03_train | 3,708            |
| students001_train   | 18,353           |
| students003_train   | 15,641           |
| uni_examples_train  | 2,266            |

#### Validation Sets

| Dataset           | Validation Samples |
| ----------------- | ------------------ |
| biwi_hotel_val    | 1,597              |
| biwi_eth_val      | 1,826              |
| crowds_zara01_val | 846                |
| crowds_zara02_val | 2,101              |
| crowds_zara03_val | 1,297              |
| students001_val   | 3,460              |
| students003_val   | 2,312              |
| uni_examples_val  | 481                |

#### Test Sets

| Dataset       | Test Samples |
| ------------- | ------------ |
| biwi_eth      | 5,492        |
| biwi_hotel    | 6,543        |
| crowds_zara01 | 5,153        |
| crowds_zara02 | 9,722        |
| students001   | 21,813       |
| students003   | 17,953       |

## Dataset Configuration

### Default Parameters

- **Observation length (`obs_len`)**: 8 time-steps
- **Prediction length (`pred_len`)**: 12 time-steps
- **Total sequence length**: 20 time-steps (8 + 12)
- **Skip frames**: 1 (no frame skipping)
- **Minimum pedestrians per sequence**: 1
- **Non-linearity threshold**: 0.002

### Data Loading Parameters

- **Delimiter**: Tab (`\t`)
- **Batch size**: Configurable
- **Shuffle**: Enabled for training
- **Number of workers**: Configurable

## Federated Learning Organization

The datasets are organized to support federated learning scenarios where each environment (ETH, Hotel, University, Zara1, Zara2) represents a different federated client:

### Client Distribution

- **Client 1 (ETH)**: Uses ETH test set, excludes ETH data from training
- **Client 2 (Hotel)**: Uses Hotel test set, excludes Hotel data from training
- **Client 3 (University)**: Uses University test sets
- **Client 4 (Zara1)**: Uses Zara1 test sets
- **Client 5 (Zara2)**: Uses Zara2 test sets

Each client trains on data from other environments to simulate the federated learning scenario where local test data is not available during training.

## Data Preprocessing

### Coordinate System

- **Pre-processed to world coordinates**: All datasets are already converted to world coordinates (meters) during preprocessing
- **No pixel-to-world transformation needed**: The dataset files contain real-world coordinates, not pixel coordinates
- **Direct metric measurements**: X and Y coordinates represent actual spatial positions in meters
- **Ready for model consumption**: No additional coordinate transformation is required in the training pipeline

### Trajectory Processing

- **Sequence extraction**: Sliding window approach with configurable observation and prediction lengths
- **Non-linearity detection**: Polynomial fitting to classify linear vs non-linear trajectories
- **Relative coordinates**: Both absolute and relative coordinate representations are computed

### Quality Filters

- **Minimum sequence length**: Trajectories must have sufficient length for obs_len + pred_len
- **Pedestrian count**: Sequences must contain minimum number of pedestrians
- **Temporal consistency**: Frame IDs must be sequential for valid trajectories

## Usage

### Loading Data

```python
from src.data.loader import data_loader
from src.core.config import Args

# Configure dataset parameters
args = Args()
args.obs_len = 8
args.pred_len = 12
args.batch_size = 64
args.delim = '\t'

# Load dataset
dset, loader = data_loader(args, 'datasets/eth/train/')
```

### Data Format in Model

- **Input trajectories**: `(batch_size, sequence_length, 2)` - observed pedestrian positions
- **Target trajectories**: `(batch_size, sequence_length, 2)` - future positions to predict
- **Relative trajectories**: Velocity/displacement representations
- **Non-linearity labels**: Binary classification for trajectory complexity

## Dependencies

The dataset processing requires the following Python packages:

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
matplotlib>=3.4.0
pandas>=1.3.0
scikit-learn>=1.0.0
```

## Citation

If you use these datasets in your research, please cite the original sources:

- ETH/Hotel datasets: [UCY and ETH datasets for pedestrian trajectory prediction]
- Zara datasets: [Crowds dataset from shopping mall environments]
- Students datasets: [University campus pedestrian movement data]

## License

Please refer to the original dataset licenses for usage terms and conditions.

## Data Integrity

### Validation Checks

- All files are tab-separated with 4 columns per line
- Coordinate values are floating-point numbers
- Frame IDs are sequential for each pedestrian
- No missing or corrupted data points

### File Sizes

- Smallest file: 481 lines (uni_examples_val.txt)
- Largest file: 21,813 lines (students001.txt)
- Total dataset size: 520,996 trajectory points across all files

---

_Last updated: August 1, 2025_
