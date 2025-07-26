"""
Configuration module for DO-TP trajectory prediction experiments.
Contains default configurations for centralized and federated training.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class ModelConfig:
    """Configuration for the DO-TP model architecture."""
    obs_len: int = 8
    pred_len: int = 12
    input_dim: int = 2
    enc_hidden_dim: int = 32
    dest_dim: int = 32
    kl_beta: float = 0.1

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    delim: str = '\t'
    skip: int = 1
    loader_num_workers: int = 4
    batch_size: int = 64
    available_datasets: List[str] = field(default_factory=lambda: ['eth', 'hotel', 'zara1', 'zara2', 'univ'])

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    num_epochs: int = 200
    learning_rate: float = 1e-3
    print_every: int = 50
    use_gpu: bool = True
    gpu_num: str = "0"
    early_stopping_patience: int = 20
    min_delta: float = 1e-4

@dataclass
class FederatedConfig:
    """Configuration specific to federated learning."""
    global_rounds: int = 10
    local_epochs: int = 7
    clients_per_round: int = 5
    aggregation_method: str = "weighted_avg"  # weighted_avg, simple_avg
    
@dataclass
class OutputConfig:
    """Configuration for output and logging."""
    output_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results"))
    checkpoint_name: str = "do_tp_checkpoint"
    save_plots: bool = True
    save_logs: bool = True
    log_level: str = "INFO"

@dataclass
class ExperimentConfig:
    """Main configuration class combining all configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Experiment specific
    experiment_name: str = "do_tp_experiment"
    validation_dataset: str = "zara1"
    train_datasets: Optional[List[str]] = None  # If None, uses all available datasets
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.train_datasets is None:
            self.train_datasets = self.data.available_datasets.copy()
        
        # Ensure output directory exists
        os.makedirs(self.output.output_dir, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for easy serialization."""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'federated': self.federated.__dict__,
            'output': self.output.__dict__,
            'experiment_name': self.experiment_name,
            'validation_dataset': self.validation_dataset,
            'train_datasets': self.train_datasets
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        config = cls()
        config.model = ModelConfig(**config_dict.get('model', {}))
        config.data = DataConfig(**config_dict.get('data', {}))
        config.training = TrainingConfig(**config_dict.get('training', {}))
        config.federated = FederatedConfig(**config_dict.get('federated', {}))
        config.output = OutputConfig(**config_dict.get('output', {}))
        config.experiment_name = config_dict.get('experiment_name', 'do_tp_experiment')
        config.validation_dataset = config_dict.get('validation_dataset', 'zara1')
        config.train_datasets = config_dict.get('train_datasets', None)
        return config

def get_centralized_config() -> ExperimentConfig:
    """Get default configuration for centralized training."""
    config = ExperimentConfig()
    config.experiment_name = "centralized_training"
    config.training.num_epochs = 200
    config.data.batch_size = 64
    # Set proper results directory for centralized training
    results_base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
    config.output.output_dir = os.path.join(results_base, "centralized")
    config.output.checkpoint_name = "do_tp_centralized"
    return config

def get_federated_config() -> ExperimentConfig:
    """Get default configuration for federated training."""
    config = ExperimentConfig()
    config.experiment_name = "federated_training"
    config.federated.global_rounds = 10
    config.federated.local_epochs = 7
    config.data.batch_size = 32
    # Set proper results directory for federated training
    results_base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
    config.output.output_dir = os.path.join(results_base, "federated")
    config.output.checkpoint_name = "do_tp_federated"
    return config

def get_comparison_config() -> ExperimentConfig:
    """Get configuration for comparison experiments."""
    config = ExperimentConfig()
    config.experiment_name = "comparison_experiment"
    config.training.num_epochs = 100  # Shorter for comparison
    config.federated.global_rounds = 5
    return config
