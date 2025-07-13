"""
Centralized training script for DO-TP model.
Trains on all specified datasets and validates on a single dataset.
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig, get_centralized_config
from core.data_manager import create_data_manager
from core.training import create_model, get_device, train_model
from core.evaluation import create_evaluator

def setup_logging(config):
    """Setup logging configuration."""
    log_format = '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s'
    log_level = getattr(logging, config.output.log_level.upper())
    
    # Create logs directory
    logs_dir = os.path.join(config.output.output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Setup file and console logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, f'centralized_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Centralized Training for DO-TP Model')
    
    # Dataset arguments
    parser.add_argument('--train_datasets', nargs='+', 
                       default=['eth', 'hotel', 'zara1', 'zara2', 'univ'],
                       help='List of datasets to use for training')
    parser.add_argument('--validation_dataset', default='zara1',
                       help='Dataset to use for validation')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    
    # Model arguments
    parser.add_argument('--enc_hidden_dim', type=int, default=32,
                       help='Encoder hidden dimension')
    parser.add_argument('--dest_dim', type=int, default=32,
                       help='Destination dimension')
    parser.add_argument('--kl_beta', type=float, default=0.1,
                       help='KL divergence loss weight')
    
    # Output arguments
    parser.add_argument('--output_dir', default=os.getcwd(),
                       help='Output directory for results')
    parser.add_argument('--experiment_name', default='centralized_training',
                       help='Name of the experiment')
    parser.add_argument('--checkpoint_name', default='do_tp_centralized',
                       help='Checkpoint name prefix')
    
    # Hardware arguments
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='Use GPU if available')
    parser.add_argument('--gpu_num', default='0',
                       help='GPU number to use')
    
    # Other arguments
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--print_every', type=int, default=50,
                       help='Print frequency during training')
    
    return parser.parse_args()

def create_config_from_args(args):
    """Create configuration from command line arguments."""
    config = get_centralized_config()
    
    # Update config with command line arguments
    config.train_datasets = args.train_datasets
    config.validation_dataset = args.validation_dataset
    config.experiment_name = args.experiment_name
    
    # Training config
    config.training.num_epochs = args.num_epochs
    config.training.learning_rate = args.learning_rate
    config.training.use_gpu = args.use_gpu
    config.training.gpu_num = args.gpu_num
    config.training.early_stopping_patience = args.early_stopping_patience
    config.training.print_every = args.print_every
    
    # Data config
    config.data.batch_size = args.batch_size
    
    # Model config
    config.model.enc_hidden_dim = args.enc_hidden_dim
    config.model.dest_dim = args.dest_dim
    config.model.kl_beta = args.kl_beta
    
    # Output config
    config.output.output_dir = args.output_dir
    config.output.checkpoint_name = args.checkpoint_name
    
    return config

def main():
    """Main training function."""
    # Parse arguments and create config
    args = parse_arguments()
    config = create_config_from_args(args)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("CENTRALIZED TRAINING FOR DO-TP MODEL")
    logger.info("="*80)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Training datasets: {config.train_datasets}")
    logger.info(f"Validation dataset: {config.validation_dataset}")
    logger.info(f"Output directory: {config.output.output_dir}")
    
    try:
        # Create data manager and validate datasets
        logger.info("Setting up data manager...")
        data_manager = create_data_manager(config)
        
        # Print data statistics
        stats = data_manager.get_data_statistics()
        logger.info("Dataset statistics:")
        for dataset, sizes in stats['dataset_sizes'].items():
            logger.info(f"  {dataset}: train={sizes['train']}, val={sizes['val']}, test={sizes['test']}")
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_datasets, train_loader = data_manager.get_centralized_train_loader()
        val_dataset, val_loader = data_manager.get_validation_loader()
        
        logger.info(f"Training samples: {sum(len(dset) for dset in train_datasets)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        # Create model and device
        logger.info("Creating model...")
        device = get_device(config.training.use_gpu, config.training.gpu_num)
        model = create_model(config)
        model.to(device)
        
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        logger.info(f"Device: {device}")
        
        # Save configuration
        config_path = os.path.join(config.output.output_dir, f'{config.experiment_name}_config.json')
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
        
        # Train model
        logger.info("Starting training...")
        checkpoint_path = os.path.join(config.output.output_dir, f'{config.output.checkpoint_name}_best.pt')
        metrics = train_model(model, train_loader, val_loader, config, device, checkpoint_path)
        
        # Get best metrics
        best_metrics = metrics.get_best_metrics()
        logger.info("Training completed!")
        logger.info(f"Best validation metrics:")
        logger.info(f"  Epoch: {best_metrics.get('best_epoch', 'N/A')}")
        logger.info(f"  Loss: {best_metrics.get('best_val_loss', 'N/A'):.4f}")
        logger.info(f"  ADE: {best_metrics.get('best_ade', 'N/A'):.4f}")
        logger.info(f"  FDE: {best_metrics.get('best_fde', 'N/A'):.4f}")
        logger.info(f"  Total training time: {best_metrics.get('total_training_time', 'N/A'):.2f}s")
        
        # Create evaluation plots
        if config.output.save_plots:
            logger.info("Creating training plots...")
            evaluator = create_evaluator(config)
            evaluator.create_training_plots(metrics, config.experiment_name, config.output.output_dir)
        
        # Save final model
        final_model_path = os.path.join(config.output.output_dir, f'{config.output.checkpoint_name}_final.pt')
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        # Save metrics
        import numpy as np
        metrics_path = os.path.join(config.output.output_dir, f'{config.experiment_name}_metrics.json')
        
        def convert_to_serializable(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            else:
                return obj
        
        metrics_dict = {
            'total_losses': [float(x) for x in metrics.total_losses],
            'traj_losses': [float(x) for x in metrics.traj_losses],
            'kl_losses': [float(x) for x in metrics.kl_losses],
            'val_losses': [float(x) for x in metrics.val_losses],
            'val_ades': [float(x) for x in metrics.val_ades],
            'val_fdes': [float(x) for x in metrics.val_fdes],
            'training_times': [float(x) for x in metrics.training_times],
            'best_metrics': convert_to_serializable(best_metrics)
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        logger.info(f"Training metrics saved to {metrics_path}")
        
        logger.info("Centralized training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == '__main__':
    import torch
    main()
