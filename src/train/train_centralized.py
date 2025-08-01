"""
Centralized training script for DO-TP model.
Trains using centralized learning across all datasets and validates on a single dataset.
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import get_centralized_config
from core.data_manager import create_data_manager
from core.training import create_centralized_trainer
from core.evaluation import create_evaluator

def cleanup_old_files():
    """Clean up scattered files from previous runs."""
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Remove any .pt files in src directory
    for file in os.listdir(src_dir):
        if file.endswith('.pt') or file.endswith('_config.json') or file.endswith('_metrics.json'):
            file_path = os.path.join(src_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Cleaned up: {file_path}")
    
    # Remove logs directory from src if it exists
    logs_dir = os.path.join(src_dir, 'logs')
    if os.path.exists(logs_dir):
        import shutil
        shutil.rmtree(logs_dir)
        print(f"Cleaned up: {logs_dir}")

def setup_experiment_directory(config):
    """Set up the experiment directory structure."""
    output_dir = config.output.output_dir
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for organization
    subdirs = ['logs', 'models', 'plots', 'configs']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # Update config paths to use the organized structure
    config.logs_dir = os.path.join(output_dir, 'logs')
    config.models_dir = os.path.join(output_dir, 'models')
    config.plots_dir = os.path.join(output_dir, 'plots')
    config.configs_dir = os.path.join(output_dir, 'configs')
    
    return config

def setup_logging(config):
    """Setup logging configuration."""
    log_format = '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s'
    log_level = getattr(logging, config.output.log_level.upper())
    
    # Use the organized logs directory
    logs_dir = getattr(config, 'logs_dir', os.path.join(config.output.output_dir, 'logs'))
    
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
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    
    # Model arguments
    parser.add_argument('--enc_hidden_dim', type=int, default=32,
                       help='Encoder hidden dimension')
    parser.add_argument('--dest_dim', type=int, default=32,
                       help='Destination embedding dimension')
    parser.add_argument('--kl_beta', type=float, default=0.1,
                       help='KL divergence beta weight')
    
    # Output arguments
    parser.add_argument('--output_dir', default=None,
                       help='Output directory for results (defaults to current working directory)')
    parser.add_argument('--experiment_name', default='centralized_training',
                       help='Name of the experiment')
    parser.add_argument('--checkpoint_name', default='do_tp_centralized',
                       help='Checkpoint name prefix')
    
    # Hardware arguments
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='Use GPU if available')
    parser.add_argument('--gpu_num', default='0',
                       help='GPU number to use')
    
    # Utility arguments
    parser.add_argument('--cleanup', action='store_true', default=False,
                       help='Clean up scattered files from previous runs before starting')
    
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
    
    # Data config
    config.data.batch_size = args.batch_size
    
    # Model config
    config.model.enc_hidden_dim = args.enc_hidden_dim
    config.model.dest_dim = args.dest_dim
    config.model.kl_beta = args.kl_beta
    
    # Output config - only update if args.output_dir is not empty
    if args.output_dir and args.output_dir.strip():
        config.output.output_dir = args.output_dir
    else:
        # Ensure we have a proper timestamp subdirectory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.output.output_dir = os.path.join(config.output.output_dir, timestamp)
    config.output.checkpoint_name = args.checkpoint_name
    
    return config

def main():
    """Main training function."""
    # Parse arguments and create config
    args = parse_arguments()
    
    # Clean up old files if requested
    if args.cleanup:
        print("Cleaning up scattered files from previous runs...")
        cleanup_old_files()
        print("Cleanup completed.")
    
    config = create_config_from_args(args)
    
    # Setup organized directory structure
    config = setup_experiment_directory(config)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Centralized Training for DO-TP Model")
    logger.info("=" * 50)
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Training datasets: {config.train_datasets}")
    logger.info(f"  Validation dataset: {config.validation_dataset}")
    logger.info(f"  Number of epochs: {config.training.num_epochs}")
    logger.info(f"  Batch size: {config.data.batch_size}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  Output directory: {config.output.output_dir}")
    logger.info(f"  Experiment name: {config.experiment_name}")
    logger.info(f"  Use GPU: {config.training.use_gpu}")
    
    # Ensure output directory is valid
    if not config.output.output_dir or not config.output.output_dir.strip():
        logger.error("Output directory is empty or invalid")
        raise ValueError("Output directory must be specified and non-empty")
    
    try:
        # Save configuration to organized configs directory
        config_path = os.path.join(getattr(config, 'configs_dir', config.output.output_dir), f'{config.experiment_name}_config.json')
        config_dict = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        logger.info(f"Configuration saved to: {config_path}")
        
        # Create data manager
        logger.info("Creating data manager...")
        data_manager = create_data_manager(config)
        
        # Load training data (combine all training datasets)
        logger.info("Loading training data...")
        train_datasets, train_loader = data_manager.get_centralized_train_loader()
        # Get the combined dataset from the loader
        combined_train_dataset = train_loader.dataset
        logger.info(f"Training data loaded: {len(combined_train_dataset)} samples")
        
        # Load validation data
        logger.info("Loading validation data...")
        val_dataset, val_loader = data_manager.get_validation_loader()
        logger.info(f"Validation data loaded: {len(val_dataset)} samples")
        
        # Create trainer
        logger.info("Creating centralized trainer...")
        trainer = create_centralized_trainer(config)
        
        # Create evaluator
        logger.info("Creating evaluator...")
        evaluator = create_evaluator(config)
        
        # Train the model
        logger.info("Starting training...")
        training_metrics_dict = trainer.train(
            train_data=combined_train_dataset,
            val_data=val_dataset,
            evaluator=evaluator
        )
        
        logger.info("Training completed successfully!")
        
        # Save training metrics to organized directory
        metrics_path = os.path.join(getattr(config, 'configs_dir', config.output.output_dir), f'{config.experiment_name}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(training_metrics_dict, f, indent=2, default=str)
        logger.info(f"Training metrics saved to: {metrics_path}")
        
        # Create training plots
        logger.info("Creating training plots...")
        plots_dir = getattr(config, 'plots_dir', config.output.output_dir)
        try:
            # Reconstruct TrainingMetrics object from dictionary for plotting
            from core.training import TrainingMetrics
            metrics_obj = TrainingMetrics()
            
            # Populate the metrics object with data from the dictionary
            metrics_obj.train_losses = training_metrics_dict.get('train_losses', [])
            metrics_obj.total_losses = training_metrics_dict.get('total_losses', [])
            metrics_obj.traj_losses = training_metrics_dict.get('traj_losses', [])
            metrics_obj.kl_losses = training_metrics_dict.get('kl_losses', [])
            metrics_obj.val_losses = training_metrics_dict.get('val_losses', [])
            metrics_obj.val_ades = training_metrics_dict.get('val_ades', [])
            metrics_obj.val_fdes = training_metrics_dict.get('val_fdes', [])
            
            evaluator.create_training_plots(
                metrics=metrics_obj,
                experiment_name=config.experiment_name,
                save_dir=plots_dir
            )
            logger.info(f"Training plots saved to: {plots_dir}")
        except Exception as e:
            logger.warning(f"Could not create training plots: {str(e)}")
        
        # Save the final model (already saved by trainer, but let's log the organized path)
        models_dir = getattr(config, 'models_dir', config.output.output_dir)
        final_model_path = os.path.join(models_dir, f'{config.output.checkpoint_name}_final.pt')
        logger.info(f"Final model saved to: {final_model_path}")
        
        logger.info("=" * 50)
        logger.info("Centralized training completed successfully!")
        logger.info(f"Results saved to: {config.output.output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.exception("Full traceback:")
        raise

if __name__ == "__main__":
    main()
