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
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    
    # Model arguments
    parser.add_argument('--enc_hidden_dim', type=int, default=128,
                       help='Encoder hidden dimension')
    parser.add_argument('--dest_dim', type=int, default=256,
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
    
    # Create output directory
    os.makedirs(config.output.output_dir, exist_ok=True)
    
    try:
        # Save configuration
        config_path = os.path.join(config.output.output_dir, f'{config.experiment_name}_config.json')
        config_dict = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        logger.info(f"Configuration saved to: {config_path}")
        
        # Create data manager
        logger.info("Creating data manager...")
        data_manager = create_data_manager(config)
        
        # Load training data (combine all training datasets)
        logger.info("Loading training data...")
        train_data = data_manager.load_combined_training_data(config.train_datasets)
        logger.info(f"Training data loaded: {len(train_data)} samples")
        
        # Load validation data
        logger.info("Loading validation data...")
        val_data = data_manager.load_validation_data(config.validation_dataset)
        logger.info(f"Validation data loaded: {len(val_data)} samples")
        
        # Create trainer
        logger.info("Creating centralized trainer...")
        trainer = create_centralized_trainer(config)
        
        # Create evaluator
        logger.info("Creating evaluator...")
        evaluator = create_evaluator(config)
        
        # Train the model
        logger.info("Starting training...")
        training_metrics = trainer.train(
            train_data=train_data,
            val_data=val_data,
            evaluator=evaluator
        )
        
        logger.info("Training completed successfully!")
        
        # Save training metrics
        metrics_path = os.path.join(config.output.output_dir, f'{config.experiment_name}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(training_metrics, f, indent=2, default=str)
        logger.info(f"Training metrics saved to: {metrics_path}")
        
        # Final evaluation
        logger.info("Performing final evaluation...")
        final_results = evaluator.evaluate_final_model(
            model=trainer.model,
            test_data=val_data,
            save_dir=config.output.output_dir
        )
        
        logger.info("Final Results:")
        for metric, value in final_results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info("=" * 50)
        logger.info("Centralized training completed successfully!")
        logger.info(f"Results saved to: {config.output.output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.exception("Full traceback:")
        raise

if __name__ == "__main__":
    main()
