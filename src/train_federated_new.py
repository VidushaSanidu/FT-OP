"""
Federated learning training script for DO-TP model.
Trains using federated learning across multiple datasets and validates on a single dataset.
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig, get_federated_config
from core.data_manager import create_data_manager
from core.federated import create_federated_trainer
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
            logging.FileHandler(os.path.join(logs_dir, f'federated_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Federated Training for DO-TP Model')
    
    # Dataset arguments
    parser.add_argument('--client_datasets', nargs='+', 
                       default=['eth', 'hotel', 'zara1', 'zara2', 'univ'],
                       help='List of datasets to use as federated clients')
    parser.add_argument('--validation_dataset', default='zara1',
                       help='Dataset to use for validation')
    
    # Federated learning arguments
    parser.add_argument('--global_rounds', type=int, default=10,
                       help='Number of global federated rounds')
    parser.add_argument('--local_epochs', type=int, default=7,
                       help='Number of local epochs per round')
    parser.add_argument('--clients_per_round', type=int, default=5,
                       help='Number of clients to select per round')
    parser.add_argument('--aggregation_method', default='weighted_avg',
                       choices=['weighted_avg', 'simple_avg'],
                       help='Aggregation method for federated learning')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
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
    parser.add_argument('--experiment_name', default='federated_training',
                       help='Name of the experiment')
    parser.add_argument('--checkpoint_name', default='do_tp_federated',
                       help='Checkpoint name prefix')
    
    # Hardware arguments
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='Use GPU if available')
    parser.add_argument('--gpu_num', default='0',
                       help='GPU number to use')
    
    return parser.parse_args()

def create_config_from_args(args):
    """Create configuration from command line arguments."""
    config = get_federated_config()
    
    # Update config with command line arguments
    config.train_datasets = args.client_datasets
    config.validation_dataset = args.validation_dataset
    config.experiment_name = args.experiment_name
    
    # Federated config
    config.federated.global_rounds = args.global_rounds
    config.federated.local_epochs = args.local_epochs
    config.federated.clients_per_round = args.clients_per_round
    config.federated.aggregation_method = args.aggregation_method
    
    # Training config
    config.training.learning_rate = args.learning_rate
    config.training.use_gpu = args.use_gpu
    config.training.gpu_num = args.gpu_num
    
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
    """Main federated training function."""
    # Parse arguments and create config
    args = parse_arguments()
    config = create_config_from_args(args)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("FEDERATED TRAINING FOR DO-TP MODEL")
    logger.info("="*80)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Client datasets: {config.train_datasets}")
    logger.info(f"Validation dataset: {config.validation_dataset}")
    logger.info(f"Global rounds: {config.federated.global_rounds}")
    logger.info(f"Local epochs: {config.federated.local_epochs}")
    logger.info(f"Clients per round: {config.federated.clients_per_round}")
    logger.info(f"Aggregation method: {config.federated.aggregation_method}")
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
        
        # Create data loaders for federated learning
        logger.info("Creating federated client data loaders...")
        client_loaders = data_manager.get_federated_client_loaders()
        val_dataset, val_loader = data_manager.get_validation_loader()
        
        logger.info(f"Created {len(client_loaders)} federated clients")
        for client_name, (dset, loader) in client_loaders.items():
            logger.info(f"  Client {client_name}: {len(dset)} samples")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        # Create federated trainer
        logger.info("Creating federated trainer...")
        federated_trainer = create_federated_trainer(config, client_loaders, val_loader)
        
        # Print client statistics
        client_stats = federated_trainer.get_client_statistics()
        logger.info("Client statistics:")
        logger.info(f"  Total clients: {client_stats['num_clients']}")
        logger.info(f"  Total samples: {client_stats['total_samples']}")
        logger.info("  Client weights:")
        for client, weight in client_stats['client_weights'].items():
            logger.info(f"    {client}: {weight:.4f}")
        
        # Save configuration
        config_path = os.path.join(config.output.output_dir, f'{config.experiment_name}_config.json')
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
        
        # Start federated training
        logger.info("Starting federated training...")
        global_model, metrics = federated_trainer.train_federated(save_checkpoints=True)
        
        # Get final metrics
        if metrics.val_losses:
            best_round = metrics.val_losses.index(min(metrics.val_losses)) + 1
            best_loss = min(metrics.val_losses)
            best_ade = metrics.val_ades[best_round - 1] if metrics.val_ades else 'N/A'
            best_fde = metrics.val_fdes[best_round - 1] if metrics.val_fdes else 'N/A'
            
            logger.info("Federated training completed!")
            logger.info(f"Best validation metrics:")
            logger.info(f"  Round: {best_round}")
            logger.info(f"  Loss: {best_loss:.4f}")
            logger.info(f"  ADE: {best_ade:.4f}" if best_ade != 'N/A' else f"  ADE: {best_ade}")
            logger.info(f"  FDE: {best_fde:.4f}" if best_fde != 'N/A' else f"  FDE: {best_fde}")
        
        # Create evaluation plots
        if config.output.save_plots:
            logger.info("Creating training plots...")
            evaluator = create_evaluator(config)
            
            # Create custom plots for federated learning
            if metrics.val_losses:
                import matplotlib.pyplot as plt
                
                # Validation metrics over rounds
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.plot(range(1, len(metrics.val_losses) + 1), metrics.val_losses, 'b-o')
                plt.title('Validation Loss')
                plt.xlabel('Global Round')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                if metrics.val_ades:
                    plt.subplot(1, 3, 2)
                    plt.plot(range(1, len(metrics.val_ades) + 1), metrics.val_ades, 'r-o')
                    plt.title('Average Displacement Error (ADE)')
                    plt.xlabel('Global Round')
                    plt.ylabel('ADE')
                    plt.grid(True, alpha=0.3)
                
                if metrics.val_fdes:
                    plt.subplot(1, 3, 3)
                    plt.plot(range(1, len(metrics.val_fdes) + 1), metrics.val_fdes, 'g-o')
                    plt.title('Final Displacement Error (FDE)')
                    plt.xlabel('Global Round')
                    plt.ylabel('FDE')
                    plt.grid(True, alpha=0.3)
                
                plt.suptitle(f'{config.experiment_name} - Federated Training Progress')
                plt.tight_layout()
                plt.savefig(os.path.join(config.output.output_dir, f'{config.experiment_name}_progress.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("Federated training progress plots saved")
        
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
            'val_losses': [float(x) for x in metrics.val_losses],
            'val_ades': [float(x) for x in metrics.val_ades],
            'val_fdes': [float(x) for x in metrics.val_fdes],
            'client_statistics': convert_to_serializable(client_stats),
            'best_metrics': {
                'best_round': int(best_round) if metrics.val_losses else 'N/A',
                'best_val_loss': float(best_loss) if metrics.val_losses else 'N/A',
                'best_ade': float(best_ade) if best_ade != 'N/A' else 'N/A',
                'best_fde': float(best_fde) if best_fde != 'N/A' else 'N/A'
            }
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        logger.info(f"Training metrics saved to {metrics_path}")
        
        logger.info("Federated training completed successfully!")
        
    except Exception as e:
        logger.error(f"Federated training failed with error: {str(e)}")
        raise

if __name__ == '__main__':
    main()
