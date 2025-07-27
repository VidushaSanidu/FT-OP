"""
Comprehensive comparison script for centralized vs federated training.
Evaluates both approaches and provides detailed performance comparison.
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.config import ExperimentConfig, get_comparison_config
from core.data_manager import create_data_manager
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
            logging.FileHandler(os.path.join(logs_dir, f'comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare Centralized vs Federated Training')
    
    # Model paths
    parser.add_argument('--centralized_model', required=True,
                       help='Path to centralized model checkpoint')
    parser.add_argument('--federated_model', required=True,
                       help='Path to federated model checkpoint')
    
    # Evaluation arguments
    parser.add_argument('--test_datasets', nargs='+', 
                       default=['eth', 'hotel', 'zara1', 'zara2', 'univ'],
                       help='List of datasets to evaluate on')
    parser.add_argument('--validation_dataset', default='zara1',
                       help='Primary validation dataset')
    
    # Model arguments (should match training)
    parser.add_argument('--enc_hidden_dim', type=int, default=32,
                       help='Encoder hidden dimension')
    parser.add_argument('--dest_dim', type=int, default=32,
                       help='Destination dimension')
    parser.add_argument('--kl_beta', type=float, default=0.1,
                       help='KL divergence loss weight')
    
    # Output arguments
    parser.add_argument('--output_dir', default=os.getcwd(),
                       help='Output directory for results')
    parser.add_argument('--experiment_name', default='centralized_vs_federated',
                       help='Name of the comparison experiment')
    
    # Hardware arguments
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='Use GPU if available')
    parser.add_argument('--gpu_num', default='0',
                       help='GPU number to use')
    
    # Comparison options
    parser.add_argument('--save_plots', action='store_true', default=True,
                       help='Save comparison plots')
    parser.add_argument('--detailed_analysis', action='store_true', default=True,
                       help='Perform detailed analysis')
    
    return parser.parse_args()

def create_config_from_args(args):
    """Create configuration from command line arguments."""
    config = get_comparison_config()
    
    # Update config with command line arguments
    config.validation_dataset = args.validation_dataset
    config.experiment_name = args.experiment_name
    
    # Training config
    config.training.use_gpu = args.use_gpu
    config.training.gpu_num = args.gpu_num
    
    # Model config
    config.model.enc_hidden_dim = args.enc_hidden_dim
    config.model.dest_dim = args.dest_dim
    config.model.kl_beta = args.kl_beta
    
    # Output config
    config.output.output_dir = args.output_dir
    config.output.save_plots = args.save_plots
    
    return config

def load_training_metrics(centralized_metrics_path: str, federated_metrics_path: str):
    """Load training metrics from both approaches."""
    centralized_metrics = None
    federated_metrics = None
    
    if os.path.exists(centralized_metrics_path):
        with open(centralized_metrics_path, 'r') as f:
            centralized_metrics = json.load(f)
    
    if os.path.exists(federated_metrics_path):
        with open(federated_metrics_path, 'r') as f:
            federated_metrics = json.load(f)
    
    return centralized_metrics, federated_metrics

def analyze_training_efficiency(centralized_metrics, federated_metrics, logger):
    """Analyze training efficiency between approaches."""
    logger.info("\n" + "="*80)
    logger.info("TRAINING EFFICIENCY ANALYSIS")
    logger.info("="*80)
    
    if centralized_metrics and 'best_metrics' in centralized_metrics:
        cent_best = centralized_metrics['best_metrics']
        logger.info("Centralized Training:")
        logger.info(f"  Best Epoch: {cent_best.get('best_epoch', 'N/A')}")
        logger.info(f"  Best Val Loss: {cent_best.get('best_val_loss', 'N/A'):.4f}")
        logger.info(f"  Best ADE: {cent_best.get('best_ade', 'N/A'):.4f}")
        logger.info(f"  Best FDE: {cent_best.get('best_fde', 'N/A'):.4f}")
        logger.info(f"  Total Training Time: {cent_best.get('total_training_time', 'N/A'):.2f}s")
    
    if federated_metrics and 'best_metrics' in federated_metrics:
        fed_best = federated_metrics['best_metrics']
        logger.info("\nFederated Training:")
        logger.info(f"  Best Round: {fed_best.get('best_round', 'N/A')}")
        logger.info(f"  Best Val Loss: {fed_best.get('best_val_loss', 'N/A'):.4f}")
        logger.info(f"  Best ADE: {fed_best.get('best_ade', 'N/A'):.4f}")
        logger.info(f"  Best FDE: {fed_best.get('best_fde', 'N/A'):.4f}")
    
    # Compare convergence
    if (centralized_metrics and federated_metrics and 
        'val_losses' in centralized_metrics and 'val_losses' in federated_metrics):
        
        cent_losses = centralized_metrics['val_losses']
        fed_losses = federated_metrics['val_losses']
        
        logger.info(f"\nConvergence Analysis:")
        logger.info(f"  Centralized converged in {len(cent_losses)} epochs")
        logger.info(f"  Federated converged in {len(fed_losses)} rounds")
        
        if cent_losses and fed_losses:
            logger.info(f"  Final centralized loss: {cent_losses[-1]:.4f}")
            logger.info(f"  Final federated loss: {fed_losses[-1]:.4f}")

def create_comprehensive_comparison_plots(comparison_results, centralized_metrics, 
                                        federated_metrics, config, logger):
    """Create comprehensive comparison plots."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    save_dir = config.output.output_dir
    
    # 1. Performance comparison bar plots
    if comparison_results:
        models = list(comparison_results.keys())
        datasets = list(comparison_results[models[0]].keys()) if models else []
        
        if len(models) == 2 and datasets:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            metrics = ['ade', 'fde', 'loss']
            
            for i, metric in enumerate(metrics):
                x = np.arange(len(datasets))
                width = 0.35
                
                model1_values = [comparison_results[models[0]][dataset][metric] for dataset in datasets]
                model2_values = [comparison_results[models[1]][dataset][metric] for dataset in datasets]
                
                bars1 = axes[i].bar(x - width/2, model1_values, width, 
                                  label=models[0], alpha=0.8, color='skyblue')
                bars2 = axes[i].bar(x + width/2, model2_values, width, 
                                  label=models[1], alpha=0.8, color='lightcoral')
                
                # Add value labels
                for bar in bars1:
                    height = bar.get_height()
                    axes[i].annotate(f'{height:.3f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3), textcoords="offset points",
                                   ha='center', va='bottom', fontsize=8)
                
                for bar in bars2:
                    height = bar.get_height()
                    axes[i].annotate(f'{height:.3f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3), textcoords="offset points",
                                   ha='center', va='bottom', fontsize=8)
                
                axes[i].set_xlabel('Datasets')
                axes[i].set_ylabel(metric.upper())
                axes[i].set_title(f'{metric.upper()} Comparison')
                axes[i].set_xticks(x)
                axes[i].set_xticklabels(datasets)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            plt.suptitle('Centralized vs Federated: Performance Comparison')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Performance comparison plot saved")
    
    # 2. Training progress comparison
    if (centralized_metrics and federated_metrics and 
        'val_losses' in centralized_metrics and 'val_losses' in federated_metrics):
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Validation loss comparison
        cent_losses = centralized_metrics['val_losses']
        fed_losses = federated_metrics['val_losses']
        
        axes[0, 0].plot(range(1, len(cent_losses) + 1), cent_losses, 
                       'b-', label='Centralized', linewidth=2)
        axes[0, 0].plot(range(1, len(fed_losses) + 1), fed_losses, 
                       'r-', label='Federated', linewidth=2)
        axes[0, 0].set_title('Validation Loss Comparison')
        axes[0, 0].set_xlabel('Epoch/Round')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # ADE comparison
        if ('val_ades' in centralized_metrics and 'val_ades' in federated_metrics):
            cent_ades = centralized_metrics['val_ades']
            fed_ades = federated_metrics['val_ades']
            
            axes[0, 1].plot(range(1, len(cent_ades) + 1), cent_ades, 
                           'b-', label='Centralized', linewidth=2)
            axes[0, 1].plot(range(1, len(fed_ades) + 1), fed_ades, 
                           'r-', label='Federated', linewidth=2)
            axes[0, 1].set_title('ADE Comparison')
            axes[0, 1].set_xlabel('Epoch/Round')
            axes[0, 1].set_ylabel('ADE')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # FDE comparison
        if ('val_fdes' in centralized_metrics and 'val_fdes' in federated_metrics):
            cent_fdes = centralized_metrics['val_fdes']
            fed_fdes = federated_metrics['val_fdes']
            
            axes[1, 0].plot(range(1, len(cent_fdes) + 1), cent_fdes, 
                           'b-', label='Centralized', linewidth=2)
            axes[1, 0].plot(range(1, len(fed_fdes) + 1), fed_fdes, 
                           'r-', label='Federated', linewidth=2)
            axes[1, 0].set_title('FDE Comparison')
            axes[1, 0].set_xlabel('Epoch/Round')
            axes[1, 0].set_ylabel('FDE')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Training loss for centralized (if available)
        if 'total_losses' in centralized_metrics:
            cent_train_losses = centralized_metrics['total_losses']
            axes[1, 1].plot(range(1, len(cent_train_losses) + 1), cent_train_losses, 
                           'b-', label='Centralized Training Loss', linewidth=2)
            axes[1, 1].set_title('Centralized Training Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Training Progress Comparison')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_progress_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Training progress comparison plot saved")

def main():
    """Main comparison function."""
    # Parse arguments and create config
    args = parse_arguments()
    config = create_config_from_args(args)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("CENTRALIZED VS FEDERATED TRAINING COMPARISON")
    logger.info("="*80)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Centralized model: {args.centralized_model}")
    logger.info(f"Federated model: {args.federated_model}")
    logger.info(f"Test datasets: {args.test_datasets}")
    logger.info(f"Output directory: {config.output.output_dir}")
    
    try:
        # Verify model files exist
        if not os.path.exists(args.centralized_model):
            raise FileNotFoundError(f"Centralized model not found: {args.centralized_model}")
        if not os.path.exists(args.federated_model):
            raise FileNotFoundError(f"Federated model not found: {args.federated_model}")
        
        # Create data manager
        logger.info("Setting up data manager...")
        config.train_datasets = args.test_datasets  # For validation purposes
        data_manager = create_data_manager(config)
        
        # Create test data loaders
        logger.info("Creating test data loaders...")
        test_loaders = {}
        for dataset_name in args.test_datasets:
            test_dset, test_loader = data_manager.get_test_loader(dataset_name)
            if test_dset is not None and test_loader is not None:
                test_loaders[dataset_name] = test_loader
                logger.info(f"Test dataset {dataset_name}: {len(test_dset)} samples")
        
        if not test_loaders:
            raise ValueError("No valid test datasets found")
        
        # Create evaluator
        logger.info("Creating evaluator...")
        evaluator = create_evaluator(config)
        
        # Evaluate both models
        logger.info("Evaluating models...")
        model_paths = {
            'Centralized': args.centralized_model,
            'Federated': args.federated_model
        }
        
        comparison_results = evaluator.compare_models(model_paths, test_loaders)
        
        # Print comparison summary
        evaluator.print_comparison_summary(comparison_results)
        
        # Load training metrics if available
        centralized_metrics_path = os.path.join(
            os.path.dirname(args.centralized_model), 'centralized_training_metrics.json'
        )
        federated_metrics_path = os.path.join(
            os.path.dirname(args.federated_model), 'federated_training_metrics.json'
        )
        
        centralized_metrics, federated_metrics = load_training_metrics(
            centralized_metrics_path, federated_metrics_path
        )
        
        # Analyze training efficiency
        if args.detailed_analysis:
            analyze_training_efficiency(centralized_metrics, federated_metrics, logger)
        
        # Create comparison plots
        if config.output.save_plots:
            logger.info("Creating comparison plots...")
            evaluator.create_comparison_plots(comparison_results, config.output.output_dir)
            
            # Create comprehensive plots
            create_comprehensive_comparison_plots(
                comparison_results, centralized_metrics, federated_metrics, config, logger
            )
        
        # Save detailed results
        results_summary = {
            'comparison_results': comparison_results,
            'centralized_metrics': centralized_metrics,
            'federated_metrics': federated_metrics,
            'model_paths': model_paths,
            'test_datasets': args.test_datasets,
            'experiment_info': {
                'date': datetime.now().isoformat(),
                'config': config.to_dict()
            }
        }
        
        results_path = os.path.join(config.output.output_dir, 'detailed_comparison_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        logger.info(f"Detailed results saved to {results_path}")
        
        # Generate summary report
        summary_path = os.path.join(config.output.output_dir, 'comparison_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("CENTRALIZED VS FEDERATED TRAINING COMPARISON SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Centralized Model: {args.centralized_model}\n")
            f.write(f"Federated Model: {args.federated_model}\n")
            f.write(f"Test Datasets: {', '.join(args.test_datasets)}\n\n")
            
            # Performance summary
            if comparison_results:
                models = list(comparison_results.keys())
                datasets = list(comparison_results[models[0]].keys()) if models else []
                
                for dataset in datasets:
                    f.write(f"\nDataset: {dataset.upper()}\n")
                    f.write("-" * 30 + "\n")
                    for model in models:
                        results = comparison_results[model][dataset]
                        f.write(f"{model:>12}: ADE={results['ade']:.4f}, FDE={results['fde']:.4f}, Loss={results['loss']:.4f}\n")
                
                # Overall averages
                f.write(f"\nOVERALL AVERAGES\n")
                f.write("-" * 30 + "\n")
                for model in models:
                    avg_ade = np.mean([comparison_results[model][d]['ade'] for d in datasets])
                    avg_fde = np.mean([comparison_results[model][d]['fde'] for d in datasets])
                    avg_loss = np.mean([comparison_results[model][d]['loss'] for d in datasets])
                    f.write(f"{model:>12}: ADE={avg_ade:.4f}, FDE={avg_fde:.4f}, Loss={avg_loss:.4f}\n")
        
        logger.info(f"Summary report saved to {summary_path}")
        logger.info("Comparison analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Comparison failed with error: {str(e)}")
        raise

if __name__ == '__main__':
    main()
