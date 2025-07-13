"""
Evaluation and comparison utilities for model performance analysis.
"""

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import torch
from torch.utils.data import DataLoader

from core.training import TrainingMetrics, evaluate_model, load_model, create_model

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation and comparison utilities."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if config.training.use_gpu and torch.cuda.is_available() else 'cpu')
    
    def evaluate_single_model(
        self, 
        model_path: str, 
        test_loaders: Dict[str, DataLoader],
        model_name: str = "model"
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate a single model on multiple test datasets."""
        # Load model
        model = create_model(self.config)
        model = load_model(model, model_path, self.device)
        
        results = {}
        logger.info(f"Evaluating {model_name} on test datasets...")
        
        for dataset_name, test_loader in test_loaders.items():
            val_loss, ade, fde = evaluate_model(model, test_loader, self.device)
            results[dataset_name] = {
                'loss': val_loss,
                'ade': ade,
                'fde': fde
            }
            logger.info(f"{model_name} on {dataset_name}: Loss={val_loss:.4f}, ADE={ade:.4f}, FDE={fde:.4f}")
        
        return results
    
    def compare_models(
        self, 
        model_paths: Dict[str, str], 
        test_loaders: Dict[str, DataLoader]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compare multiple models on multiple test datasets."""
        comparison_results = {}
        
        for model_name, model_path in model_paths.items():
            if os.path.exists(model_path):
                comparison_results[model_name] = self.evaluate_single_model(
                    model_path, test_loaders, model_name
                )
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        return comparison_results
    
    def create_comparison_plots(
        self, 
        comparison_results: Dict[str, Dict[str, Dict[str, float]]],
        save_dir: str = None
    ):
        """Create comparison plots for model performance."""
        if save_dir is None:
            save_dir = self.config.output.output_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract data for plotting
        models = list(comparison_results.keys())
        datasets = list(comparison_results[models[0]].keys()) if models else []
        metrics = ['ade', 'fde', 'loss']
        
        for metric in metrics:
            self._plot_metric_comparison(comparison_results, metric, models, datasets, save_dir)
    
    def _plot_metric_comparison(
        self, 
        results: Dict[str, Dict[str, Dict[str, float]]], 
        metric: str, 
        models: List[str], 
        datasets: List[str], 
        save_dir: str
    ):
        """Plot comparison for a specific metric."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(datasets))
        width = 0.35
        
        if len(models) == 2:
            # Two models comparison
            model1_values = [results[models[0]][dataset][metric] for dataset in datasets]
            model2_values = [results[models[1]][dataset][metric] for dataset in datasets]
            
            bars1 = ax.bar(x - width/2, model1_values, width, label=models[0], alpha=0.8)
            bars2 = ax.bar(x + width/2, model2_values, width, label=models[1], alpha=0.8)
            
            # Add value labels on bars
            self._add_value_labels(ax, bars1)
            self._add_value_labels(ax, bars2)
        else:
            # Multiple models - use different approach
            for i, model in enumerate(models):
                values = [results[model][dataset][metric] for dataset in datasets]
                offset = (i - len(models)/2 + 0.5) * width / len(models)
                bars = ax.bar(x + offset, values, width/len(models), label=model, alpha=0.8)
        
        ax.set_xlabel('Datasets')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} Comparison Across Datasets')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{metric}_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved {metric} comparison plot")
    
    def _add_value_labels(self, ax, bars):
        """Add value labels on top of bars."""
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)
    
    def create_training_plots(
        self, 
        metrics: TrainingMetrics, 
        experiment_name: str, 
        save_dir: str = None
    ):
        """Create training progress plots."""
        if save_dir is None:
            save_dir = self.config.output.output_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss curves
        if metrics.total_losses:
            self._plot_loss_curves(metrics, experiment_name, save_dir)
        
        # Validation metrics
        if metrics.val_losses:
            self._plot_validation_metrics(metrics, experiment_name, save_dir)
    
    def _plot_loss_curves(self, metrics: TrainingMetrics, experiment_name: str, save_dir: str):
        """Plot training loss curves."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(metrics.total_losses) + 1)
        
        # Total loss
        ax1.plot(epochs, metrics.total_losses, 'b-', label='Total Loss')
        ax1.set_title('Total Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Trajectory loss
        ax2.plot(epochs, metrics.traj_losses, 'r-', label='Trajectory Loss')
        ax2.set_title('Trajectory Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # KL loss
        ax3.plot(epochs, metrics.kl_losses, 'g-', label='KL Loss')
        ax3.set_title('KL Divergence Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Validation loss (if available)
        if metrics.val_losses:
            val_epochs = range(1, len(metrics.val_losses) + 1)
            ax4.plot(val_epochs, metrics.val_losses, 'purple', label='Validation Loss')
            ax4.set_title('Validation Loss')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        plt.suptitle(f'{experiment_name} - Training Progress')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{experiment_name}_training_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training curves for {experiment_name}")
    
    def _plot_validation_metrics(self, metrics: TrainingMetrics, experiment_name: str, save_dir: str):
        """Plot validation metrics."""
        if not metrics.val_ades or not metrics.val_fdes:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = range(1, len(metrics.val_ades) + 1)
        
        # ADE
        ax1.plot(epochs, metrics.val_ades, 'b-', label='ADE')
        ax1.set_title('Average Displacement Error (ADE)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('ADE')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # FDE
        ax2.plot(epochs, metrics.val_fdes, 'r-', label='FDE')
        ax2.set_title('Final Displacement Error (FDE)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('FDE')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.suptitle(f'{experiment_name} - Validation Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{experiment_name}_validation_metrics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved validation metrics for {experiment_name}")
    
    def save_results_summary(
        self, 
        comparison_results: Dict[str, Dict[str, Dict[str, float]]], 
        save_path: str = None
    ):
        """Save comparison results to JSON file."""
        if save_path is None:
            save_path = os.path.join(self.config.output.output_dir, 'comparison_results.json')
        
        # Add summary statistics
        summary = {
            'detailed_results': comparison_results,
            'summary': self._compute_summary_stats(comparison_results)
        }
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results summary saved to {save_path}")
    
    def _compute_summary_stats(
        self, 
        results: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Any]:
        """Compute summary statistics for comparison results."""
        summary = {}
        
        models = list(results.keys())
        if not models:
            return summary
        
        datasets = list(results[models[0]].keys())
        metrics = ['ade', 'fde', 'loss']
        
        for metric in metrics:
            summary[metric] = {}
            for model in models:
                values = [results[model][dataset][metric] for dataset in datasets]
                summary[metric][model] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return summary
    
    def print_comparison_summary(
        self, 
        comparison_results: Dict[str, Dict[str, Dict[str, float]]]
    ):
        """Print a formatted summary of comparison results."""
        models = list(comparison_results.keys())
        if not models:
            logger.warning("No models to compare")
            return
        
        datasets = list(comparison_results[models[0]].keys())
        
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        for dataset in datasets:
            print(f"\nDataset: {dataset.upper()}")
            print("-" * 40)
            print(f"{'Model':<20} {'ADE':<10} {'FDE':<10} {'Loss':<10}")
            print("-" * 40)
            
            for model in models:
                results = comparison_results[model][dataset]
                print(f"{model:<20} {results['ade']:<10.4f} {results['fde']:<10.4f} {results['loss']:<10.4f}")
        
        # Overall best performance
        print(f"\n{'='*80}")
        print("OVERALL BEST PERFORMANCE")
        print("="*80)
        
        for metric in ['ade', 'fde', 'loss']:
            best_model = None
            best_value = float('inf') if metric != 'accuracy' else 0
            
            for model in models:
                avg_value = np.mean([comparison_results[model][dataset][metric] 
                                   for dataset in datasets])
                if avg_value < best_value:
                    best_value = avg_value
                    best_model = model
            
            print(f"Best {metric.upper()}: {best_model} ({best_value:.4f})")

def create_evaluator(config) -> ModelEvaluator:
    """Factory function to create model evaluator."""
    return ModelEvaluator(config)
