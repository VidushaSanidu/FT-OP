"""
Core training module with reusable training and evaluation functions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import DataLoader

from models.DO_TP_model import DO_TP

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop early."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop

class TrainingMetrics:
    """Container for training metrics."""
    
    def __init__(self):
        self.total_losses = []
        self.traj_losses = []
        self.kl_losses = []
        self.val_losses = []
        self.val_ades = []
        self.val_fdes = []
        self.training_times = []
    
    def add_train_metrics(self, total_loss: float, traj_loss: float, kl_loss: float):
        """Add training metrics for an epoch."""
        self.total_losses.append(total_loss)
        self.traj_losses.append(traj_loss)
        self.kl_losses.append(kl_loss)
    
    def add_val_metrics(self, val_loss: float, ade: float, fde: float):
        """Add validation metrics for an epoch."""
        self.val_losses.append(val_loss)
        self.val_ades.append(ade)
        self.val_fdes.append(fde)
    
    def add_training_time(self, time_seconds: float):
        """Add training time for an epoch."""
        self.training_times.append(time_seconds)
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best validation metrics."""
        if not self.val_losses:
            return {}
        
        best_idx = np.argmin(self.val_losses)
        return {
            'best_epoch': best_idx + 1,
            'best_val_loss': self.val_losses[best_idx],
            'best_ade': self.val_ades[best_idx],
            'best_fde': self.val_fdes[best_idx],
            'total_training_time': sum(self.training_times)
        }

def get_device(use_gpu: bool = True, gpu_num: str = "0") -> torch.device:
    """Get the appropriate device for training."""
    if use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_num}')
        logger.info(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    return device

def create_model(config) -> DO_TP:
    """Create and initialize the DO-TP model."""
    model = DO_TP(
        obs_len=config.model.obs_len,
        pred_len=config.model.pred_len,
        input_dim=config.model.input_dim,
        enc_hidden_dim=config.model.enc_hidden_dim,
        dest_dim=config.model.dest_dim,
        kl_beta=config.model.kl_beta
    )
    return model

def create_optimizer(model: nn.Module, learning_rate: float) -> optim.Optimizer:
    """Create optimizer for the model."""
    return optim.Adam(model.parameters(), lr=learning_rate)

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    print_every: int = 50
) -> Tuple[float, float, float]:
    """Train the model for one epoch."""
    model.train()
    epoch_total_loss = 0.0
    epoch_traj_loss = 0.0
    epoch_kl_loss = 0.0
    num_batches = 0
    
    for i, batch in enumerate(train_loader):
        batch = [tensor.to(device) for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, 
         non_linear_ped, loss_mask, seq_start_end) = batch
        
        optimizer.zero_grad()
        pred_disp, d_i, d_hat_i = model(obs_traj, pred_traj_gt)
        total_loss, traj_loss, kl_loss = model.compute_loss(
            pred_disp, pred_traj_gt_rel, d_i, d_hat_i
        )
        
        total_loss.backward()
        optimizer.step()
        
        epoch_total_loss += total_loss.item()
        epoch_traj_loss += traj_loss.item()
        epoch_kl_loss += kl_loss.item()
        num_batches += 1
        
        if (i + 1) % print_every == 0:
            logger.info(
                f'Batch [{i+1}/{len(train_loader)}], '
                f'Total Loss: {total_loss.item():.4f}, '
                f'Traj Loss: {traj_loss.item():.4f}, '
                f'KL Loss: {kl_loss.item():.4f}'
            )
    
    return (epoch_total_loss / num_batches, 
            epoch_traj_loss / num_batches, 
            epoch_kl_loss / num_batches)

def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float]:
    """Evaluate the model on validation data."""
    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = [tensor.to(device) for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, 
             non_linear_ped, loss_mask, seq_start_end) = batch
            
            pred_disp, d_i, d_hat_i = model(obs_traj, pred_traj_gt)
            loss, _, _ = model.compute_loss(pred_disp, pred_traj_gt_rel, d_i, d_hat_i)
            
            # Calculate prediction trajectory from displacement
            pred_traj = pred_disp + obs_traj[:, -1].unsqueeze(1)
            
            # Calculate ADE and FDE
            batch_size = pred_traj.size(0)
            ade = torch.norm(pred_traj - pred_traj_gt, dim=2).mean().item()
            fde = torch.norm(pred_traj[:, -1] - pred_traj_gt[:, -1], dim=1).mean().item()
            
            total_loss += loss.item() * batch_size
            total_ade += ade * batch_size
            total_fde += fde * batch_size
            total_samples += batch_size
    
    avg_loss = total_loss / total_samples
    avg_ade = total_ade / total_samples
    avg_fde = total_fde / total_samples
    
    return avg_loss, avg_ade, avg_fde

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    device: torch.device,
    save_path: Optional[str] = None
) -> TrainingMetrics:
    """Complete training loop for the model."""
    optimizer = create_optimizer(model, config.training.learning_rate)
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        min_delta=config.training.min_delta
    )
    metrics = TrainingMetrics()
    
    logger.info(f"Starting training for {config.training.num_epochs} epochs")
    
    for epoch in range(config.training.num_epochs):
        start_time = time.time()
        logger.info(f'Starting epoch {epoch + 1}/{config.training.num_epochs}')
        
        # Training phase
        total_loss, traj_loss, kl_loss = train_one_epoch(
            model, train_loader, optimizer, device, config.training.print_every
        )
        
        # Validation phase
        val_loss, val_ade, val_fde = evaluate_model(model, val_loader, device)
        
        # Record metrics
        epoch_time = time.time() - start_time
        metrics.add_train_metrics(total_loss, traj_loss, kl_loss)
        metrics.add_val_metrics(val_loss, val_ade, val_fde)
        metrics.add_training_time(epoch_time)
        
        logger.info(
            f'Epoch [{epoch+1}/{config.training.num_epochs}] '
            f'Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, '
            f'ADE: {val_ade:.4f}, FDE: {val_fde:.4f}, Time: {epoch_time:.2f}s'
        )
        
        # Early stopping check
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break
        
        # Save best model
        if save_path and val_loss == min(metrics.val_losses):
            torch.save(model.state_dict(), save_path)
            logger.info(f"Best model saved to {save_path}")
    
    return metrics

def load_model(model: nn.Module, checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load model from checkpoint."""
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    logger.info(f"Model loaded from {checkpoint_path}")
    return model

def save_model(model: nn.Module, save_path: str):
    """Save model to file."""
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")
