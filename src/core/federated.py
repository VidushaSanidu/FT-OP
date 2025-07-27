"""
Federated learning implementation for DO-TP model.
"""

import copy
import logging
import os
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from core.training import TrainingMetrics, create_optimizer, get_device
from models.DO_TP_model import DO_TP

logger = logging.getLogger(__name__)

class FederatedTrainer:
    """Handles federated learning training process."""
    
    def __init__(self, config, client_loaders: Dict[str, Tuple[any, DataLoader]], client_val_loaders: Dict[str, Tuple[any, DataLoader]]):
        self.config = config
        self.client_loaders = client_loaders
        self.client_val_loaders = client_val_loaders
        self.device = get_device(config.training.use_gpu, config.training.gpu_num)
        
        # Calculate client weights based on data size
        total_samples = sum(len(dset) for dset, _ in client_loaders.values())
        self.client_weights = {
            name: len(dset) / total_samples 
            for name, (dset, _) in client_loaders.items()
        }
        
        logger.info(f"Federated trainer initialized with {len(client_loaders)} clients")
        logger.info(f"Client weights: {self.client_weights}")
        logger.info(f"Client validation datasets: {list(client_val_loaders.keys())}")
    
    def create_global_model(self) -> DO_TP:
        """Create and initialize the global model."""
        model = DO_TP(
            obs_len=self.config.model.obs_len,
            pred_len=self.config.model.pred_len,
            input_dim=self.config.model.input_dim,
            enc_hidden_dim=self.config.model.enc_hidden_dim,
            dest_dim=self.config.model.dest_dim,
            kl_beta=self.config.model.kl_beta
        )
        model.to(self.device)
        return model
    
    def client_update(
        self, 
        global_model: DO_TP, 
        client_loader: DataLoader, 
        epochs: int
    ) -> Dict[str, torch.Tensor]:
        """Perform local training on a client."""
        # Create local model copy
        local_model = copy.deepcopy(global_model)
        local_model.train()
        
        optimizer = create_optimizer(local_model, self.config.training.learning_rate)
        
        for epoch in range(epochs):
            for batch in client_loader:
                batch = [tensor.to(self.device) for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, 
                 non_linear_ped, loss_mask, seq_start_end) = batch
                
                optimizer.zero_grad()
                pred_disp, d_i, d_hat_i = local_model(obs_traj, pred_traj_gt)
                total_loss, _, _ = local_model.compute_loss(pred_disp, pred_traj_gt_rel, d_i, d_hat_i)
                total_loss.backward()
                optimizer.step()
        
        return local_model.state_dict()
    
    def aggregate_weights(
        self, 
        local_weights: List[Tuple[str, Dict[str, torch.Tensor]]], 
        global_model: DO_TP
    ) -> Dict[str, torch.Tensor]:
        """Aggregate local model weights using weighted averaging."""
        if self.config.federated.aggregation_method == "simple_avg":
            return self._simple_average(local_weights, global_model)
        else:  # weighted_avg
            return self._weighted_average(local_weights, global_model)
    
    def _weighted_average(
        self, 
        local_weights: List[Tuple[str, Dict[str, torch.Tensor]]], 
        global_model: DO_TP
    ) -> Dict[str, torch.Tensor]:
        """Perform weighted averaging of local weights."""
        global_weights = OrderedDict()
        
        # Initialize with zeros
        for key in global_model.state_dict().keys():
            global_weights[key] = torch.zeros_like(global_model.state_dict()[key])
        
        # Weighted sum
        for client_name, local_state_dict in local_weights:
            weight = self.client_weights[client_name]
            for key in global_weights.keys():
                global_weights[key] += weight * local_state_dict[key]
        
        return global_weights
    
    def _simple_average(
        self, 
        local_weights: List[Tuple[str, Dict[str, torch.Tensor]]], 
        global_model: DO_TP
    ) -> Dict[str, torch.Tensor]:
        """Perform simple averaging of local weights."""
        global_weights = OrderedDict()
        
        # Initialize with zeros
        for key in global_model.state_dict().keys():
            global_weights[key] = torch.zeros_like(global_model.state_dict()[key])
        
        # Simple average
        for client_name, local_state_dict in local_weights:
            for key in global_weights.keys():
                global_weights[key] += local_state_dict[key]
        
        # Divide by number of clients
        num_clients = len(local_weights)
        for key in global_weights.keys():
            global_weights[key] /= num_clients
        
        return global_weights
    
    def evaluate_global_model(self, model: DO_TP) -> Tuple[float, float, float]:
        """Evaluate the global model on validation data."""
        model.eval()
        total_loss = 0.0
        total_ade = 0.0
        total_fde = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = [tensor.to(self.device) for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, 
                 non_linear_ped, loss_mask, seq_start_end) = batch
                
                pred_disp, d_i, d_hat_i = model(obs_traj, pred_traj_gt)
                loss, _, _ = model.compute_loss(pred_disp, pred_traj_gt_rel, d_i, d_hat_i)
                
                # Calculate prediction trajectory
                pred_traj = pred_disp + obs_traj[:, -1].unsqueeze(1)
                
                # Calculate metrics
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
    
    def evaluate_client_local_validation(self, model: DO_TP, client_name: str) -> Tuple[float, float, float]:
        """Evaluate the global model on a specific client's local validation data."""
        if client_name not in self.client_val_loaders:
            logger.warning(f"No validation data available for client {client_name}")
            return float('inf'), float('inf'), float('inf')
            
        model.eval()
        _, val_loader = self.client_val_loaders[client_name]
        
        total_loss = 0.0
        total_ade = 0.0
        total_fde = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = [tensor.to(self.device) for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, 
                 non_linear_ped, loss_mask, seq_start_end) = batch
                
                pred_disp, d_i, d_hat_i = model(obs_traj, pred_traj_gt)
                loss, _, _ = model.compute_loss(pred_disp, pred_traj_gt_rel, d_i, d_hat_i)
                
                # Calculate prediction trajectory
                pred_traj = pred_disp + obs_traj[:, -1].unsqueeze(1)
                
                # Calculate metrics
                batch_size = pred_traj.size(0)
                ade = torch.norm(pred_traj - pred_traj_gt, dim=2).mean().item()
                fde = torch.norm(pred_traj[:, -1] - pred_traj_gt[:, -1], dim=1).mean().item()
                
                total_loss += loss.item() * batch_size
                total_ade += ade * batch_size
                total_fde += fde * batch_size
                total_samples += batch_size
        
        if total_samples == 0:
            return float('inf'), float('inf'), float('inf')
            
        avg_loss = total_loss / total_samples
        avg_ade = total_ade / total_samples
        avg_fde = total_fde / total_samples
        
        return avg_loss, avg_ade, avg_fde
    
    def evaluate_all_clients_local_validation(self, model: DO_TP) -> Dict[str, Tuple[float, float, float]]:
        """Evaluate the global model on all clients' local validation data."""
        client_metrics = {}
        
        for client_name in self.client_loaders.keys():
            val_loss, val_ade, val_fde = self.evaluate_client_local_validation(model, client_name)
            client_metrics[client_name] = (val_loss, val_ade, val_fde)
            logger.info(f"Client {client_name} local validation - Loss: {val_loss:.4f}, ADE: {val_ade:.4f}, FDE: {val_fde:.4f}")
        
        return client_metrics
    
    def train_federated(self, save_checkpoints: bool = True) -> TrainingMetrics:
        """Main federated training loop."""
        global_model = self.create_global_model()
        metrics = TrainingMetrics()
        
        # Add lists to track per-client metrics
        metrics.client_val_losses = {client: [] for client in self.client_loaders.keys()}
        metrics.client_val_ades = {client: [] for client in self.client_loaders.keys()}
        metrics.client_val_fdes = {client: [] for client in self.client_loaders.keys()}
        
        logger.info(f"Starting federated training for {self.config.federated.global_rounds} rounds")
        
        for round_num in range(self.config.federated.global_rounds):
            logger.info(f"--- Global Round {round_num + 1}/{self.config.federated.global_rounds} ---")
            
            # Select clients for this round
            selected_clients = list(self.client_loaders.keys())[:self.config.federated.clients_per_round]
            local_weights = []
            
            # Local training on selected clients
            for client_name in selected_clients:
                logger.info(f"Training client: {client_name}")
                _, client_loader = self.client_loaders[client_name]
                
                updated_weights = self.client_update(
                    global_model=global_model,
                    client_loader=client_loader,
                    epochs=self.config.federated.local_epochs
                )
                local_weights.append((client_name, updated_weights))
            
            # Aggregate weights
            logger.info("Aggregating model updates...")
            aggregated_weights = self.aggregate_weights(local_weights, global_model)
            global_model.load_state_dict(aggregated_weights)
            
            # Evaluate global model on each client's local validation data
            logger.info("Evaluating global model on client local validation data...")
            client_metrics = self.evaluate_all_clients_local_validation(global_model)
            
            # Calculate weighted average metrics across all clients
            total_weighted_loss = 0.0
            total_weighted_ade = 0.0
            total_weighted_fde = 0.0
            total_weight = 0.0
            
            for client_name, (val_loss, val_ade, val_fde) in client_metrics.items():
                weight = self.client_weights[client_name]
                total_weighted_loss += val_loss * weight
                total_weighted_ade += val_ade * weight
                total_weighted_fde += val_fde * weight
                total_weight += weight
                
                # Store per-client metrics
                metrics.client_val_losses[client_name].append(val_loss)
                metrics.client_val_ades[client_name].append(val_ade)
                metrics.client_val_fdes[client_name].append(val_fde)
            
            # Calculate global weighted averages
            global_val_loss = total_weighted_loss / total_weight if total_weight > 0 else float('inf')
            global_val_ade = total_weighted_ade / total_weight if total_weight > 0 else float('inf')
            global_val_fde = total_weighted_fde / total_weight if total_weight > 0 else float('inf')
            
            # Store global metrics (weighted average of all clients)
            metrics.add_val_metrics(global_val_loss, global_val_ade, global_val_fde)
            
            logger.info(
                f"Round {round_num + 1} - Global Weighted Avg: Loss: {global_val_loss:.4f}, "
                f"ADE: {global_val_ade:.4f}, FDE: {global_val_fde:.4f}"
            )
            
            # Save checkpoint
            if save_checkpoints:
                models_dir = getattr(self.config, 'models_dir', os.path.join(self.config.output.output_dir, 'models'))
                checkpoint_path = os.path.join(models_dir, f"{self.config.output.checkpoint_name}_federated_round_{round_num + 1}.pt")
                torch.save(global_model.state_dict(), checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save final model
        models_dir = getattr(self.config, 'models_dir', os.path.join(self.config.output.output_dir, 'models'))
        final_path = os.path.join(models_dir, f"{self.config.output.checkpoint_name}_federated_final.pt")
        torch.save(global_model.state_dict(), final_path)
        logger.info(f"Final federated model saved: {final_path}")
        
        return global_model, metrics
    
    def get_client_statistics(self) -> Dict[str, Any]:
        """Get statistics about federated clients."""
        stats = {
            'num_clients': len(self.client_loaders),
            'client_names': list(self.client_loaders.keys()),
            'client_data_sizes': {name: len(dset) for name, (dset, _) in self.client_loaders.items()},
            'client_weights': self.client_weights,
            'total_samples': sum(len(dset) for dset, _ in self.client_loaders.values())
        }
        return stats

def create_federated_trainer(
    config, 
    client_loaders: Dict[str, Tuple[any, DataLoader]], 
    client_val_loaders: Dict[str, Tuple[any, DataLoader]]
) -> FederatedTrainer:
    """Factory function to create federated trainer."""
    return FederatedTrainer(config, client_loaders, client_val_loaders)
