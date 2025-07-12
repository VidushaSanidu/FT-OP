import argparse
import logging
import os
import sys
import copy
import json
from collections import OrderedDict

import torch
import torch.optim as optim

from data.loader import data_loader
from models.DO_TP_model import DO_TP
from utils import get_dset_path

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

# --- Configuration ---
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
    
    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self.__dict__.update(self)
    
    def __deepcopy__(self, memo):
        return AttrDict(copy.deepcopy(dict(self), memo))

args = AttrDict({
    'obs_len': 8, 'pred_len': 12, 'skip': 1, 'delim': '\t', 'loader_num_workers': 0,
    'batch_size': 32, 'learning_rate': 1e-3,
    'enc_hidden_dim': 32, 'dest_dim': 32, 'kl_beta': 0.1,
    'global_rounds': 10,
    'local_epochs': 7,
    'clients_per_round': 5
})

# --- Data Preparation ---
def prepare_clients():
    logger.info("Preparing client data loaders...")
    client_names = ['eth', 'hotel', 'zara1', 'zara2', 'univ']
    train_loaders, train_dsets = {}, {}

    for name in client_names:
        client_args = copy.deepcopy(args)
        client_args.dataset_name = name
        dset_path = get_dset_path(name, 'train')
        dset, loader = data_loader(client_args, dset_path)
        train_dsets[name] = dset
        train_loaders[name] = loader
        logger.info(f"Client '{name}' loaded with {len(dset)} samples.")
    
    return client_names, train_loaders, train_dsets

def prepare_validation_loader():
    logger.info("Preparing validation loader...")
    val_args = copy.deepcopy(args)
    val_args.dataset_name = 'zara1'
    dset_path = get_dset_path('zara1', 'val')
    dset, loader = data_loader(val_args, dset_path)
    logger.info(f"Validation set loaded with {len(dset)} samples.")
    return loader

# --- Training/Evaluation ---
def client_update(client_model, optimizer, train_loader, epochs):
    client_model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            batch = [tensor.to(device) for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, _, _, _) = batch
            
            optimizer.zero_grad()
            pred_disp, d_i, d_hat_i = client_model(obs_traj, pred_traj_gt)
            total_loss, _, _ = client_model.compute_loss(pred_disp, pred_traj_gt_rel, d_i, d_hat_i)
            total_loss.backward()
            optimizer.step()
    return client_model.state_dict()

def evaluate_model(model, val_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = [tensor.to(device) for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, _, _, _) = batch
            pred_disp, d_i, d_hat_i = model(obs_traj, pred_traj_gt)
            loss, _, _ = model.compute_loss(pred_disp, pred_traj_gt_rel, d_i, d_hat_i)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    logger.info(f"[Validation] Avg Loss: {avg_loss:.4f}")
    return avg_loss

# --- Main Federated Training Loop ---
def main():
    client_names, train_loaders, train_dsets = prepare_clients()
    val_loader = prepare_validation_loader()
    validation_losses = []

    global_model = DO_TP(
        obs_len=args.obs_len, pred_len=args.pred_len,
        enc_hidden_dim=args.enc_hidden_dim, dest_dim=args.dest_dim, kl_beta=args.kl_beta
    ).to(device)

    for round_num in range(args.global_rounds):
        logger.info(f"--- Global Round {round_num + 1} / {args.global_rounds} ---")
        
        selected_clients = client_names[:args.clients_per_round]
        local_weights = []
        total_data_points = sum([len(train_dsets[name]) for name in selected_clients])
        client_weights = {name: len(train_dsets[name]) / total_data_points for name in selected_clients}

        for client in selected_clients:
            logger.info(f"-> Training client: {client}")
            local_model = copy.deepcopy(global_model)
            optimizer = optim.Adam(local_model.parameters(), lr=args.learning_rate)

            updated_weights = client_update(
                client_model=local_model,
                optimizer=optimizer,
                train_loader=train_loaders[client],
                epochs=args.local_epochs
            )
            local_weights.append((client_weights[client], updated_weights))

        logger.info("Aggregating model updates...")
        global_weights = OrderedDict()
        for key in global_model.state_dict().keys():
            global_weights[key] = torch.zeros_like(global_model.state_dict()[key])
        for weight, local_state_dict in local_weights:
            for key in global_weights.keys():
                global_weights[key] += weight * local_state_dict[key]
        global_model.load_state_dict(global_weights)

        val_loss = evaluate_model(global_model, val_loader)
        validation_losses.append(val_loss)

        model_path = f"do_tp_federated_round_{round_num + 1}.pt"
        torch.save(global_model.state_dict(), model_path)
        logger.info(f"Saved model at: {model_path}")

    logger.info("Federated Training Complete!")
    torch.save(global_model.state_dict(), "do_tp_federated_final.pt")
    logger.info("Final federated model saved.")

    with open("val_loss_history.json", "w") as f:
        json.dump(validation_losses, f)
    logger.info("Validation loss history saved to val_loss_history.json")

if __name__ == '__main__':
    main()
