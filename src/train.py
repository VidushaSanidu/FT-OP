import argparse
import logging
import os
import sys

import torch
import torch.optim as optim

from data.loader import data_loader
from models.DO_TP_model import DO_TP
from utils import get_dset_path



torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Argument Parsing ---
# Dataset options
parser.add_argument('--dataset_name', default='zara1', type=str)
parser.add_argument('--delim', default='\t', type=str)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=12, type=int)
parser.add_argument('--skip', default=1, type=int)
parser.add_argument('--loader_num_workers', default=4, type=int)

# Optimization
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--learning_rate', default=1e-3, type=float)

# Model Options for DO-TP
parser.add_argument('--enc_hidden_dim', default=32, type=int)
parser.add_argument('--dest_dim', default=32, type=int)
parser.add_argument('--kl_beta', default=0.1, type=float)

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=50, type=int)
parser.add_argument('--checkpoint_name', default='do_tp_checkpoint', type=str)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--gpu_num', default="0", type=str)

def get_device_and_dtypes(args):
    device = torch.device('cpu')
    if args.use_gpu == 1 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_num}')
    
    long_dtype = torch.long
    float_dtype = torch.float32
    return device, long_dtype, float_dtype

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = get_dset_path(args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_name, 'val')
    device, long_dtype, float_dtype = get_device_and_dtypes(args)

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    val_dset, val_loader = data_loader(args, val_path)

    # --- Model Definition ---
    model = DO_TP(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        enc_hidden_dim=args.enc_hidden_dim,
        dest_dim=args.dest_dim,
        kl_beta=args.kl_beta
    )
    model.to(device).train()
    logger.info('Here is the DO-TP model:')
    logger.info(model)

    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # --- Training Loop ---
    for epoch in range(args.num_epochs):
        logger.info('Starting epoch {}'.format(epoch + 1))
        for i, batch in enumerate(train_loader):
            batch = [tensor.to(device) for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
             loss_mask, seq_start_end) = batch
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_disp, d_i, d_hat_i = model(obs_traj, pred_traj_gt)
            
            # Compute loss
            total_loss, traj_loss, kl_loss = model.compute_loss(
                pred_disp, pred_traj_gt_rel, d_i, d_hat_i
            )

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            
            if (i + 1) % args.print_every == 0:
                logger.info(
                    f'Epoch [{epoch+1}/{args.num_epochs}], Batch [{i+1}/{len(train_loader)}], '
                    f'Total Loss: {total_loss.item():.4f}, '
                    f'Traj Loss: {traj_loss.item():.4f}, '
                    f'KL Loss: {kl_loss.item():.4f}'
                )

    # --- Save final model ---
    checkpoint_path = os.path.join(
        args.output_dir, f'{args.checkpoint_name}_final.pt'
    )
    logger.info(f'Saving final model to {checkpoint_path}')
    torch.save(model.state_dict(), checkpoint_path)
    logger.info('Done.')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)