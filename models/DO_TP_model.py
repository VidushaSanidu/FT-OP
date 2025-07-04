import torch
import torch.nn as nn

class DO_TP(nn.Module):
    """
    Implementation of the Destination-Oriented Trajectory Prediction (DO-TP) model
    as described in "A federated pedestrian trajectory prediction model with data privacy protection".
    """
    def __init__(self, obs_len=8, pred_len=12, input_dim=2, 
                 enc_hidden_dim=32, dest_dim=32, kl_beta=0.1):
        super(DO_TP, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.kl_beta = kl_beta
        
        # Encoder components
        self.linear_enc = nn.Linear(input_dim, 64)
        self.encoder_lstm = nn.LSTM(64, enc_hidden_dim, batch_first=True)
        
        # Destination prediction
        self.dest_lstm_obs = nn.LSTM(input_dim, enc_hidden_dim, batch_first=True)
        self.dest_lstm_fut = nn.LSTM(input_dim, enc_hidden_dim, batch_first=True)
        self.linear_dest_obs = nn.Linear(enc_hidden_dim, dest_dim)
        self.linear_dest_fut = nn.Linear(enc_hidden_dim, dest_dim)
        
        # Decoder components
        self.decoder_lstm = nn.LSTM(64, enc_hidden_dim, batch_first=True)
        self.linear_dec = nn.Linear(enc_hidden_dim, input_dim)
        self.linear_dec_input = nn.Linear(input_dim, 64)

    def forward(self, obs_traj, fut_traj=None):
        """
        Forward pass for the DO-TP model.
        """
        
        # 1. Compute relative displacements for the encoder input
        displacements = obs_traj[:, 1:] - obs_traj[:, :-1]
        zeros = torch.zeros_like(obs_traj[:, 0:1])
        displacements = torch.cat([zeros, displacements], dim=1)
        
        # 2. Encoder processing to get motion patterns
        enc_in = self.linear_enc(displacements)
        enc_out, (h_enc, c_enc) = self.encoder_lstm(enc_in)
        last_motion_feature = enc_out[:, -1]
        
        # 3. Destination prediction from the observed trajectory
        _, (h_dest_obs, _) = self.dest_lstm_obs(obs_traj)
        h_dest_obs = h_dest_obs.squeeze(0)
        D_i = self.linear_dest_obs(h_dest_obs)
        
        # 4. Destination prediction from the full trajectory during training
        D_hat_i = None
        if fut_traj is not None:
            full_traj = torch.cat([obs_traj, fut_traj], dim=1)
            _, (h_dest_fut, _) = self.dest_lstm_fut(full_traj)
            h_dest_fut = h_dest_fut.squeeze(0)
            D_hat_i = self.linear_dest_fut(h_dest_fut)
            
        # 5. Decoder initialization and autoregressive prediction
        pred_disp = []
        h_dec, c_dec = h_enc, c_enc
        dec_input = self.linear_dec_input(displacements[:, -1, :]).unsqueeze(1)

        for _ in range(self.pred_len):
            output, (h_dec, c_dec) = self.decoder_lstm(dec_input, (h_dec, c_dec))
            disp = self.linear_dec(output.squeeze(1))
            pred_disp.append(disp)
            dec_input = self.linear_dec_input(disp).unsqueeze(1)
            
        pred_disp = torch.stack(pred_disp, dim=1)
        
        return pred_disp, D_i, D_hat_i

    def compute_loss(self, pred_disp, gt_disp, D_i, D_hat_i):
        """
        Computes the total loss for the DO-TP model.
        """
        traj_loss = torch.mean((pred_disp - gt_disp)**2)
        
        kl_loss = torch.tensor(0.0, device=pred_disp.device)
        if D_hat_i is not None:
            kl_loss = nn.functional.kl_div(
                nn.functional.log_softmax(D_i, dim=-1),
                nn.functional.softmax(D_hat_i, dim=-1),
                reduction='batchmean'
            )
            
        total_loss = traj_loss + self.kl_beta * kl_loss
        return total_loss, traj_loss, kl_loss