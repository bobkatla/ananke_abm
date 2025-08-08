"""
Model architecture for the Generative Latent ODE, updated for segment-based mode prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsde import sdeint

from ananke_abm.data_generator.feature_engineering import get_mode_features, MODE_ID_MAP

class ResidualBlock(nn.Module):
    """A residual block with two linear layers and a skip connection."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(x + self.net(x))

class ODEFunc(nn.Module):
    """
    The dynamics function for the SDE. Models dv/dt = a(p,v,h,t).
    The state p = [loc_emb, purp_emb] is now simpler.
    """
    def __init__(self, config, state_dim, position_dim, hidden_dim, num_residual_blocks):
        super().__init__()
        self.config = config
        self.state_dim = state_dim

        # Input to the dynamics network: [p, v, h, t]
        net_input_dim = state_dim + config.hidden_dim + 2 
        net_output_dim = position_dim # Learns acceleration dv/dt
            
        self.net = nn.Sequential(
            nn.Linear(net_input_dim, hidden_dim),
            nn.ReLU(),
            *[ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)],
            nn.Linear(hidden_dim, net_output_dim),
            nn.Tanh() # Constrain acceleration output
        )
        
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def forward(self, t, y):
        """Calculates dy/dt for the combined state y = [state, h]."""
        state, h = torch.split(y, [self.state_dim, self.config.hidden_dim], dim=-1)
        
        batch_size = state.shape[0]
        t_vec = torch.stack([torch.sin(t * 2 * torch.pi / 24), torch.cos(t * 2 * torch.pi / 24)]).expand(batch_size, -1)

        p, v = torch.split(state, self.state_dim // 2, dim=-1)
        dp_dt = v
        dv_dt = self.net(torch.cat([p, v, h, t_vec], dim=-1))
        d_state = torch.cat([dp_dt, dv_dt], dim=-1)
        
        d_h = torch.zeros_like(h)
        return torch.cat([d_state, d_h], dim=-1)

    def g(self, t, y):
        """Diffusion function for SDE."""
        state_noise = torch.full((y.shape[0], self.state_dim), self.config.sde_noise_strength, device=y.device)
        h_noise = torch.zeros((y.shape[0], self.config.hidden_dim), device=y.device)
        return torch.cat([state_noise, h_noise], dim=-1)

    def f(self, t, y):
        """Drift function for SDE."""
        return self.forward(t, y)

class ModeFromPathHead(nn.Module):
    """
    Predicts travel mode from latent path segment descriptors.
    """
    def __init__(self, config, position_dim):
        super().__init__()
        self.config = config
        
        # Descriptors: delta_t, arc_length, mean_velocity, mean_speed, speed_variance, heading_change_rate
        descriptor_dim = 2 + position_dim + 3
        
        self.mlp = nn.Sequential(
            nn.Linear(descriptor_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.mode_feature_dim),
            nn.Tanh() # Constrain the output space
        )

        # Register mode prototypes as non-trainable buffers
        mode_protos = torch.stack([get_mode_features(mid) for mid in sorted(MODE_ID_MAP.values())])
        self.register_buffer("mode_prototypes", mode_protos)

    def forward(self, p_slice, v_slice, t_slice):
        # 1. Compute Descriptors
        delta_t = t_slice[-1] - t_slice[0] # Duration in hours
        arc_length = torch.sum(torch.norm(p_slice[1:] - p_slice[:-1], p=2, dim=-1))
        
        # --- New Richer Descriptors ---
        v_bar = v_slice.mean(dim=0) # Mean velocity vector
        
        speeds = torch.norm(v_slice, p=2, dim=-1)
        speed_bar = speeds.mean()
        speed_variance = torch.var(speeds)

        # Calculate heading change rate
        v_norm = F.normalize(v_slice, p=2, dim=-1, eps=1e-8) # Use eps for safety
        # Clamp to avoid acos errors from floating point inaccuracies and ensure safe gradient
        dot_products = torch.clamp((v_norm[:-1] * v_norm[1:]).sum(dim=-1), -1.0 + 1e-6, 1.0 - 1e-6)
        heading_changes = torch.acos(dot_products)
        total_heading_change = heading_changes.sum()
        heading_change_rate = total_heading_change / delta_t.clamp(min=1e-6)

        # Ensure all descriptors are tensors with one element for concatenation
        if speed_bar.dim() == 0: speed_bar = speed_bar.unsqueeze(0)
        if speed_variance.dim() == 0: speed_variance = speed_variance.unsqueeze(0)
        if heading_change_rate.dim() == 0: heading_change_rate = heading_change_rate.unsqueeze(0)
            
        x_seg = torch.cat([
            delta_t.unsqueeze(0),
            arc_length.unsqueeze(0),
            v_bar, 
            speed_bar,
            speed_variance,
            heading_change_rate
        ])
        
        # 2. Map descriptor to mode feature space
        h = self.mlp(x_seg)
        
        # 3. Score against prototypes
        # Logits are negative squared distance: shape (num_modes,)
        logits = -torch.sum((h.unsqueeze(0) - self.mode_prototypes).pow(2), dim=1)
        
        return logits, h

class GenerativeODE(nn.Module):
    """
    Conditional Latent SDE model with state = [loc_emb, purp_emb]
    and a separate head for segment-based mode prediction.
    """
    def __init__(self, person_feat_dim, num_zone_features, config):
        super().__init__()
        self.config = config
        
        self.zone_feature_encoder = nn.Linear(num_zone_features, config.zone_embed_dim)

        # Encoder input: static features + initial purpose
        encoder_input_dim = person_feat_dim + config.zone_embed_dim * 2 + config.purpose_feature_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, config.encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.encoder_hidden_dim, config.hidden_dim * 2),
        )
        
        self.position_dim = config.zone_embed_dim + config.purpose_feature_dim
        self.state_dim = self.position_dim * 2  # Second-order system: [p, v]
        
        self.ode_func = ODEFunc(
            config=config,
            state_dim=self.state_dim,
            position_dim=self.position_dim,
            hidden_dim=config.ode_hidden_dim,
            num_residual_blocks=config.num_residual_blocks
        )
        
        # Decoders for snap-level predictions (state only)
        self.decoder_loc = nn.Linear(config.zone_embed_dim, config.zone_embed_dim)
        self.decoder_purpose = nn.Linear(config.purpose_feature_dim, len(config.purpose_groups))
        
        # New head for segment-level mode prediction
        self.mode_predictor = ModeFromPathHead(config, self.position_dim)

    def forward(self, person_features, home_zone_features, work_zone_features, 
                initial_purpose_features, times, all_zone_features):
        
        home_embed = self.zone_feature_encoder(home_zone_features)
        work_embed = self.zone_feature_encoder(work_zone_features)
        
        # 1. ENCODER: Get initial hidden state h(0)
        encoder_input = torch.cat([person_features, home_embed, work_embed, initial_purpose_features], dim=-1)
        h0_mu, h0_log_var = self.encoder(encoder_input).split(self.config.hidden_dim, dim=-1)
        h0 = h0_mu + torch.exp(0.5 * h0_log_var) * torch.randn_like(h0_mu)
        
        # 2. INITIAL STATE: p(0) = [loc_emb, purp_emb], v(0) = 0
        p0 = torch.cat([home_embed, initial_purpose_features], dim=-1)
        s0 = torch.cat([p0, torch.zeros_like(p0)], dim=-1)

        y0 = torch.cat([s0, h0], dim=-1)
        
        # 3. SOLVER: Evolve the combined state y(t)
        options = {'dtype': torch.float32}
        pred_y_path = sdeint(self.ode_func, y0, times, method='euler', dt=1.0, options=options).permute(1, 0, 2)
        
        # 4. DECODE SNAPS: Split path and decode location/purpose at snap times
        pred_s, _ = torch.split(pred_y_path, [self.state_dim, self.config.hidden_dim], dim=-1)
        pred_p, pred_v = torch.split(pred_s, self.position_dim, dim=-1)
        
        slice_dims = [self.config.zone_embed_dim, self.config.purpose_feature_dim]
        pred_loc_embed, pred_purpose_features = torch.split(pred_p, slice_dims, dim=-1)
        
        candidate_zone_embeds = self.zone_feature_encoder(all_zone_features)
        target_loc_embeds = self.decoder_loc(pred_loc_embed)

        # --- L2 Normalize embeddings for stable logit calculation ---
        target_loc_embeds_norm = F.normalize(target_loc_embeds, p=2, dim=-1)
        candidate_zone_embeds_norm = F.normalize(candidate_zone_embeds, p=2, dim=-1)
        
        # Logits are now based on cosine similarity
        pred_loc_logits = torch.matmul(target_loc_embeds_norm, candidate_zone_embeds_norm.t())
        
        # --- L2 Normalize purpose features for stable logit calculation ---
        pred_purpose_features_norm = F.normalize(pred_purpose_features, p=2, dim=-1)
        pred_purp_logits = self.decoder_purpose(pred_purpose_features_norm)
        
        return (
            pred_loc_logits, pred_loc_embed, 
            pred_purp_logits, pred_purpose_features,
            pred_p, pred_v, # Return full path for segment processing
            h0_mu, h0_log_var
        )

    def predict_mode_from_segments(self, pred_p, pred_v, grid_times, segments_batch):
        """Processes a batch of segments to predict travel modes."""
        all_logits = []
        all_h = []

        for seg in segments_batch:
            b, i0, i1 = seg['b'], seg['i0'], seg['i1']
            
            p_slice = pred_p[b, i0:i1+1]
            v_slice = pred_v[b, i0:i1+1]
            t_slice = grid_times[i0:i1+1]

            logits, h = self.mode_predictor(p_slice, v_slice, t_slice)
            all_logits.append(logits)
            all_h.append(h)
        
        if not all_logits: # Handle cases with no travel segments in batch
            return torch.empty(0, self.config.num_modes), torch.empty(0, self.config.mode_feature_dim)

        return torch.stack(all_logits), torch.stack(all_h)
