"""
Model architecture for the Generative Latent ODE.
"""
import torch
import torch.nn as nn
from torchdiffeq import odeint
from torchsde import sdeint

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
    The dynamics function for the conditional SDE.
    If configured as a second-order system, it models acceleration dv/dt = a(p,v,t).
    Otherwise, it models velocity dp/dt = f(p,t).
    """
    def __init__(self, config, state_dim, position_dim, hidden_dim, num_residual_blocks):
        super().__init__()
        self.config = config
        self.state_dim = state_dim
        self.position_dim = position_dim

        # Define which column in the feature vectors corresponds to our physical constraints
        self.IS_MOVING_DIM = 0      # from mode features
        self.IS_STATIONARY_DIM = 0  # from purpose features

        # Core network to learn the main dynamics
        # The input is the full state [s] and the conditioning variable [h]
        if self.config.use_second_order_sde:
            # Input: [p, v, h, t]
            net_input_dim = self.state_dim + config.hidden_dim + 2 
            net_output_dim = self.position_dim # learns acceleration
        else:
            # Input: [p, h, t]
            net_input_dim = self.state_dim + config.hidden_dim + 2
            net_output_dim = self.position_dim # learns velocity
            
        self.net = nn.Sequential(
            nn.Linear(net_input_dim, hidden_dim),
            nn.ReLU(),
            *[ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)],
            nn.Linear(hidden_dim, net_output_dim)
        )
        
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def _calculate_potential(self, position):
        """
        Calculates a penalty potential for forbidden state combinations.
        The state is physically implausible if (is_moving and is_stationary) 
        or (not is_moving and not is_stationary).
        A simple potential is (is_moving - (1 - is_stationary))^2.
        """
        slice_dims = [
            self.config.zone_embed_dim, 
            self.config.purpose_feature_dim, 
            self.config.mode_feature_dim
        ]
        _, purpose_features, mode_features = torch.split(position, slice_dims, dim=-1)
        
        is_moving_prob = torch.sigmoid(mode_features[..., self.IS_MOVING_DIM])
        is_stationary_prob = torch.sigmoid(purpose_features[..., self.IS_STATIONARY_DIM])
        
        potential = (is_moving_prob - (1.0 - is_stationary_prob))**2
        return potential.sum()


    def forward(self, t, y):
        """
        Calculates dy/dt for the combined state y = [state, h].
        """
        y.requires_grad_(True)
        state, h = torch.split(y, [self.state_dim, self.config.hidden_dim], dim=-1)
        
        batch_size = state.shape[0]
        if t.dim() == 0: t = t.expand(1)
        t_vec = torch.cat([torch.sin(t * 2 * torch.pi / 24), torch.cos(t * 2 * torch.pi / 24)]).expand(batch_size, -1)

        if self.config.use_second_order_sde:
            p, v = torch.split(state, self.position_dim, dim=-1)
            dp_dt = v
            dv_dt = self.net(torch.cat([p, v, h, t_vec], dim=-1))
            
            potential = self._calculate_potential(p)
            if torch.any(potential > 0):
                constraint_accel = -torch.autograd.grad(potential.sum(), p, create_graph=True)[0]
                final_dv_dt = dv_dt + self.config.correction_strength * constraint_accel
            else:
                final_dv_dt = dv_dt
            
            d_state = torch.cat([dp_dt, final_dv_dt], dim=-1)

        else: # First-order system
            p = state
            dp_dt = self.net(torch.cat([p, h, t_vec], dim=-1))
            
            potential = self._calculate_potential(p)
            if torch.any(potential > 0):
                constraint_vel = -torch.autograd.grad(potential.sum(), p, create_graph=True)[0]
                final_dp_dt = dp_dt + self.config.correction_strength * constraint_vel
            else:
                final_dp_dt = dp_dt
            
            d_state = final_dp_dt
        
        # The conditioning variable h is constant, so its derivative is zero.
        d_h = torch.zeros_like(h)
        return torch.cat([d_state, d_h], dim=-1)

    def g(self, t, y):
        """Diffusion function for SDE. Noise is only applied to the state."""
        state, h = torch.split(y, [self.state_dim, self.config.hidden_dim], dim=-1)
        
        state_noise = torch.full_like(state, self.config.sde_noise_strength)
        h_noise = torch.zeros_like(h)
        
        return torch.cat([state_noise, h_noise], dim=-1)

    def f(self, t, y):
        """Drift function for SDE (just a wrapper for forward)."""
        return self.forward(t, y)

class GenerativeODE(nn.Module):
    """
    A conditional, autoregressive Latent SDE model where the latent state
    includes rich feature vectors for purpose and mode, which are evolved directly.
    """
    def __init__(self, person_feat_dim, num_zone_features, config):
        super().__init__()
        self.config = config
        
        self.zone_feature_encoder = nn.Linear(num_zone_features, config.zone_embed_dim)

        # The encoder produces the initial hidden state h(0)
        encoder_input_dim = person_feat_dim + config.zone_embed_dim * 2 + config.purpose_feature_dim + config.mode_feature_dim
        encoder_output_dim = config.hidden_dim * 2 # mu and log_var for h0
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, config.encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.encoder_hidden_dim, encoder_output_dim),
        )
        
        self.position_dim = config.zone_embed_dim + config.purpose_feature_dim + config.mode_feature_dim
        self.state_dim = self.position_dim * 2 if config.use_second_order_sde else self.position_dim
        
        self.ode_func = ODEFunc(
            config=config,
            state_dim=self.state_dim,
            position_dim=self.position_dim,
            hidden_dim=config.ode_hidden_dim,
            num_residual_blocks=config.num_residual_blocks
        )
        
        self.decoder_loc = nn.Linear(config.zone_embed_dim, config.zone_embed_dim)
        self.decoder_purpose = nn.Linear(config.purpose_feature_dim, len(config.purpose_groups))
        self.decoder_mode = nn.Linear(config.mode_feature_dim, config.num_modes)

    def forward(self, person_features, home_zone_features, work_zone_features, 
                initial_purpose_features, initial_mode_features,
                times, all_zone_features):
        
        candidate_zone_embeds = self.zone_feature_encoder(all_zone_features)
        home_embed = self.zone_feature_encoder(home_zone_features)
        work_embed = self.zone_feature_encoder(work_zone_features)
        
        # 1. ENCODER: Get initial hidden state h(0) from all static features
        encoder_input = torch.cat([
            person_features, home_embed, work_embed, 
            initial_purpose_features, initial_mode_features
        ], dim=-1)
        h0_mu, h0_log_var = self.encoder(encoder_input).split(self.config.hidden_dim, dim=-1)
        h0 = h0_mu + torch.exp(0.5 * h0_log_var) * torch.randn_like(h0_mu)
        
        # 2. INITIAL STATE: Construct the initial position p(0) and full state s(0)
        p0 = torch.cat([home_embed, initial_purpose_features, initial_mode_features], dim=-1)
        s0 = torch.cat([p0, torch.zeros_like(p0)], dim=-1) if self.config.use_second_order_sde else p0

        # Combine s0 and h0 into a single tensor for the solver
        y0 = torch.cat([s0, h0], dim=-1)
        
        # 3. SOLVER: Evolve the combined state y(t) over time
        with torch.enable_grad():
            options = {'dtype': torch.float32}
            if self.config.enable_sde:
                pred_y_path = sdeint(self.ode_func, y0, times, method='euler', dt=0.01, options=options)
            else:
                pred_y_path = odeint(self.ode_func, y0, times, method=self.config.ode_method, options=options)
        
        # Transpose to [batch, time, feature]
        pred_y = pred_y_path.permute(1, 0, 2)
        
        # 4. DECODE: Split the path back into the evolved state and the (constant) h
        pred_s, _ = torch.split(pred_y, [self.state_dim, self.config.hidden_dim], dim=-1)
        pred_p = torch.split(pred_s, self.position_dim, dim=-1)[0] if self.config.use_second_order_sde else pred_s
        
        slice_dims = [self.config.zone_embed_dim, self.config.purpose_feature_dim, self.config.mode_feature_dim]
        pred_y_loc_embed, pred_purpose_features, pred_mode_features = torch.split(pred_p, slice_dims, dim=-1)
        
        # 5. PREDICT: Get final logits for classification
        target_loc_embeds = self.decoder_loc(pred_y_loc_embed)
        pred_y_loc_logits = torch.einsum('bsd,zd->bsz', target_loc_embeds, candidate_zone_embeds)
        
        pred_y_purp_logits = self.decoder_purpose(pred_purpose_features)
        pred_y_mode_logits = self.decoder_mode(pred_mode_features)
        
        return (
            pred_y_loc_logits, pred_y_loc_embed, 
            pred_y_purp_logits, pred_y_mode_logits, 
            pred_purpose_features, pred_mode_features,
            h0_mu, h0_log_var
        )
