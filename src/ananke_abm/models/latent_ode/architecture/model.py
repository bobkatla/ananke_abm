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

        # We assume specific dimensions in the embeddings correspond to concepts
        self.STAY_MODE_DIM = 0
        self.TRAVEL_PURPOSE_DIM = 5 # Last purpose in the list

        # Core network to learn the main dynamics (acceleration)
        if self.config.use_second_order_sde:
            net_input_dim = self.state_dim + 2 # [p, v, t]
            net_output_dim = self.position_dim # learns acceleration
        else:
            net_input_dim = self.state_dim + 2 # [p, t]
            net_output_dim = self.position_dim # learns velocity
            
        self.net = nn.Sequential(
            nn.Linear(net_input_dim, hidden_dim),
            nn.ReLU(),
            *[ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)],
            nn.Linear(hidden_dim, net_output_dim)
        )
        
        # Required for torchsde compatibility
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def _calculate_potential(self, position):
        """Calculates a penalty potential for forbidden state combinations based on position."""
        slice_dims = [self.config.hidden_dim, self.config.zone_embed_dim, self.config.latent_purpose_embed_dim, self.config.latent_mode_embed_dim]
        _, _, purpose_embed, mode_embed = torch.split(position, slice_dims, dim=-1)

        activations_purp = torch.tanh(purpose_embed)
        activations_mode = torch.tanh(mode_embed)

        travel_purp_activation = activations_purp[..., self.TRAVEL_PURPOSE_DIM]
        stay_mode_activation = activations_mode[..., self.STAY_MODE_DIM]

        other_purp_dims = [i for i in range(activations_purp.shape[-1]) if i != self.TRAVEL_PURPOSE_DIM]
        other_purp_activation = activations_purp[..., other_purp_dims].mean(dim=-1)

        other_mode_dims = [i for i in range(activations_mode.shape[-1]) if i != self.STAY_MODE_DIM]
        other_mode_activation = activations_mode[..., other_mode_dims].mean(dim=-1)

        contradiction_1 = travel_purp_activation + stay_mode_activation - 1.5
        potential_1 = (torch.relu(contradiction_1))**2

        contradiction_2 = other_purp_activation + other_mode_activation - 1.5
        potential_2 = (torch.relu(contradiction_2))**2
        
        contradiction_3 = travel_purp_activation - other_mode_activation - 1.0 
        potential_3 = (torch.relu(contradiction_3))**2

        total_potential = potential_1 + potential_2 + potential_3
        return total_potential

    def forward(self, t, z):
        """Calculates dz/dt = f(z,t) with dynamic correction."""
        z.requires_grad_(True)
        
        # Time features
        batch_size = z.shape[0]
        if t.dim() == 0: t = t.expand(1)
        t_vec = torch.cat([torch.sin(t * 2 * torch.pi / 24), torch.cos(t * 2 * torch.pi / 24)]).expand(batch_size, -1)

        if self.config.use_second_order_sde:
            p, v = torch.split(z, self.position_dim, dim=-1)
            dp_dt = v
            dv_dt = self.net(torch.cat([p, v, t_vec], dim=-1))
            
            # Apply constraint as a corrective acceleration
            potential = self._calculate_potential(p)
            if torch.any(potential > 0):
                constraint_accel = -torch.autograd.grad(potential.sum(), p, create_graph=True)[0]
                final_dv_dt = dv_dt + self.config.correction_strength * constraint_accel
            else:
                final_dv_dt = dv_dt

            return torch.cat([dp_dt, final_dv_dt], dim=-1)

        else: # First-order system
            p = z
            dp_dt = self.net(torch.cat([p, t_vec], dim=-1))

            # Apply constraint as a corrective velocity
            potential = self._calculate_potential(p)
            if torch.any(potential > 0):
                constraint_vel = -torch.autograd.grad(potential.sum(), p, create_graph=True)[0]
                final_dp_dt = dp_dt + self.config.correction_strength * constraint_vel
            else:
                final_dp_dt = dp_dt
            
            return final_dp_dt

    def g(self, t, z):
        """Diffusion function for SDE (noise term)."""
        return torch.full_like(z, self.config.sde_noise_strength)

    def f(self, t, z):
        """Drift function for SDE (just a wrapper for forward)."""
        return self.forward(t, z)

class GenerativeODE(nn.Module):
    """
    A conditional, autoregressive Latent SDE model where the latent state
    includes embeddings for purpose and mode, which are evolved directly.
    """
    def __init__(self, person_feat_dim, num_zone_features, config):
        super().__init__()
        self.config = config
        
        self.zone_feature_encoder = nn.Linear(num_zone_features, config.zone_embed_dim)

        # The encoder produces the initial latent state z(0)
        # It now creates mu and log_var for the hidden state h,
        # plus initial embeddings for purpose and mode.
        encoder_input_dim = person_feat_dim + config.zone_embed_dim * 2
        encoder_output_dim = (config.hidden_dim * 2) + config.latent_purpose_embed_dim + config.latent_mode_embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, config.encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.encoder_hidden_dim, encoder_output_dim),
        )
        
        # Define the dimensions of the new SDE state components
        self.position_dim = config.hidden_dim + config.zone_embed_dim + config.latent_purpose_embed_dim + config.latent_mode_embed_dim
        
        # If using a second-order system, the state is [position, velocity]
        if config.use_second_order_sde:
            self.state_dim = self.position_dim * 2
        else:
            self.state_dim = self.position_dim
        
        self.ode_func = ODEFunc(
            config=config,
            state_dim=self.state_dim,
            position_dim=self.position_dim,
            hidden_dim=config.ode_hidden_dim,
            num_residual_blocks=config.num_residual_blocks
        )
        
        # --- New Decoders ---
        # These decode the final latent embeddings from the SDE into logits
        self.decoder_loc = nn.Linear(config.hidden_dim, config.zone_embed_dim)
        self.decoder_purpose = nn.Linear(config.latent_purpose_embed_dim, len(config.purpose_groups))
        self.decoder_mode = nn.Linear(config.latent_mode_embed_dim, config.num_modes)

    def forward(self, person_features, home_zone_features, work_zone_features, 
                times, all_zone_features):
        
        # Create candidate embeddings for all zones in the geography
        candidate_zone_embeds = self.zone_feature_encoder(all_zone_features)

        # Encode home and work zones from their features
        home_embed = self.zone_feature_encoder(home_zone_features)
        work_embed = self.zone_feature_encoder(work_zone_features)
        
        # 1. ENCODER: Get initial latent "position" p(0) from static features
        encoder_input = torch.cat([person_features, home_embed, work_embed], dim=-1)
        encoder_output = self.encoder(encoder_input)
        
        # Split the encoder output into its parts
        h0_mu, h0_log_var, initial_purpose_embed, initial_mode_embed = torch.split(
            encoder_output,
            [self.config.hidden_dim, self.config.hidden_dim, self.config.latent_purpose_embed_dim, self.config.latent_mode_embed_dim],
            dim=-1
        )
        h0 = h0_mu + torch.exp(0.5 * h0_log_var) * torch.randn_like(h0_mu)
        
        # Assemble the initial position state p(0)
        p0_loc = home_embed
        p0 = torch.cat([h0, p0_loc, initial_purpose_embed, initial_mode_embed], dim=-1)

        # 2. INITIAL STATE: Construct the full initial state s(0)
        if self.config.use_second_order_sde:
            # State is [p0, v0], where v0 is initialized to zero
            v0 = torch.zeros_like(p0)
            s0 = torch.cat([p0, v0], dim=-1)
        else:
            s0 = p0

        # 3. SOLVER: Evolve the state s(t) over the entire time series
        with torch.enable_grad():
            if self.config.enable_sde:
                pred_s = sdeint(self.ode_func, s0, times, method='euler', dt=0.01).permute(1, 0, 2)
            else:
                pred_s = odeint(self.ode_func, s0, times, method=self.config.ode_method).permute(1, 0, 2)
        
        # 4. DECODE: Extract the position component from the solved state path
        if self.config.use_second_order_sde:
            pred_p, _ = torch.split(pred_s, self.position_dim, dim=-1)
        else:
            pred_p = pred_s

        # Split the position path back into its components
        slice_dims = [self.config.hidden_dim, self.config.zone_embed_dim, self.config.latent_purpose_embed_dim, self.config.latent_mode_embed_dim]
        pred_h, pred_y_loc_embed, pred_purpose_embed, pred_mode_embed = torch.split(pred_p, slice_dims, dim=-1)
        
        # 5. PREDICT: Use the hidden state and latent embeddings to get final logits
        # Location prediction
        target_loc_embeds = self.decoder_loc(pred_h)
        pred_y_loc_logits = torch.einsum('bsd,zd->bsz', target_loc_embeds, candidate_zone_embeds)
        
        # Purpose and Mode prediction
        pred_y_purp_logits = self.decoder_purpose(pred_purpose_embed)
        pred_y_mode_logits = self.decoder_mode(pred_mode_embed)
        
        return pred_y_loc_logits, pred_y_loc_embed, pred_y_purp_logits, pred_y_mode_logits, h0_mu, h0_log_var 