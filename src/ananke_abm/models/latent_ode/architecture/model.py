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
    The dynamics function for the conditional SDE, now with dynamic correction.
    It computes the derivative of the combined state [h, loc, purp_embed, mode_embed].
    """
    def __init__(self, config, state_dim, hidden_dim, num_residual_blocks):
        super().__init__()
        self.config = config
        self.state_dim = state_dim
        
        # We assume specific dimensions in the embeddings correspond to concepts
        # This is a simplification; a more advanced model could learn this mapping.
        self.STAY_MODE_DIM = 0
        self.TRAVEL_PURPOSE_DIM = 5 # Last purpose in the list

        # Core network to learn the main dynamics
        self.net = nn.Sequential(
            nn.Linear(state_dim + 2, hidden_dim), # +2 for sine/cosine time features
            nn.ReLU(),
            *[ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)],
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Required for torchsde compatibility
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def _calculate_potential(self, purpose_embed, mode_embed):
        """Calculates a penalty potential for a comprehensive set of logical rules."""
        # Use Tanh to squash activations to a predictable [-1, 1] range
        activations_purp = torch.tanh(purpose_embed)
        activations_mode = torch.tanh(mode_embed)

        travel_purp_activation = activations_purp[..., self.TRAVEL_PURPOSE_DIM]
        stay_mode_activation = activations_mode[..., self.STAY_MODE_DIM]

        # Activation of all non-travel purposes
        other_purp_dims = [i for i in range(activations_purp.shape[-1]) if i != self.TRAVEL_PURPOSE_DIM]
        other_purp_activation = activations_purp[..., other_purp_dims].mean(dim=-1)

        # Activation of all non-stay modes (i.e., moving modes)
        other_mode_dims = [i for i in range(activations_mode.shape[-1]) if i != self.STAY_MODE_DIM]
        other_mode_activation = activations_mode[..., other_mode_dims].mean(dim=-1)

        # Penalty 1: (Purpose=Travel AND Mode=Stay) should not happen.
        contradiction_1 = travel_purp_activation + stay_mode_activation - 1.5
        potential_1 = (torch.relu(contradiction_1))**2

        # Penalty 2: (Purpose is NOT Travel AND Mode is NOT Stay) should not happen.
        # This enforces that activities must occur in "Stay" mode.
        contradiction_2 = other_purp_activation + other_mode_activation - 1.5
        potential_2 = (torch.relu(contradiction_2))**2
        
        # Penalty 3: (Purpose=Travel AND Mode is NOT a moving mode) should not happen.
        # This is implicitly handled by penalties 1 and 2, but we can add a specific one.
        # This one encourages a moving mode to be active when purpose is travel.
        contradiction_3 = travel_purp_activation - other_mode_activation - 1.0 
        potential_3 = (torch.relu(contradiction_3))**2

        total_potential = potential_1 + potential_2 + potential_3
        return total_potential

    def forward(self, t, z):
        """Calculates dz/dt = f(z,t) with dynamic correction."""
        z.requires_grad_(True) # Essential for calculating the gradient of the potential
        
        # --- 1. Calculate Raw Dynamics ---
        batch_size = z.shape[0]
        if t.dim() == 0:
            t = t.expand(1)
        t_vec = torch.cat([torch.sin(t * 2 * torch.pi / 24), torch.cos(t * 2 * torch.pi / 24)]).expand(batch_size, -1)
        
        raw_dynamics = self.net(torch.cat([z, t_vec], dim=-1))

        # --- 2. Calculate Dynamic Correction Force ---
        # Slice the state to get the relevant embeddings
        slice_dims = [self.config.hidden_dim, self.config.zone_embed_dim, self.config.latent_purpose_embed_dim, self.config.latent_mode_embed_dim]
        _, _, purpose_embed, mode_embed = torch.split(z, slice_dims, dim=-1)

        # Calculate the penalty potential
        potential = self._calculate_potential(purpose_embed, mode_embed)
        
        # If the potential is non-zero, calculate the repulsive force
        if torch.any(potential > 0):
            # The force is the negative gradient of the potential
            repulsive_force = -torch.autograd.grad(
                outputs=potential.sum(), 
                inputs=z, 
                create_graph=True # Allows backpropping through the correction
            )[0]
            
            # Add the corrective force to the raw dynamics
            final_dynamics = raw_dynamics + self.config.correction_strength * repulsive_force
        else:
            final_dynamics = raw_dynamics
            
        return final_dynamics

    def g(self, t, z):
        """Diffusion function for SDE (noise term)."""
        # A simple noise model for now, can be made adaptive later
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
        
        # Define the dimensions of the new combined SDE state
        # State: [hidden_state_h, location_embed, purpose_embed, mode_embed]
        self.state_dim = config.hidden_dim + config.zone_embed_dim + config.latent_purpose_embed_dim + config.latent_mode_embed_dim
        
        self.ode_func = ODEFunc(
            config=config,
            state_dim=self.state_dim,
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
        
        # 1. ENCODER: Get initial latent state components from static features
        encoder_input = torch.cat([person_features, home_embed, work_embed], dim=-1)
        encoder_output = self.encoder(encoder_input)
        
        # Split the encoder output into its parts
        h0_mu, h0_log_var, initial_purpose_embed, initial_mode_embed = torch.split(
            encoder_output,
            [self.config.hidden_dim, self.config.hidden_dim, self.config.latent_purpose_embed_dim, self.config.latent_mode_embed_dim],
            dim=-1
        )
        h0 = h0_mu + torch.exp(0.5 * h0_log_var) * torch.randn_like(h0_mu)

        # 2. INITIAL STATE: Construct the full initial state s(0)
        # We assume the agent starts at home, with a purpose of home and mode of stay
        # The initial embeddings from the encoder act as a starting "bias"
        s0_loc = home_embed
        s0 = torch.cat([h0, s0_loc, initial_purpose_embed, initial_mode_embed], dim=-1)

        # 3. SOLVER: Evolve the state s(t) over the entire time series
        # We must enable gradients for the dynamic correction to work during inference
        with torch.enable_grad():
            if self.config.enable_sde:
                pred_s = sdeint(self.ode_func, s0, times, method='euler', dt=0.01).permute(1, 0, 2)
            else:
                pred_s = odeint(self.ode_func, s0, times, method=self.config.ode_method).permute(1, 0, 2)
        
        # 4. DECODE: Split the solved state path back into its components
        slice_dims = [self.config.hidden_dim, self.config.zone_embed_dim, self.config.latent_purpose_embed_dim, self.config.latent_mode_embed_dim]
        pred_h, pred_y_loc_embed, pred_purpose_embed, pred_mode_embed = torch.split(pred_s, slice_dims, dim=-1)
        
        # 5. PREDICT: Use the hidden state and latent embeddings to get final logits
        # Location prediction
        target_loc_embeds = self.decoder_loc(pred_h)
        pred_y_loc_logits = torch.einsum('bsd,zd->bsz', target_loc_embeds, candidate_zone_embeds)
        
        # Purpose and Mode prediction
        pred_y_purp_logits = self.decoder_purpose(pred_purpose_embed)
        pred_y_mode_logits = self.decoder_mode(pred_mode_embed)
        
        return pred_y_loc_logits, pred_y_loc_embed, pred_y_purp_logits, pred_y_mode_logits, h0_mu, h0_log_var 