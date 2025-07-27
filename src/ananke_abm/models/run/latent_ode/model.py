"""
Model architecture for the Generative Latent ODE.
"""
import torch
import torch.nn as nn
from torchdiffeq import odeint

from .config import GenerativeODEConfig

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
    The dynamics function for the conditional ODE. It computes the derivative of
    the combined state [h, y_loc, y_purp] at time t.
    """
    def __init__(self, state_dim, hidden_dim, num_residual_blocks):
        super().__init__()
        # A single network computes the derivative of the entire state vector
        self.net = nn.Sequential(
            nn.Linear(state_dim + 2, hidden_dim), # +2 for sine/cosine time features
            nn.ReLU(),
            *[ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)],
            nn.Linear(hidden_dim, state_dim)
        )
        self.static_features = None
        self.home_zone_embedding = None

    def forward(self, t, state):
        # state is the combined vector [h, y_loc, y_purp]
        # Ensure t is a tensor and then expand for sine/cosine features
        if t.dim() == 0:
            t = t.expand(1)
        t_vec = torch.cat([torch.sin(t * 2 * torch.pi / 24), torch.cos(t * 2 * torch.pi / 24)]).expand(state.shape[0], -1)
        # The dynamics are conditioned on the entire current state and time
        return self.net(torch.cat([state, t_vec], dim=-1))

class GenerativeODE(nn.Module):
    """
    A conditional, autoregressive Latent ODE model.
    The ODE state itself contains the evolving location and purpose embeddings.
    """
    def __init__(self, person_feat_dim, num_zones, config):
        super().__init__()
        self.config = config
        
        self.zone_embedder = nn.Embedding(num_zones, config.zone_embed_dim)
        self.purpose_embedder = nn.Embedding(len(config.purpose_groups), config.purpose_embed_dim)

        # The encoder produces the initial hidden state h(0)
        encoder_input_dim = person_feat_dim + config.zone_embed_dim * 2
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, config.encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.encoder_hidden_dim, config.hidden_dim * 2), # mu and log_var for h(0)
        )
        
        # Define the dimensions of the combined ODE state
        self.state_dim = config.hidden_dim + config.zone_embed_dim + config.purpose_embed_dim
        
        self.ode_func = ODEFunc(
            state_dim=self.state_dim,
            hidden_dim=config.ode_hidden_dim,
            num_residual_blocks=config.num_residual_blocks
        )
        
        # Decoders now operate on the hidden state component h(t)
        self.decoder_loc_logits = nn.Linear(config.hidden_dim, num_zones)
        self.decoder_purp_logits = nn.Linear(config.hidden_dim, len(config.purpose_groups))

    def forward(self, person_features, home_zone_id, work_zone_id, start_purp_id, times):
        
        home_embed = self.zone_embedder(home_zone_id)
        work_embed = self.zone_embedder(work_zone_id)
        
        # 1. ENCODER: Get initial hidden state h(0) from static features
        encoder_input = torch.cat([person_features, home_embed, work_embed], dim=-1)
        h0_params = self.encoder(encoder_input)
        mu, log_var = h0_params.chunk(2, dim=-1)
        h0 = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

        # 2. INITIAL STATE: Construct the full initial state s(0)
        s0_loc = self.zone_embedder(home_zone_id) # Start at home location
        s0_purp = self.purpose_embedder(start_purp_id) # Start with home purpose
        s0 = torch.cat([h0, s0_loc, s0_purp], dim=-1)

        # 3. SOLVER: Evolve the state s(t) over the entire time series
        pred_s = odeint(self.ode_func, s0, times, method=self.config.ode_method).permute(1, 0, 2)
        
        # 4. DECODE: Split the solved state back into its components
        pred_h = pred_s[..., :self.config.hidden_dim]
        pred_y_loc_embed = pred_s[..., self.config.hidden_dim : self.config.hidden_dim + self.config.zone_embed_dim]
        # pred_y_purp_embed is also available if we want to use it in the future
        
        # Use the hidden state trajectory to predict logits
        pred_y_loc_logits = self.decoder_loc_logits(pred_h)
        pred_y_purp_logits = self.decoder_purp_logits(pred_h)
        
        return pred_y_loc_logits, pred_y_loc_embed, pred_y_purp_logits, mu, log_var 