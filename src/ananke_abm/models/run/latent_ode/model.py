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
    """The dynamics function f(z, t, ...) using a ResNet-like architecture."""
    def __init__(self, latent_dim, hidden_dim, static_dim, home_embed_dim):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(latent_dim + static_dim + home_embed_dim + 1, hidden_dim), nn.Tanh())
        self.residual_blocks = nn.Sequential(ResidualBlock(hidden_dim), ResidualBlock(hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, t, z):
        t_vec = torch.ones(z.shape[0], 1).to(z.device) * t
        z_t_static_home = torch.cat([z, t_vec, self.static_features, self.home_zone_embedding], dim=-1)
        h = self.input_layer(z_t_static_home)
        h = self.residual_blocks(h)
        return self.output_layer(h)

class GenerativeODE(nn.Module):
    """A generative VAE model that decodes a latent vector z into a full trajectory."""
    def __init__(self, person_feat_dim, num_zones, config: GenerativeODEConfig):
        super().__init__()
        self.config = config
        self.zone_embedder = nn.Embedding(num_zones, config.zone_embed_dim)
        
        encoder_input_dim = person_feat_dim + config.zone_embed_dim * 2 + len(config.purpose_groups)
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, config.encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.encoder_hidden_dim, config.latent_dim * 2),
        )
        
        self.decoder = nn.Linear(config.latent_dim, num_zones)
        self.ode_func = ODEFunc(
            latent_dim=config.latent_dim, 
            hidden_dim=config.ode_hidden_dim,
            static_dim=person_feat_dim + config.zone_embed_dim, # person_features + work_embed
            home_embed_dim=config.zone_embed_dim
        )

    def forward(self, person_features, home_zone_id, work_zone_id, purpose_features, times):
        home_embed = self.zone_embedder(home_zone_id)
        work_embed = self.zone_embedder(work_zone_id)
        
        encoder_input = torch.cat([person_features, home_embed, work_embed, purpose_features], dim=-1)
        
        latent_params = self.encoder(encoder_input)
        mu, log_var = latent_params.chunk(2, dim=-1)
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z0 = mu + eps * std

        self.ode_func.static_features = torch.cat([person_features, work_embed], dim=-1).expand(z0.shape[0], -1)
        self.ode_func.home_zone_embedding = home_embed.expand(z0.shape[0], -1)
        
        pred_z = odeint(self.ode_func, z0, times, method=self.config.ode_method).permute(1, 0, 2)
        pred_y_logits = self.decoder(pred_z)
        
        return pred_y_logits, mu, log_var 