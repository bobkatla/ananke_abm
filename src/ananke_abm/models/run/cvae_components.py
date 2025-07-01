import torch
from torch import nn

class Encoder(nn.Module):
    """
    Encodes a sequence of observations into a latent variable distribution (mu and log_var).
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        # Embed discrete zone IDs into a continuous space
        self.embed = nn.Embedding(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x is expected to be a sequence of zone IDs, shape [batch_size, seq_len]
        embedded = self.embed(x)
        _, h_n = self.gru(embedded)  # We only need the final hidden state
        h_n = h_n.squeeze(0)  # Shape [batch_size, hidden_dim]
        
        mu = self.fc_mu(h_n)
        log_var = self.fc_log_var(h_n)
        return mu, log_var

class LatentVariable(nn.Module):
    """
    Handles the reparameterization trick to sample from the latent space.
    """
    def forward(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

class LatentODEFunc(nn.Module):
    """
    Defines the dynamics of the latent variable z, i.e., dz/dt.
    """
    def __init__(self, latent_dim, hidden_dim):
        super(LatentODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, t, z):
        return self.net(z) 