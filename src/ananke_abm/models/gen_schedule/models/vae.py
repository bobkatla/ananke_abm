
import torch
import torch.nn as nn
import torch.nn.functional as F
from ananke_abm.models.gen_schedule.models.encoders import CNNEncoder
from ananke_abm.models.gen_schedule.models.decoders import TimeBasisDecoder

def kl_gaussian(mu, logvar):
    return 0.5 * torch.mean(mu.pow(2) + logvar.exp() - 1.0 - logvar)

class ScheduleVAE(nn.Module):
    def __init__(self, L:int, P:int, z_dim:int=12, emb_dim:int=32):
        super().__init__()
        self.encoder = CNNEncoder(P, L, emb_dim=emb_dim, z_dim=z_dim)
        self.decoder = TimeBasisDecoder(L, P, z_dim=z_dim, H=32)

    def forward(self, y_idx):
        mu, logvar = self.encoder(y_idx)
        std = torch.exp(0.5*logvar)
        z = mu + torch.randn_like(std) * std
        U = self.decoder(z)
        return U, mu, logvar

    def recon_ce(self, U, y_idx):
        return F.cross_entropy(U.permute(0,2,1), y_idx, reduction="mean")
