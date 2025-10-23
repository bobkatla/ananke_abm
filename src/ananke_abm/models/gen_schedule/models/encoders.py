
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, P: int, L: int, emb_dim: int=32, z_dim: int=12, channels=(64,64)):
        super().__init__()
        self.emb = nn.Embedding(P, emb_dim)
        c = emb_dim
        convs = []
        for ch in channels:
            convs += [nn.Conv1d(c, ch, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(ch)]
            c = ch
        self.conv = nn.Sequential(*convs)
        self.head_mu = nn.Linear(c, z_dim)
        self.head_logvar = nn.Linear(c, z_dim)

    def forward(self, y_idx):  # (B,L) int
        x = self.emb(y_idx).transpose(1,2)   # (B,emb,L)
        h = self.conv(x).mean(dim=-1)        # (B,C)
        mu = self.head_mu(h)
        logvar = self.head_logvar(h).clamp(min=-6., max=6.)
        return mu, logvar
