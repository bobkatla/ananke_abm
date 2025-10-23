
import torch
import torch.nn as nn

class TimeBasisDecoder(nn.Module):
    def __init__(self, L: int, P: int, z_dim: int=12, H: int=32):
        super().__init__()
        self.L, self.P, self.H = L, P, H
        self.B_time = nn.Parameter(torch.randn(L, H) * 0.05)
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, 128), nn.ReLU(),
            nn.Linear(128, P*H)
        )
        self.bias_p = nn.Parameter(torch.zeros(P))

    def forward(self, z):
        B = z.size(0)
        C = self.mlp(z).view(B, self.P, self.H)  # (B,P,H)
        U = torch.einsum("lh,bph->blp", self.B_time, C) + self.bias_p
        return U
