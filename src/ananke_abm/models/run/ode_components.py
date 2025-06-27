import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint


class ODEFunc(nn.Module):
    def __init__(self, in_features, hidden_features, time_embed_dim):
        super(ODEFunc, self).__init__()
        self.time_embed = nn.Linear(1, time_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(in_features + time_embed_dim, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, in_features)
        )
        # Learnable parameter to control the strength of the restart connection
        self.restart_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, t, x, h0):
        # Embed time and concatenate with input state
        t_embedded = self.time_embed(t.expand(x.shape[0], 1))
        x_with_time = torch.cat([x, t_embedded], dim=-1)
        
        # Calculate the derivative and add the restart connection
        # This pulls the derivative towards the initial state, preventing drift
        dx_dt = self.net(x_with_time) + self.restart_alpha * (h0 - x)
        return dx_dt


class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol=1e-5, atol=1e-5, method='dopri5'):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.rtol = rtol
        self.atol = atol
        self.method = method

    def forward(self, x0, t):
        # Wrap the odefunc and the initial state `x0` in a new nn.Module
        # This is necessary for the adjoint method to be able to track parameters.
        class WrappedODEFunc(nn.Module):
            def __init__(self, odefunc, h0):
                super().__init__()
                self.odefunc = odefunc
                self.h0 = h0

            def forward(self, t, y):
                return self.odefunc(t, y, self.h0)

        wrapped_func = WrappedODEFunc(self.odefunc, x0)
        return odeint(wrapped_func, x0, t, rtol=self.rtol, atol=self.atol, method=self.method) 