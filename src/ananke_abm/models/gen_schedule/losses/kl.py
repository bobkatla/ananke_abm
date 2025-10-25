import torch

def kl_gaussian(mu, logvar):
    return 0.5 * torch.mean(mu.pow(2) + logvar.exp() - 1.0 - logvar)