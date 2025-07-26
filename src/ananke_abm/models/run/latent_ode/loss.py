"""
Loss function for the Generative Latent ODE model.
"""
import torch
import torch.nn.functional as F

def calculate_loss(pred_y_logits, trajectory_y, mu, log_var, kl_weight):
    """
    Calculates the VAE loss, which is a combination of reconstruction loss
    and KL divergence loss.
    """
    # Reconstruction loss (how well the model reproduces the data)
    recon_loss = F.cross_entropy(pred_y_logits, trajectory_y)
    
    # KL divergence loss (a regularizer on the latent space)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Combine the two losses
    loss = recon_loss + kl_weight * kl_loss
    
    return loss, recon_loss, kl_loss 