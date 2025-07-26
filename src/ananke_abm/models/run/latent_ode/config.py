"""
Configuration for the Generative Latent ODE model.
"""

class GenerativeODEConfig:
    latent_dim: int = 32
    zone_embed_dim: int = 16
    encoder_hidden_dim: int = 128
    ode_hidden_dim: int = 128
    
    # Training parameters
    learning_rate: float = 1e-3
    num_iterations: int = 10000
    kl_weight: float = 0.01  # Weight for the KL divergence term
    
    # ODE solver settings
    ode_method: str = 'dopri5'
    
    # Purpose configuration
    purpose_groups: list = ["Home", "Work/Education", "Subsistence", "Leisure & Recreation", "Social", "Travel/Transit"] 