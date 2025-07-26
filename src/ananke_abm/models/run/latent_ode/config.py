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
    kl_weight: float = 0.5
    num_iterations: int = 8000
    
    # --- New Composite Loss Weights ---
    loss_weight_classification: float = 1.0
    loss_weight_embedding: float = 0.5
    loss_weight_distance: float = 2.0
    loss_weight_purpose: float = 0.75

    # --- New Anchor Loss Weight ---
    initial_step_loss_weight: float = 10.0

    # ODE solver settings
    ode_method: str = 'dopri5'
    
    # Purpose configuration
    purpose_groups: tuple = ("Home", "Work/Education", "Subsistence", "Leisure & Recreation", "Social", "Travel/Transit") 