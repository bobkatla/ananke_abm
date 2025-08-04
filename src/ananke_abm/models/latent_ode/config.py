"""
Configuration for the Generative Latent ODE model.
"""
from dataclasses import dataclass

@dataclass
class GenerativeODEConfig:
    """Configuration for the Generative ODE model."""
    hidden_dim: int = 32
    encoder_hidden_dim: int = 64
    ode_hidden_dim: int = 128
    zone_embed_dim: int = 8
    latent_purpose_embed_dim: int = 8
    latent_mode_embed_dim: int = 8
    num_residual_blocks: int = 2

    # Dynamic Correction for SDE
    correction_strength: float = 1.0
    
    # Training parameters
    learning_rate: float = 1e-3
    kl_weight: float = 0.5
    num_iterations: int = 10000
    
    # --- New Composite Loss Weights ---
    loss_weight_classification: float = 1.0
    loss_weight_embedding: float = 0.5
    loss_weight_distance: float = 2.0
    loss_weight_purpose: float = 0.75
    loss_weight_mode: float = 1.0

    # --- New Anchor Loss Weight ---
    anchor_loss_weight: float = 15.0

    # --- New Training Mode ---
    train_on_interpolated_points: bool = False

    # ODE solver settings
    ode_method: str = 'dopri5'

    # SDE settings for stochastic dynamics
    enable_sde: bool = True  # Enable stochastic differential equations
    sde_noise_strength: float = 0.1  # Base noise level for transitions
    
    # Attention mechanisms
    enable_attention: bool = True # Keep for now, but will be removed from ODEFunc
    attention_strength: float = 0.1
    
    # Mode choice parameters
    num_modes: int = 4  # Stay, Walk, Car, Public_Transit
    
    # Purpose configuration
    purpose_groups: tuple = ("Home", "Work/Education", "Subsistence", "Leisure & Recreation", "Social", "Travel/Transit") 