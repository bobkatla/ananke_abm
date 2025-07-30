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
    purpose_embed_dim: int = 4
    num_residual_blocks: int = 2
    
    # Training parameters
    learning_rate: float = 1e-3
    kl_weight: float = 0.5
    num_iterations: int = 6000
    
    # --- New Composite Loss Weights ---
    loss_weight_classification: float = 1.0
    loss_weight_embedding: float = 0.5
    loss_weight_distance: float = 2.0
    loss_weight_purpose: float = 0.75
    loss_weight_mode: float = 1.0  # NEW: Weight for mode classification loss

    # --- New Anchor Loss Weight ---
    anchor_loss_weight: float = 10.0

    # --- New Training Mode ---
    train_on_interpolated_points: bool = False

    # ODE solver settings
    ode_method: str = 'dopri5'
    
    # Attention mechanisms
    enable_attention: bool = True
    attention_strength: float = 0.1
    
    # Mode choice parameters
    num_modes: int = 4  # Stay, Walk, Car, Public_Transit
    mode_embed_dim: int = 4  # For backward compatibility if needed
    
    # Purpose configuration
    purpose_groups: tuple = ("Home", "Work/Education", "Subsistence", "Leisure & Recreation", "Social", "Travel/Transit") 