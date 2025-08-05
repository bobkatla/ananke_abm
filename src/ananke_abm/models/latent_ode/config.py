"""
Configuration for the Generative Latent ODE model.
"""
from dataclasses import dataclass, field
from ananke_abm.data_generator.feature_engineering import (
    get_feature_dimensions,
    PURPOSE_ID_MAP,
    MODE_ID_MAP,
)

# Dynamically get feature dimensions
MODE_FEAT_DIM, PURPOSE_FEAT_DIM = get_feature_dimensions()
# Dynamically get purpose groups in the correct order
PURPOSE_GROUPS = tuple(sorted(PURPOSE_ID_MAP, key=PURPOSE_ID_MAP.get))
# Dynamically get the number of modes
NUM_MODES = len(MODE_ID_MAP)

@dataclass
class GenerativeODEConfig:
    """Configuration for the Generative ODE model."""
    hidden_dim: int = 32
    encoder_hidden_dim: int = 64
    ode_hidden_dim: int = 128
    zone_embed_dim: int = 8
    
    # New rich feature dimensions, set dynamically
    purpose_feature_dim: int = PURPOSE_FEAT_DIM
    mode_feature_dim: int = MODE_FEAT_DIM

    num_residual_blocks: int = 2

    # Dynamic Correction for SDE
    correction_strength: float = 1.0
    use_second_order_sde: bool = True
    
    # Training parameters
    learning_rate: float = 1e-3
    kl_weight: float = 0.5
    num_iterations: int = 25000
    
    # --- Composite Loss Weights ---
    loss_weight_classification: float = 1.0
    loss_weight_embedding: float = 0.5
    loss_weight_distance: float = 2.0
    loss_weight_purpose_class: float = 0.75
    loss_weight_mode_class: float = 1.0
    loss_weight_purpose_mse: float = 0.5  # New MSE weight
    loss_weight_mode_mse: float = 0.5   # New MSE weight

    # --- New Anchor Loss Weight ---
    anchor_loss_weight: float = 15.0

    # --- New Training Mode ---
    train_on_interpolated_points: bool = False

    # ODE solver settings
    ode_method: str = 'dopri5'

    # SDE settings for stochastic dynamics
    enable_sde: bool = True
    sde_noise_strength: float = 0.1
    
    # Attention mechanisms
    enable_attention: bool = True
    attention_strength: float = 0.1
    
    # Mode choice parameters, set dynamically
    num_modes: int = NUM_MODES
    
    # Purpose configuration, set dynamically
    purpose_groups: tuple = field(default_factory=lambda: PURPOSE_GROUPS)
