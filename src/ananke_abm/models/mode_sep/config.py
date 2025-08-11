"""
Configuration for the deterministic, second-order ODE location-only model (mode_sep).

All constants live here; there should be no magic numbers elsewhere.
"""
from dataclasses import dataclass


@dataclass
class ModeSepConfig:
    # Reproducibility and device
    seed: int = 42
    device: str = "cuda"  # or "cpu"

    # Embedding & context dimensions
    emb_dim: int = 64           # E: location embedding dimension
    context_dim: int = 32       # H: static person context dimension
    zone_emb_dim: int = 8       # learnable embeddings for zone IDs (home/work)

    # Drift network
    hidden_dim: int = 128
    num_res_blocks: int = 2

    # Time grid & solver
    K_internal: int = 8                 # internal points between adjacent snaps
    ode_method: str = "rk4"             # method for torchdiffeq.odeint
    rtol: float = 1e-5
    atol: float = 1e-5
    time_match_tol: float = 1e-6        # tolerance to match union time to snap time

    # Loss weights
    softmax_tau: float = 0.10
    w_ce: float = 1.0
    w_mse: float = 0.5
    w_dist: float = 0.1
    w_stay_vel: float = 5.0

    # Training
    max_epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # Evaluation / plots
    dense_resolution: int = 500
    epsilon_v: float = 0.05             # stay compliance threshold on |v|
    transition_window_h: float = 0.25   # +/- window for transition sharpness (hours)

    # IO
    checkpoints_dir: str = "saved_models/mode_sep/mode_sep_checkpoints"
    figures_dir: str = "saved_models/mode_sep/mode_sep_figures"
    runs_dir: str = "saved_models/mode_sep/mode_sep_runs"


