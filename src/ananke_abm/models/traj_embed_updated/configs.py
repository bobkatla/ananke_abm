from dataclasses import dataclass, field
from typing import Dict

# ---- TIME & GRIDS ----
@dataclass
class TimeConfig:
    # Allocation horizon (what we decode/optimize over) and circadian clock
    ALLOCATION_HORIZON_MINS: int = 1800      # 30h window to allow spillovers
    T_clock_minutes: int = 1440      # 24h periodic clock

    # Grid step sizes
    TRAIN_GRID_MINS: int = 10      # CRF training grid (fast)
    VALID_GRID_MINS: int = 2       # CRF/Viterbi decode grid (fidelity)

    # (legacy) old normalized flag retained for back-compat; unused going forward
    t_norm: bool = True


# ---- BASES ----
@dataclass
class BasisConfig:
    # Periodic prior is now explicitly 24h-clock-based
    K_clock_prior: int = 6           # Fourier pairs for λ_p(clock); replaces K_time_prior
    K_decoder_time: int = 8          # Fourier pairs for decoder utilities over allocation time


# ---- PURPOSE EMBEDDINGS ----
@dataclass
class PurposeEmbeddingConfig:
    d_p: int = 32
    hidden: int = 64


# ---- LATENT (β-VAE) ----
@dataclass
class VAEConfig:
    latent_dim: int = 32             # dim(s); z = s / ||s||
    beta: float = 0.25                # KL weight
    kl_anneal_start: int = 0         # optional linear anneal (epoch)
    kl_anneal_end:   int = 50


# ---- CRF DECODER (resource allocation on a grid) ----
@dataclass
class CRFConfig:
    eta: float = 0.5                 # Potts switching penalty (off-diagonal)
    learn_eta: bool = True          # make eta learnable if True
    use_transition_mask: bool = False  # enable bigram feasibility masks
    semi_Dmax_minutes: int = 300   # max segment duration for semi-CRF


# ---- ALPHA PRIOR ----
@dataclass
class AlphaPriorConfig:
    # Per-purpose initial alpha; names must match the "purpose" strings from purposes.csv
    alpha_init_per_purpose: Dict[str, float] = field(default_factory=lambda: {
        "Home": 1.8,
        "Work": 1.3,
        "Education": 1.1,
        "Shopping": 0.9,
        "Social": 0.8,
        "Accompanying": 0.9,
        "Other": 0.7,
    })
    # L2 strength for pullback to init (phase-1 regularizer)
    alpha_l2: float = 1e-3

@dataclass
class DecoderRegConfig:
    # Global base L2 for allocation-Fourier coefficients (latent-driven part)
    coeff_l2_global: float = 1e-4
    # Optional per-purpose overrides (if a key is here, it replaces the global value for that purpose)
    coeff_l2_per_purpose: Dict[str, float] = field(default_factory=lambda: {
        "Shopping": 3e-4,
        "Accompanying": 3e-4,
        # You can add others here if needed later
    })

# ---- DECODER MISC ----
@dataclass
class DecoderConfig:
    # Keep m_latent for temporary back-compat with existing code; should match VAEConfig.latent_dim
    m_latent: int = 32
    alpha_prior: float = 1.0         # weight for log λ_p(clock(t)) bias in utilities (deprecated; use alpha_cfg instead)
    alpha_cfg: AlphaPriorConfig = field(default_factory=AlphaPriorConfig)
    # NEW: Phase 3 regularization knobs
    reg_cfg: DecoderRegConfig = field(default_factory=DecoderRegConfig)

@dataclass
class LossBalanceConfig:
    # Training-only weights; keys must match purposes from purposes.csv
    # Start modest; you can tune up/down after 1–2 runs
    loss_weights_per_purpose: Dict[str, float] = field(default_factory=lambda: {
        "Home": 1.0,
        "Work": 1.0,
        "Education": 1.0,
        "Shopping": 1.8,
        "Social": 1.3,
        "Accompanying": 1.6,
        "Other": 1.2,
    })
