from dataclasses import dataclass

@dataclass
class TimeConfig:
    T_minutes: int = 1800           # 30 hours (your small set)
    t_norm: bool = True             # normalize to [0,1]

@dataclass
class BasisConfig:
    # Fourier basis orders for priors and decoder
    K_time_prior: int = 6           # number of cosine/sine pairs for lambda_p(t) (low frequency)
    K_decoder_time: int = 8         # number of cosine/sine pairs for decoder utilities

@dataclass
class QuadratureConfig:
    Q_nodes_train: int = 48
    Q_nodes_val: int = 96

@dataclass
class PurposeEmbeddingConfig:
    d_p: int = 16                   # dimension for purpose embedding e_p
    hidden: int = 64                # hidden width for purpose MLP

@dataclass
class DecoderConfig:
    m_latent: int = 16              # latent dimension z
    alpha_prior: float = 1.0        # weight of log(lambda_p(t)) prior term in utilities
    tv_weight: float = 0.01         # default loss weights (placeholders; tune later)
    ce_weight: float = 1.0
    emd_weight: float = 0.5
    durlen_weight: float = 0.1
    lap_weight: float = 0.0         # optional if you later add Laplacian reg
    home_weight: float = 0.05        # weight of terminal home loss
    nonhome_mse_weight: float = 0.01 # weight of non-home mass MSE loss
    bigram_prior_weight: float = 0.08 # weight of bigram prior loss
    alpha_home_factor: float = 0.7    # factor for Home's alpha_per_p