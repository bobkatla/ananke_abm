# Trajectory Embedding – Continuous Time-Field Scaffold

## Files
- `configs.py` – knobs for time horizon, basis sizes, quadrature nodes, and default loss weights.
- `utils_bases.py` – Fourier time features and Gauss–Legendre quadrature nodes/weights.
- `pds_loader.py` – derives per-purpose priors from schedules (time-of-day λ_p and duration prior).
- `purpose_space.py` – Purpose Distribution Space (φ_p → e_p) + log λ_p(t) evaluation.
- `rasterize.py` – convert segments to one-hot over time nodes (for supervision).
- `decoder_timefield.py` – continuous time-field decoder + CE/EMD/TV/durlen losses.

## Minimal smoke test
```python
import numpy as np, torch, pandas as pd
from traj_embed.configs import TimeConfig, BasisConfig
from traj_embed.pds_loader import derive_priors_from_activities
from traj_embed.purpose_space import PurposeDistributionSpace
from traj_embed.decoder_timefield import TimeFieldDecoder
from traj_embed.utils_bases import gauss_legendre_nodes
from traj_embed.rasterize import rasterize_batch

acts = pd.read_csv("/mnt/data/small_activities_homebound_wd.csv")
purp = pd.read_csv("/mnt/data/purposes.csv")

Tm = TimeConfig().T_minutes
Kt = BasisConfig().K_time_prior
priors, purposes = derive_priors_from_activities(acts, purp, Tm, Kt)

# Build phi_p and standardize
rows = []
for p in purposes:
    pr = priors[p]
    rows.append(np.concatenate([pr.time_fourier, [pr.mu_t, pr.sigma_t, pr.mu_d, pr.sigma_d]]).astype("float32"))
phi = np.stack(rows, axis=0)
phi = (phi - phi.mean(0, keepdims=True)) / (phi.std(0, keepdims=True)+1e-6)

pds = PurposeDistributionSpace(torch.tensor(phi), d_p=16, hidden=64)
pds.set_time_prior_K(Kt)
e_p = pds()  # (P,16)

# Decoder
P = len(purposes)
dec = TimeFieldDecoder(P=P, m_latent=16, d_p=16, K_decoder_time=8, alpha_prior=1.0)

# Quadrature and priors
t_q, w_q = gauss_legendre_nodes(96, dtype=torch.float32)
loglam = pds.lambda_log(t_q)

# Fake latent and batch
B = 2
z = torch.randn(B, 16)

u = dec.utilities(z, e_p, t_q, loglam, masks=None)
q = dec.soft_assign(u)

# Build small GT batch from first two persons (normalized times)
segs = []
for pid, g in acts.groupby("persid"):
    g = g.sort_values("startime")
    day = [(r["purpose"], r["startime"]/Tm, r["total_duration"]/Tm) for _, r in g.iterrows()]
    segs.append(day)
    if len(segs) >= B: break
purpose_to_idx = {p:i for i,p in enumerate(purposes)}
y = rasterize_batch(segs, purpose_to_idx, t_q)

# Losses
ce = dec.ce_loss(q, y, w_q)
emd = dec.emd1d_loss(q, y, w_q)
tv  = dec.tv_loss(q, w_q)

print("CE:", float(ce), "EMD:", float(emd), "TV:", float(tv))
```
