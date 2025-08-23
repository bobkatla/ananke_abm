import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import torch.nn.functional as F
from ananke_abm.models.traj_embed.configs import TimeConfig, BasisConfig, QuadratureConfig, PurposeEmbeddingConfig, DecoderConfig
from ananke_abm.models.traj_embed.model.purpose_space import PurposeDistributionSpace
from ananke_abm.models.traj_embed.model.decoder_timefield import TimeFieldDecoder
from ananke_abm.models.traj_embed.model.utils_bases import gauss_legendre_nodes
from ananke_abm.models.traj_embed.train import ScheduleDataset, TrajEncoderGRU, build_masks, collate_fn
from ananke_abm.models.traj_embed.model.rasterize import rasterize_batch
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader

def _spd_cov(Z: torch.Tensor, shrink: float = 0.1, eps: float = 1e-5) -> torch.Tensor:
    """
    Robust covariance with shrinkage and eigenvalue flooring.
    Z: (N, m) on CPU
    """
    N, m = Z.shape
    mu = Z.mean(0, keepdim=True) if N > 0 else torch.zeros(1, m)
    X = Z - mu
    # Sample covariance (Bessel)
    if N > 1:
        S = (X.T @ X) / (N - 1)
    else:
        S = torch.zeros(m, m)
    # Shrinkage toward spherical
    trace = torch.trace(S)
    avg_var = (trace / m) if m > 0 else torch.tensor(1.0)
    S = (1.0 - shrink) * S + shrink * avg_var * torch.eye(m)
    # Eigen floor to ensure PD
    try:
        evals, evecs = torch.linalg.eigh(S)
        evals = torch.clamp(evals, min=eps)
        S_pd = (evecs * evals) @ evecs.T
    except Exception:
        # Fallback: diagonal covariance
        var = torch.clamp(torch.var(Z, dim=0, unbiased=True) if N > 1 else torch.ones(m), min=eps)
        S_pd = torch.diag(var)
    return S_pd, mu.squeeze(0)

def _merge_micro_segments(decoded, min_minutes: int, Tm: int):
    """
    Post-process: merge segments shorter than min_minutes into neighbors.
    decoded: list[list[(p_idx, t0, d)]], t0,d in [0,1]
    """
    out_all = []
    thr = min_minutes / float(Tm)
    for segs in decoded:
        if not segs:
            out_all.append(segs); continue
        out = [segs[0]]
        for p, t0, d in segs[1:]:
            if out and out[-1][2] < thr and out[-1][0] == p:
                # extend previous if same purpose
                prev_p, prev_t0, prev_d = out[-1]
                out[-1] = (prev_p, prev_t0, prev_d + d)
            elif d < thr and out:
                # absorb tiny segment into previous
                prev_p, prev_t0, prev_d = out[-1]
                out[-1] = (prev_p, prev_t0, prev_d + d)
            else:
                out.append((p, t0, d))
        out_all.append(out)
    return out_all

def sample_and_decode(enc, pds, dec, ds, purposes, masks_t, device,
                      num_samples: int = 5, L: int = 360, seed: int = 0,
                      shrink: float = 0.1, renorm_samples: bool = True,
                      min_minutes_merge: int = 0):
    """
    Post-hoc generator:
      1) embed whole dataset -> Z (unit-norm),
      2) fit robust Gaussian on Z (with shrinkage),
      3) sample z, optionally renorm to unit sphere,
      4) decode deterministically (argmax on dense grid),
      5) optional micro-segment merge.
    """
    torch.manual_seed(seed)
    purpose_to_idx = {p: i for i, p in enumerate(purposes)}

    # --- Embed dataset ---
    def _collate(batch):
        return collate_fn(batch, purpose_to_idx)

    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=_collate)
    Z = []
    with torch.no_grad():
        e_p = pds()
        for p_pad, t_pad, d_pad, lengths in dl:
            p_pad = p_pad.to(device); t_pad = t_pad.to(device); d_pad = d_pad.to(device)
            z = enc(p_pad, t_pad, d_pad, lengths, e_p)  # (B, m) unit-norm from your encoder
            Z.append(z.cpu())
    if not Z:
        return []

    Z = torch.cat(Z, dim=0)  # (N, m) on CPU

    # --- Fit robust covariance ---
    Sigma, mu = _spd_cov(Z, shrink=shrink, eps=1e-5)

    # --- Sample and (optionally) re-normalize onto sphere ---
    dist = MultivariateNormal(loc=mu, covariance_matrix=Sigma)
    z_samp = dist.sample((num_samples,))  # CPU
    if renorm_samples:
        z_samp = F.normalize(z_samp, p=2, dim=-1)
    z_samp = z_samp.to(device)

    # --- Decode deterministically on dense grid ---
    t_dense = torch.linspace(0, 1, L, device=device)
    with torch.no_grad():
        e_p = pds()
        loglam_dense = pds.lambda_log(t_dense)
        u_dense = dec.utilities(z_samp, e_p, t_dense, loglam_dense, masks=masks_t)
        decoded = dec.argmax_decode(u_dense, t_dense)

    # --- Optional post-process to reduce micro fragments ---
    if min_minutes_merge > 0:
        decoded = _merge_micro_segments(decoded, min_minutes=min_minutes_merge, Tm=TimeConfig().T_minutes)

    return decoded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--activities_csv", type=str, default="/mnt/data/small_activities_homebound_wd.csv")
    ap.add_argument("--purposes_csv", type=str, default="/mnt/data/purposes.csv")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    priors = ck["priors"]
    purposes = ck["purposes"]
    purpose_to_idx = ck["purpose_to_idx"]

    time_cfg = TimeConfig()
    basis_cfg = BasisConfig()
    quad_cfg = QuadratureConfig()
    pep_cfg = PurposeEmbeddingConfig()
    dec_cfg = DecoderConfig()

    rows = []
    for p in purposes:
        pr = priors[p]
        rows.append(np.concatenate([pr.time_fourier, [pr.mu_t, pr.sigma_t, pr.mu_d, pr.sigma_d]]).astype("float32"))
    phi = np.stack(rows, axis=0); phi = (phi - phi.mean(0, keepdims=True)) / (phi.std(0, keepdims=True) + 1e-6)
    phi_t = torch.tensor(phi, dtype=torch.float32, device=args.device)

    pds = PurposeDistributionSpace(phi_t, d_p=pep_cfg.d_p, hidden=pep_cfg.hidden).to(args.device); pds.set_time_prior_K(basis_cfg.K_time_prior)
    e_p = pds()

    dec = TimeFieldDecoder(P=len(purposes), m_latent=dec_cfg.m_latent, d_p=pep_cfg.d_p, K_decoder_time=basis_cfg.K_decoder_time, alpha_prior=dec_cfg.alpha_prior).to(args.device)
    enc = TrajEncoderGRU(d_p=pep_cfg.d_p, K_time_token=4, K_dur_token=4, m_latent=dec_cfg.m_latent).to(args.device)

    pds.load_state_dict(ck["model_state"]["pds"])
    enc.load_state_dict(ck["model_state"]["enc"])
    dec.load_state_dict(ck["model_state"]["dec"])
    pds.eval()
    enc.eval()
    dec.eval()

    # acts = pd.read_csv(args.activities_csv)
    purp = pd.read_csv(args.purposes_csv)
    ds = ScheduleDataset(args.activities_csv, time_cfg.T_minutes)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, purpose_to_idx))

    t_q, w_q = gauss_legendre_nodes(quad_cfg.Q_nodes, dtype=torch.float32, device=args.device)
    loglam = pds.lambda_log(t_q)

    masks = build_masks(purp, purposes)
    masks_t = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k,v in masks.items()}

    totals = []; ces=[]; emds=[]; tvs=[]
    last_z = None
    with torch.no_grad():
        e_p = pds()
        for p_pad, t_pad, d_pad, lengths in dl:
            p_pad = p_pad.to(args.device); t_pad = t_pad.to(args.device); d_pad = d_pad.to(args.device)
            z = enc(p_pad, t_pad, d_pad, lengths, e_p); last_z = z
            u = dec.utilities(z, e_p, t_q, loglam, masks=masks_t); q = dec.soft_assign(u)

            # rasterize GT
            B, Lmax = p_pad.shape
            segs = []
            for b in range(B):
                L = lengths[b]
                seq = []
                for i in range(L):
                    pid = int(p_pad[b,i].item())
                    pstr = purposes[pid]
                    seq.append((pstr, float(t_pad[b,i].item()), float(d_pad[b,i].item())))
                segs.append(seq)
            y = rasterize_batch(segs, purpose_to_idx, t_q).to(args.device)

            ce = dec.ce_loss(q, y, w_q); emd = dec.emd1d_loss(q, y, w_q); tv = dec.tv_loss(q, w_q)
            total = DecoderConfig.ce_weight*ce + DecoderConfig.emd_weight*emd + DecoderConfig.tv_weight*tv + DecoderConfig.durlen_weight*dec.durlen_loss(q, torch.tensor([priors[p].mu_d for p in purposes], dtype=torch.float32, device=args.device), torch.tensor([priors[p].sigma_d for p in purposes], dtype=torch.float32, device=args.device), w_q)
            totals.append(float(total.item())); ces.append(float(ce.item())); emds.append(float(emd.item())); tvs.append(float(tv.item()))

    print(f"Validation: total={np.mean(totals):.4f}  CE={np.mean(ces):.4f}  EMD={np.mean(emds):.4f}  TV={np.mean(tvs):.4f}")

    # Decode a few examples deterministically on dense grid
    Tm = time_cfg.T_minutes
    L = Tm // 5
    t_dense = torch.linspace(0, 1, L, device=args.device)
    loglam_dense = pds.lambda_log(t_dense)
    with torch.no_grad():
        u_dense = dec.utilities(last_z, e_p, t_dense, loglam_dense, masks=masks_t)  # (B,P,L)
        decoded = dec.argmax_decode(u_dense, t_dense)
        def seg_to_str(seg):
            p_idx, t0, d = seg
            return f"{purposes[p_idx]} @ {int(t0*Tm)}m for {int(d*Tm)}m"
        for b, segs in enumerate(decoded[:3]):
            print(f"Decoded sample {b}:")
            print(" | ".join(seg_to_str(s) for s in segs))

    gen = sample_and_decode(
        enc, pds, dec, ds, purposes, masks_t, args.device,
        num_samples=5, L=L, seed=0, shrink=0.1, renorm_samples=True,
        min_minutes_merge=5   # e.g., merge <5-minute slivers; set 0 to disable
    )
    print("=== Generated samples (from latent Gaussian on z) ===")
    
    for i, segs in enumerate(gen):
        pretty = " | ".join(f"{purposes[p]} @ {int(t0*Tm)}m for {int(d*Tm)}m" for (p,t0,d) in segs)
        print(f"Gen {i}: {pretty}")

if __name__ == "__main__":
    main()
