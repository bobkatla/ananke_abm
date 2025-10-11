import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# --- your refactor imports (as you described) ---
from ananke_abm.models.traj_syn.configs import (
    TimeConfig, BasisConfig, PurposeEmbeddingConfig, DecoderConfig, VAEConfig
)
from ananke_abm.models.traj_syn.core.data_utils.randomness import set_seed
from ananke_abm.models.traj_syn.core.data_utils.sanitize import sanitize_theta
from ananke_abm.models.traj_syn.core.data_utils.ScheduleDataset import ScheduleDataset, collate_fn
from ananke_abm.models.traj_syn.core.utils_bases import make_alloc_grid
from ananke_abm.models.traj_syn.core.rasterize import rasterize_from_padded_to_grid
from ananke_abm.models.traj_syn.vae.purpose_space import PurposeDistributionSpace
from ananke_abm.models.traj_syn.vae.decoder_timefield import TimeFieldDecoder
from ananke_abm.models.traj_syn.vae.encoder import TrajEncoderGRU, kl_gaussian_standard


# -------------------------
# utilities
# -------------------------
def build_phi_features(purposes: List[str], priors: Dict[str, object], device: torch.device) -> torch.Tensor:
    rows = []
    for p in purposes:
        pr = priors[p]
        rows.append(
            np.concatenate(
                [pr.time_fourier, [pr.mu_t, pr.sigma_t, pr.mu_d, pr.sigma_d, pr.is_primary_ooh]]
            ).astype("float32")
        )
    phi = np.stack(rows, axis=0)
    phi = (phi - phi.mean(0, keepdims=True)) / (phi.std(0, keepdims=True) + 1e-6)
    return torch.tensor(phi, dtype=torch.float32, device=device)


def top1_ce_loss(theta: torch.Tensor, y_grid: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy over grid (no CRF). theta: [B,P,L], y_grid: [B,L].
    Returns mean CE across B*L.
    """
    B, P, L = theta.shape
    logp = torch.log_softmax(theta, dim=1)          # [B,P,L]
    idx = y_grid.unsqueeze(1)                       # [B,1,L]
    gold = logp.gather(dim=1, index=idx).squeeze(1) # [B,L]
    return -gold.mean()


def compute_real_time_marginals_from_activities(
    activities_csv: str,
    purposes: List[str],
    T_alloc_minutes: int,
    step_minutes: int,
) -> np.ndarray:
    """
    Return q[p, t] as empirical purpose occupancy over allocation horizon on a grid.
    """
    acts = pd.read_csv(activities_csv)
    acts = acts.sort_values(["persid", "startime", "stopno"])
    L = math.ceil(T_alloc_minutes / step_minutes)
    P = len(purposes)
    p2i = {p: i for i, p in enumerate(purposes)}
    counts = np.zeros((P, L), dtype=np.float64)
    total_people = acts["persid"].nunique()

    # Fill occupancy per bin
    for pid, g in acts.groupby("persid"):
        occ = np.full(L, -1, dtype=np.int32)  # label per bin
        for _, r in g.iterrows():
            p = p2i.get(str(r["purpose"]), None)
            if p is None: 
                continue
            start = int(r["startime"] // step_minutes)
            dur_bins = max(1, int(round(r["total_duration"] / step_minutes)))
            end = min(L, start + dur_bins)
            occ[start:end] = p
        for t in range(L):
            if occ[t] >= 0:
                counts[occ[t], t] += 1.0

    # Normalize to probabilities across purposes at each t
    denom = counts.sum(axis=0, keepdims=True) + 1e-9
    q = counts / denom
    return q  # [P,L]


def match_grid(source: np.ndarray, target_L: int) -> np.ndarray:
    """Linearly resample along time axis to target length."""
    P, Ls = source.shape
    if Ls == target_L:
        return source
    xp = np.linspace(0.0, 1.0, Ls)
    xq = np.linspace(0.0, 1.0, target_L)
    out = np.zeros((P, target_L), dtype=source.dtype)
    for p in range(P):
        out[p] = np.interp(xq, xp, source[p])
    # renormalize across purposes per t
    out = out / (out.sum(axis=0, keepdims=True) + 1e-9)
    return out


def jsd_batch(p_batch: torch.Tensor, q_target: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Jensen-Shannon divergence per t, averaged over p,t. p_batch: [P,L], q_target: [P,L] (both sum to 1 over P per t).
    """
    # clamp to avoid log(0)
    p = torch.clamp(p_batch, eps, 1.0)
    q = torch.clamp(q_target, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=0)   # [L]
    kl_qm = (q * (q.log() - m.log())).sum(dim=0)   # [L]
    js = 0.5 * (kl_pm + kl_qm)                     # [L]
    return js.mean()


def batch_time_marginal(theta: torch.Tensor) -> torch.Tensor:
    """
    Average softmax(theta) across batch. theta: [B,P,L] -> [P,L]
    """
    p = torch.softmax(theta, dim=1)      # [B,P,L]
    return p.mean(dim=0)                 # [P,L]


def nonhome_presence_loss(
    theta: torch.Tensor, home_idx: int, target_rate: float
) -> torch.Tensor:
    """
    Encourage batch probability of having ≥1 non-Home to match target_rate.
    Proxy per person: u_b = 1 - prod_t p(home|t).
    """
    p = torch.softmax(theta, dim=1)             # [B,P,L]
    p_home = p[:, home_idx, :]                  # [B,L]
    # Sum of log to avoid underflow for product
    log_prod = torch.log(torch.clamp(p_home, 1e-9, 1.0)).sum(dim=1)  # [B]
    prod = torch.exp(log_prod)                                       # [B]
    u = 1.0 - prod                                                   # [B]
    mean_u = u.mean()
    return (mean_u - float(target_rate))**2


def duration_hist_from_activities(
    activities_csv: str, purposes: List[str], step_minutes: int, T_alloc_minutes: int
) -> Dict[int, np.ndarray]:
    """
    Very simple duration histogram per purpose with step-aligned bins up to T_alloc_minutes.
    Returns: {p_idx: hist[nbins]} normalized.
    """
    acts = pd.read_csv(activities_csv)
    p2i = {p: i for i, p in enumerate(purposes)}
    nbins = math.ceil(T_alloc_minutes / step_minutes)
    out = {p2i[p]: np.zeros(nbins, dtype=np.float64) for p in purposes}
    for _, r in acts.iterrows():
        p = p2i.get(str(r["purpose"]), None)
        if p is None: 
            continue
        bins = max(1, int(round(r["total_duration"] / step_minutes)))
        bins = min(bins, nbins)
        out[p][bins - 1] += 1.0
    for p in out:
        s = out[p].sum()
        if s > 0:
            out[p] /= s
    return out  # per purpose hist over duration bins


def duration_proxy_from_soft(theta: torch.Tensor) -> Dict[int, torch.Tensor]:
    """
    Crude duration surrogate per purpose using run-length-like smoothing with 1D conv kernels.
    Returns dict p_idx -> histogram (torch) over proxy duration bins (same L as theta).
    """
    B, P, L = theta.shape
    p = torch.softmax(theta, dim=1)  # [B,P,L]
    # Use cumulative autocorrelation proxy: conv with length-k box filters
    device = theta.device
    out: Dict[int, torch.Tensor] = {}
    # Build box filters 1..L, but to keep it cheap, sample a subset of window sizes
    # e.g., 1,2,3,4,6,8,12,16,24,32,... up to L
    ks = [1,2,3,4,6,8,12,16,24,32]
    ks = [k for k in ks if k <= L]
    for p_idx in range(P):
        sig = p[:, p_idx, :].mean(dim=0, keepdim=True).unsqueeze(0)  # [1,1,L]
        scores = []
        for k in ks:
            kernel = torch.ones((1,1,k), device=device) / k
            # padding so length preserved
            pad = (k - 1) // 2
            sm = torch.nn.functional.conv1d(sig, kernel, padding=pad)  # [1,1,L]
            scores.append(sm.squeeze())  # [L]
        # aggregate per k into a simple histogram proxy by taking global max over t
        hist = torch.tensor([s.max().item() for s in scores], device=device, dtype=torch.float32)
        hist = hist / (hist.sum() + 1e-9)
        out[p_idx] = hist  # len(ks)
    return out, ks


def duration_proxy_mse(
    proxy_hist: Dict[int, torch.Tensor],
    proxy_bins: List[int],
    real_hist: Dict[int, np.ndarray],
) -> torch.Tensor:
    """
    MSE between proxy duration (downsampled to proxy_bins) and real_hist resampled to same bins.
    """
    # Convert real_hist to proxy bins by grouping ranges
    losses = []
    for p_idx, ph in proxy_hist.items():
        # map real histogram (length L) to len(proxy_bins)
        rh = real_hist.get(p_idx, None)
        if rh is None or rh.sum() <= 0:
            continue
        Lr = len(rh)
        # indices in rh corresponding to proxy_bins by nearest ratio
        # simple pooling: split rh into len(proxy_bins) chunks
        chunks = np.array_split(rh, len(proxy_bins))
        pooled = np.array([c.sum() for c in chunks], dtype=np.float32)
        pooled = pooled / (pooled.sum() + 1e-9)
        pooled_t = torch.tensor(pooled, device=ph.device, dtype=torch.float32)
        losses.append(torch.mean((ph - pooled_t) ** 2))
    if not losses:
        return torch.tensor(0.0, device=list(proxy_hist.values())[0].device if proxy_hist else "cpu")
    return torch.stack(losses).mean()


# -------------------------
# training
# -------------------------
@click.command()
@click.option("--activities_csv", type=click.Path(exists=True), required=True)
@click.option("--purposes_csv", type=click.Path(exists=True), required=True)
@click.option("--outdir", type=click.Path(), default="./runs_vae")
@click.option("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
@click.option("--epochs", type=int, default=200)
@click.option("--batch_size", type=int, default=64)
@click.option("--lr", type=float, default=1e-3)
@click.option("--val_ratio", type=float, default=0.2)
@click.option("--mode", type=click.Choice(["recon", "recon+marginals"]), default="recon")
@click.option("--weights_tod", type=float, default=0.2, help="λ for time-of-day marginal loss")
@click.option("--weights_dur", type=float, default=0.1, help="λ for duration proxy loss")
@click.option("--weights_presence", type=float, default=0.05, help="λ for non-Home presence loss")
def main(
    activities_csv: str,
    purposes_csv: str,
    outdir: str,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_ratio: float,
    mode: str,
    weights_tod: float,
    weights_dur: float,
    weights_presence: float,
):
    set_seed(42)
    dev = torch.device(device)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # configs (reuse your existing default values)
    time_cfg  = TimeConfig()
    basis_cfg = BasisConfig()
    pep_cfg   = PurposeEmbeddingConfig()
    dec_cfg   = DecoderConfig()
    vae_cfg   = VAEConfig()

    # data
    acts = pd.read_csv(activities_csv)
    purp = pd.read_csv(purposes_csv)
    purposes = purp["purpose"].astype(str).tolist()
    P = len(purposes)
    p2i = {p: i for i,p in enumerate(purposes)}
    home_idx = p2i.get("Home", 0)

    # empirical targets (eval grid for reporting; train grid for loss)
    real_tod_eval = compute_real_time_marginals_from_activities(
        activities_csv, purposes, time_cfg.ALLOCATION_HORIZON_MINS, time_cfg.VALID_GRID_MINS
    )  # [P, L_eval]
    real_tod_train = match_grid(real_tod_eval, math.ceil(time_cfg.ALLOCATION_HORIZON_MINS / time_cfg.TRAIN_GRID_MINS))  # [P, L_train]
    real_tod_train_t = torch.tensor(real_tod_train, dtype=torch.float32, device=dev)

    # presence target: fraction of people with ≥1 non-Home activity
    ppl = acts["persid"].unique()
    has_nonhome = (
        acts.assign(is_nonhome=(acts["purpose"].astype(str) != "Home").astype(int))
            .groupby("persid")["is_nonhome"].max()
    )
    presence_target = float((has_nonhome > 0).mean())

    # duration hist target
    real_dur_hist = duration_hist_from_activities(
        activities_csv, purposes, step_minutes=time_cfg.TRAIN_GRID_MINS, T_alloc_minutes=time_cfg.ALLOCATION_HORIZON_MINS
    )

    # priors for PDS
    from ananke_abm.models.traj_syn.core.pds_loader import derive_priors_from_activities
    priors = derive_priors_from_activities(
        acts, purp,
        T_alloc_minutes=time_cfg.ALLOCATION_HORIZON_MINS,
        K_clock_prior=basis_cfg.K_clock_prior,
        T_clock_minutes=time_cfg.T_clock_minutes,
    )
    phi_t = build_phi_features(purposes, priors, dev)

    # PDS / grids
    pds = PurposeDistributionSpace(phi_t, d_p=pep_cfg.d_p, hidden=pep_cfg.hidden).to(dev)
    pds.set_clock_prior_K(basis_cfg.K_clock_prior)

    t_alloc_minutes_train, _ = make_alloc_grid(
        T_alloc_minutes=time_cfg.ALLOCATION_HORIZON_MINS,
        step_minutes=time_cfg.TRAIN_GRID_MINS,
        device=dev,
    )
    L_train = t_alloc_minutes_train.numel()
    t_alloc_minutes_eval, _ = make_alloc_grid(
        T_alloc_minutes=time_cfg.ALLOCATION_HORIZON_MINS,
        step_minutes=time_cfg.VALID_GRID_MINS,
        device=dev,
    )
    L_eval = t_alloc_minutes_eval.numel()

    with torch.no_grad():
        loglam_train = pds.lambda_log_on_alloc_grid(t_alloc_minutes_train, T_clock_minutes=time_cfg.T_clock_minutes)  # [P, L_train]
        loglam_eval  = pds.lambda_log_on_alloc_grid(t_alloc_minutes_eval,  T_clock_minutes=time_cfg.T_clock_minutes)  # [P, L_eval]

    # models
    enc = TrajEncoderGRU(
        d_p=pep_cfg.d_p,
        K_time_token_clock=4,
        K_time_token_alloc=0,
        K_dur_token=4,
        m_latent=vae_cfg.latent_dim,
        gru_hidden=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=False
    ).to(dev)

    # per-purpose alpha init (kept from your full trainer)
    alpha_init_per_purpose = {
        "Home": 1.6, "Work": 1.1, "Education": 1.1,
        "Shopping": 0.9, "Social": 0.8, "Accompanying": 0.9, "Other": 0.7,
    }
    dec = TimeFieldDecoder(
        P=P, m_latent=vae_cfg.latent_dim, d_p=pep_cfg.d_p,
        K_decoder_time=basis_cfg.K_decoder_time, alpha_prior=dec_cfg.alpha_prior,
        time_cfg=vars(time_cfg),
        idx2purpose=purposes,
        alpha_init_per_purpose=alpha_init_per_purpose,
        alpha_l2=1e-3,
        coeff_l2_global=dec_cfg.reg_cfg.coeff_l2_global,
        coeff_l2_per_purpose=dec_cfg.reg_cfg.coeff_l2_per_purpose,
    ).to(dev)

    params = list(pds.parameters()) + list(enc.parameters()) + list(dec.parameters())
    opt = optim.Adam(params, lr=lr, weight_decay=1e-4)

    # dataset / loaders
    ds = ScheduleDataset(activities_csv, T_alloc_minutes=time_cfg.ALLOCATION_HORIZON_MINS)
    def split_indices(n, val_ratio, seed=42):
        idx = list(range(n))
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        n_val = int(round(n * val_ratio))
        return idx[n_val:], idx[:n_val]
    tr_idx, va_idx = split_indices(len(ds), val_ratio=val_ratio)
    tr_subset = torch.utils.data.Subset(ds, tr_idx)
    va_subset = torch.utils.data.Subset(ds, va_idx)

    purpose_to_idx = {p: i for i, p in enumerate(purposes)}
    def loader(subset, shuffle):
        return torch.utils.data.DataLoader(
            subset, batch_size=batch_size, shuffle=shuffle,
            collate_fn=lambda batch: collate_fn(batch, purpose_to_idx),
            num_workers=0, pin_memory=False,
        )
    dl_tr = loader(tr_subset, True)
    dl_va = loader(va_subset, False)

    # training
    history = {"epoch": [], "train_total": [], "val_total": [], "rec": [], "kl": [], "tod": [], "dur": [], "presence": []}

    def beta_at_epoch(ep: int) -> float:
        if vae_cfg.kl_anneal_end and ep < vae_cfg.kl_anneal_end:
            return vae_cfg.beta * (ep / max(1, vae_cfg.kl_anneal_end))
        return vae_cfg.beta

    click.echo(f"VAE training (mode={mode}) on {device} for {epochs} epochs.")
    for epoch in range(1, epochs + 1):
        enc.train(); dec.train(); pds.train()
        tot, rec_sum, kl_sum, tod_sum, dur_sum, pres_sum, nb = 0.0,0.0,0.0,0.0,0.0,0.0,0
        beta = beta_at_epoch(epoch)

        for p_pad, t_pad, d_pad, lengths in dl_tr:
            p_pad = p_pad.to(dev); t_pad = t_pad.to(dev); d_pad = d_pad.to(dev)
            e_p = pds()
            z, s, mu, logvar = enc(
                p_pad, t_pad, d_pad, lengths, e_p,
                T_alloc_minutes=time_cfg.ALLOCATION_HORIZON_MINS,
                T_clock_minutes=time_cfg.T_clock_minutes,
                sample=True,
            )
            theta = dec.utilities_on_grid(z, e_p, loglam_train, grid_type="train", endpoint_mask=None)  # [B,P,L_train]
            theta = sanitize_theta(theta)

            # CE recon
            y_grid = rasterize_from_padded_to_grid(p_pad, t_pad, d_pad, lengths, L=L_train, fallback_idx=0)  # [B,L]
            rec = top1_ce_loss(theta, y_grid)

            # KL + alpha regularizer
            kl = kl_gaussian_standard(mu, logvar, reduction="mean")
            reg = dec.regularization_loss()

            # Extra losses (optional)
            L_tod = torch.tensor(0.0, device=dev)
            L_dur = torch.tensor(0.0, device=dev)
            L_pres = torch.tensor(0.0, device=dev)

            if mode == "recon+marginals":
                # time-of-day marginal loss
                p_batch = batch_time_marginal(theta)                             # [P,L_train]
                L_tod = jsd_batch(p_batch, real_tod_train_t)

                # duration proxy loss
                proxy, k_bins = duration_proxy_from_soft(theta)
                L_dur = duration_proxy_mse(proxy, k_bins, real_dur_hist)

                # presence loss
                L_pres = nonhome_presence_loss(theta, home_idx, presence_target)

            loss = rec + beta * kl + reg + weights_tod * L_tod + weights_dur * L_dur + weights_presence * L_pres

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            tot += float(loss.item()); rec_sum += float(rec.item()); kl_sum += float(kl.item())
            tod_sum += float(L_tod.item()); dur_sum += float(L_dur.item()); pres_sum += float(L_pres.item())
            nb += 1

        # validation (monitor only CE+KL+reg and TOD fit)
        enc.eval(); dec.eval(); pds.eval()
        with torch.no_grad():
            tot_v, rec_v, kl_v, tod_v, nb_v = 0.0,0.0,0.0,0.0,0
            e_p = pds()
            for p_pad, t_pad, d_pad, lengths in dl_va:
                p_pad = p_pad.to(dev); t_pad = t_pad.to(dev); d_pad = d_pad.to(dev)
                z, s, mu, logvar = enc(
                    p_pad, t_pad, d_pad, lengths, e_p,
                    T_alloc_minutes=time_cfg.ALLOCATION_HORIZON_MINS,
                    T_clock_minutes=time_cfg.T_clock_minutes,
                    sample=False,
                )
                theta = dec.utilities_on_grid(z, e_p, loglam_train, grid_type="train", endpoint_mask=None)
                theta = sanitize_theta(theta)

                y_grid = rasterize_from_padded_to_grid(p_pad, t_pad, d_pad, lengths, L=L_train, fallback_idx=0)
                rec = top1_ce_loss(theta, y_grid)
                kl  = kl_gaussian_standard(mu, logvar, reduction="mean")
                reg = dec.regularization_loss()
                p_batch = batch_time_marginal(theta)
                L_tod = jsd_batch(p_batch, real_tod_train_t)
                loss = rec + beta * kl + reg + (weights_tod * L_tod if mode=="recon+marginals" else 0.0)

                tot_v += float(loss.item()); rec_v += float(rec.item()); kl_v += float(kl.item()); tod_v += float(L_tod.item()); nb_v += 1

        # logging
        train_tot = tot / max(nb,1)
        val_tot   = tot_v / max(nb_v,1)
        if epoch % 10 == 0 or epoch == epochs:
            click.echo(f"[{epoch:03d}] train={train_tot:.4f}  val={val_tot:.4f}  (rec={rec_v/max(nb_v,1):.4f}, kl={kl_v/max(nb_v,1):.4f}, tod={tod_v/max(nb_v,1):.4f}, beta={beta:.3f})")

        history["epoch"].append(epoch)
        history["train_total"].append(train_tot)
        history["val_total"].append(val_tot)
        history["rec"].append(rec_v / max(nb_v,1))
        history["kl"].append(kl_v / max(nb_v,1))
        history["tod"].append(tod_v / max(nb_v,1))
        history["dur"].append(dur_sum / max(nb,1))
        history["presence"].append(pres_sum / max(nb,1))

        # save checkpoint & diagnostics periodically
        if epoch % 1000 == 0 or epoch == epochs:
            ckpt = {
                "epoch": epoch,
                "model_state": {"pds": pds.state_dict(), "enc": enc.state_dict(), "dec": dec.state_dict()},
                "purposes": purposes,
                "configs": {
                    "time": vars(time_cfg),
                    "basis": vars(basis_cfg),
                    "pep": vars(pep_cfg),
                    "dec": vars(dec_cfg),
                    "vae": vars(vae_cfg),
                },
            }
            torch.save(ckpt, outdir / f"vae_ep{epoch:04d}.pt")

            # Save decoder average curves on eval grid
            with torch.no_grad():
                # sample a big batch of z to estimate population curves
                Bz = 512
                z = torch.randn(Bz, vae_cfg.latent_dim, device=dev)
                z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
                e_p = pds()
                theta_eval = dec.utilities_on_grid(z, e_p, loglam_eval, grid_type="eval", endpoint_mask=None)
                p_eval = torch.softmax(sanitize_theta(theta_eval), dim=1).mean(0).detach().cpu().numpy()  # [P,L_eval]
                np.save(outdir / f"vae_mean_probs_ep{epoch:04d}.npy", p_eval)

    # save history
    pd.DataFrame(history).to_csv(outdir / "vae_history.csv", index=False)


if __name__ == "__main__":
    main()
