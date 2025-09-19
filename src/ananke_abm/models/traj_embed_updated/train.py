import click
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

# --- configs & model pieces ---
from ananke_abm.models.traj_embed_updated.configs import (
    TimeConfig,
    BasisConfig,
    PurposeEmbeddingConfig,
    DecoderConfig,
    VAEConfig,
    CRFConfig,
)
from ananke_abm.models.traj_embed_updated.model.pds_loader import derive_priors_from_activities
from ananke_abm.models.traj_embed_updated.model.purpose_space import PurposeDistributionSpace
from ananke_abm.models.traj_embed_updated.model.utils_bases import make_alloc_grid, merge_primary_slivers, segments_from_padded_to_grid
from ananke_abm.models.traj_embed_updated.model.rasterize import rasterize_from_padded_to_grid
from ananke_abm.models.traj_embed_updated.model.decoder_timefield import TimeFieldDecoder
from ananke_abm.models.traj_embed_updated.model.encoder import TrajEncoderGRU, kl_gaussian_standard
from ananke_abm.models.traj_embed_updated.model.crf_linear import LinearChainCRF
from ananke_abm.models.traj_embed_updated.model.crf_semi import SemiMarkovCRF, build_duration_logprob_table
from ananke_abm.models.traj_embed_updated.model.train_masks import build_endpoint_mask, endpoint_time_mask


# -------------------
# utils & dataset
# -------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ScheduleDataset(torch.utils.data.Dataset):
    """
    Reads activities and forms per-person sequences of (purpose, start_norm, dur_norm),
    normalized by the ALLOCATION horizon (e.g., 30h).
    """
    def __init__(self, activities_csv: str, T_alloc_minutes: int):
        acts = pd.read_csv(activities_csv)
        self.T = float(T_alloc_minutes)
        self.seqs = []
        for _, g in acts.groupby("persid"):
            g = g.sort_values(["startime", "stopno"])
            day = [
                (str(r["purpose"]), float(r["startime"] / self.T), float(r["total_duration"] / self.T))
                for _, r in g.iterrows()
            ]
            self.seqs.append(day)

    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx): return self.seqs[idx]


def split_indices(n, val_ratio=0.2, seed=42):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = int(round(n * val_ratio))
    return idx[n_val:], idx[:n_val]


def collate_fn(batch, purpose_to_idx):
    """
    Pack a batch of variable-length sequences into padded tensors.
    Returns:
        p_pad: [B, Lmax] long
        t_pad: [B, Lmax] float in [0,1] (allocation-normalized start)
        d_pad: [B, Lmax] float in [0,1] (allocation-normalized duration)
        lengths: list[int] valid lengths
    """
    p_lists, t_lists, d_lists, lens = [], [], [], []
    for seq in batch:
        p_idx = [purpose_to_idx[p] for p, _, _ in seq]
        t0 = [t for _, t, _ in seq]
        dd = [d for _, _, d in seq]
        p_lists.append(torch.tensor(p_idx, dtype=torch.long))
        t_lists.append(torch.tensor(t0, dtype=torch.float32))
        d_lists.append(torch.tensor(dd, dtype=torch.float32))
        lens.append(len(seq))
    p_pad = pad_sequence(p_lists, batch_first=True, padding_value=0)
    t_pad = pad_sequence(t_lists, batch_first=True, padding_value=0.0)
    d_pad = pad_sequence(d_lists, batch_first=True, padding_value=0.0)
    return p_pad, t_pad, d_pad, lens


# -------------------
# training script
# -------------------
def sanitize_theta(theta: torch.Tensor) -> torch.Tensor:
    """Numerically stabilizes CRF emissions."""
    theta_max = torch.max(theta, dim=1, keepdim=True).values
    theta_stable = theta - theta_max
    return torch.clamp(theta_stable, min=-30.0)

def train_traj_embed(
    activities_csv: str,
    purposes_csv: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_ratio: float = 0.2,
    outdir: str = "./runs",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    crf_mode: str = "linear",  # "linear" or "semi"
):
    set_seed(42)
    click.echo(f"Training for {epochs} epochs...")
    click.echo(f"Using device: {device}")

    device = torch.device(device)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- configs ---
    time_cfg  = TimeConfig()
    basis_cfg = BasisConfig()
    pep_cfg   = PurposeEmbeddingConfig()
    dec_cfg   = DecoderConfig()
    vae_cfg   = VAEConfig()
    crf_cfg   = CRFConfig()

    # --- data & priors (24h clock prior; durations by allocation) ---
    acts = pd.read_csv(activities_csv)
    purp = pd.read_csv(purposes_csv)
    purposes = purp["purpose"].tolist()
    priors = derive_priors_from_activities(
        acts,
        purp,
        T_alloc_minutes=time_cfg.ALLOCATION_HORIZON_MINS,
        K_clock_prior=basis_cfg.K_clock_prior,
        T_clock_minutes=time_cfg.T_clock_minutes,
    )
    assert len(priors) == len(purposes)
    is_primary = torch.tensor(
        purp["is_primary_ooh"].fillna(0).astype(int).to_numpy(),
        dtype=torch.bool, device=device
    )
    tau_bins = max(1, int(round(60 / time_cfg.TRAIN_GRID_MINS)))

    # feature matrix phi_p = [Fourier_clock | mu_t | sigma_t | mu_d | sigma_d], standardized
    rows = []
    for p in purposes:
        pr = priors[p]
        rows.append(np.concatenate([pr.time_fourier, [pr.mu_t, pr.sigma_t, pr.mu_d, pr.sigma_d, pr.is_primary_ooh]]).astype("float32"))
    phi = np.stack(rows, axis=0)
    phi = (phi - phi.mean(0, keepdims=True)) / (phi.std(0, keepdims=True) + 1e-6)
    phi_t = torch.tensor(phi, dtype=torch.float32, device=device)

    # --- PDS (embeddings + clock prior accessors) ---
    pds = PurposeDistributionSpace(phi_t, d_p=pep_cfg.d_p, hidden=pep_cfg.hidden).to(device)
    pds.set_clock_prior_K(basis_cfg.K_clock_prior)

    # --- allocation grids & priors (train/eval) ---
    loglam_grids = {}
    for grid_type, step_mins in [("train", time_cfg.TRAIN_GRID_MINS), ("eval", time_cfg.VALID_GRID_MINS)]:
        t_alloc_minutes, _ = make_alloc_grid(
            T_alloc_minutes=time_cfg.ALLOCATION_HORIZON_MINS,
            step_minutes=step_mins,
            device=device,
        )
        with torch.no_grad():
            loglam_grids[grid_type] = pds.lambda_log_on_alloc_grid(
                t_alloc_minutes, T_clock_minutes=time_cfg.T_clock_minutes
            ) # [P, L]

    if crf_mode == "semi":
        dur_logprob = build_duration_logprob_table(
            priors=priors,
            purposes=purposes,
            step_minutes=time_cfg.TRAIN_GRID_MINS,
            T_alloc_minutes=time_cfg.T_clock_minutes,
            Dmax_minutes=getattr(crf_cfg, "semi_Dmax_minutes", 300),  # default 5h,
            device=device,
        )  # [P, Dmax_bins]

    # --- unified endpoint mask ---
    ep_masks = build_endpoint_mask(purp, purposes, can_open_col="can_open_day", can_close_col="can_close_day")
    L_train = loglam_grids["train"].shape[1]
    endpoint_mask_train = endpoint_time_mask(ep_masks.open_allowed, ep_masks.close_allowed, L_train, step_mins=time_cfg.TRAIN_GRID_MINS, device=device)  # [L,P]

    # --- models ---
    P = len(purposes)
    dec = TimeFieldDecoder(
        P=P,
        m_latent=vae_cfg.latent_dim,
        d_p=pep_cfg.d_p,
        K_decoder_time=basis_cfg.K_decoder_time,
        alpha_prior=dec_cfg.alpha_prior,
        time_cfg=vars(time_cfg),
    ).to(device)

    enc = TrajEncoderGRU(
        d_p=pep_cfg.d_p,
        K_time_token_clock=4,
        K_time_token_alloc=0,    # optional; set >0 to include alloc-position Fourier tokens
        K_dur_token=4,
        m_latent=vae_cfg.latent_dim,
        gru_hidden=64,
        num_layers=1,
        dropout=0.2
    ).to(device)

    if crf_mode == "linear":
        crf = LinearChainCRF(
            P=P,
            eta=crf_cfg.eta,
            learn_eta=crf_cfg.learn_eta,
            transition_mask=None,   # plug in a [P,P] Bool mask if you want to forbid transitions
        ).to(device)
    elif crf_mode == "semi":
        crf = SemiMarkovCRF(
            P=P,
            eta=crf_cfg.eta,
            learn_eta=crf_cfg.learn_eta,
        ).to(device)

    params = list(pds.parameters()) + list(enc.parameters()) + list(dec.parameters())
    if crf_cfg.learn_eta:
        params += list(crf.parameters())
    opt = optim.Adam(params, lr=lr, weight_decay=1e-4)

    # --- dataset / loaders ---
    ds = ScheduleDataset(activities_csv, T_alloc_minutes=time_cfg.ALLOCATION_HORIZON_MINS)
    tr_idx, va_idx = split_indices(len(ds), val_ratio=val_ratio, seed=42)
    tr_subset = torch.utils.data.Subset(ds, tr_idx)
    va_subset = torch.utils.data.Subset(ds, va_idx)

    purpose_to_idx = {p: i for i, p in enumerate(purposes)}

    def loader(subset, shuffle):
        return torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: collate_fn(batch, purpose_to_idx),
            num_workers=0,   # tune on your machine
            pin_memory=False,
        )

    dl_tr = loader(tr_subset, True)
    dl_va = loader(va_subset, False)

    # --- training loop ---
    history = {"epoch": [], "train_total": [], "val_total": [], "nll": [], "kl": []}
    best_val_total = float("inf")

    def beta_at_epoch(ep: int) -> float:
        if vae_cfg.kl_anneal_end and ep < vae_cfg.kl_anneal_end:
            # optional linear warmup; start at beta * (ep / end)
            return vae_cfg.beta * (ep / max(1, vae_cfg.kl_anneal_end))
        return vae_cfg.beta

    for epoch in range(1, epochs + 1):
        enc.train(); dec.train(); pds.train(); crf.train()
        total_tr = 0.0; nll_tr = 0.0; kl_tr = 0.0

        beta_weight = beta_at_epoch(epoch)

        for i, (p_pad, t_pad, d_pad, lengths) in enumerate(dl_tr):
            p_pad = p_pad.to(device); t_pad = t_pad.to(device); d_pad = d_pad.to(device)
            e_p = pds()  # [P, d_p]

            # encode (β-VAE)
            z, s, mu, logvar = enc(
                p_pad, t_pad, d_pad, lengths, e_p,
                T_alloc_minutes=time_cfg.ALLOCATION_HORIZON_MINS,
                T_clock_minutes=time_cfg.T_clock_minutes,
                sample=True,
            )

            if torch.isnan(z).any():
                click.echo(f"NaN detected in z at batch {i}. Norms: {torch.linalg.norm(s, dim=-1)}")
                # Consider stopping or saving state here
                break

            # unaries θ on train grid (no mask applied here; CRF will apply)
            theta = dec.utilities_on_grid(
                z, e_p, loglam_grids["train"], grid_type="train",
                endpoint_mask=None,  # avoid double-application; CRF handles endpoints
            )  # [B,P,L]
            theta = sanitize_theta(theta)

            if torch.isnan(theta).any():
                click.echo(f"NaN detected in theta at batch {i}.")
                break

            # Log stats periodically
            # if i % 50 == 0:
            #     click.echo(f"Theta stats: min={theta.min():.2f}, max={theta.max():.2f}")
            
            if crf_mode == "linear":
                fallback_idx = 0 #ep_masks.open_allowed.argmax() if ep_masks.open_allowed.any() else 0
                y_grid = rasterize_from_padded_to_grid(p_pad, t_pad, d_pad, lengths, L=L_train, fallback_idx=fallback_idx)  # [B,L]
                y_grid = merge_primary_slivers(y_grid, is_primary, tau_bins)
                nll = crf.nll(theta, y_grid, endpoint_mask=endpoint_mask_train)
            elif crf_mode == "semi":
                y_segs = segments_from_padded_to_grid(p_pad, t_pad, d_pad, lengths, L=L_train)
                nll = crf.nll(
                    theta=theta,
                    gold_segments=y_segs,
                    dur_logprob=dur_logprob,                 # [P, Dmax_bins] on TRAIN grid
                    endpoint_mask=endpoint_mask_train,       # [L,P] on TRAIN grid
                )

            kl  = kl_gaussian_standard(mu, logvar, reduction="mean")
            loss = nll + beta_weight * kl

            # step
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            total_tr += float(loss.item()); nll_tr += float(nll.item()); kl_tr += float(kl.item())

        # ---- validation ----
        enc.eval(); dec.eval(); pds.eval(); crf.eval()
        total_va = 0.0; nll_va = 0.0; kl_va = 0.0; n_batches = 0

        with torch.no_grad():
            e_p = pds()
            for p_pad, t_pad, d_pad, lengths in dl_va:
                p_pad = p_pad.to(device); t_pad = t_pad.to(device); d_pad = d_pad.to(device)

                z, s, mu, logvar = enc(
                    p_pad, t_pad, d_pad, lengths, e_p,
                    T_alloc_minutes=time_cfg.ALLOCATION_HORIZON_MINS,
                    T_clock_minutes=time_cfg.T_clock_minutes,
                    sample=False,  # deterministic μ at val
                )
                theta = dec.utilities_on_grid(
                    z, e_p, loglam_grids["train"], grid_type="train",
                    endpoint_mask=endpoint_mask_train,
                )
                theta = sanitize_theta(theta)

                if crf_mode == "linear":
                    fallback_idx = 0 #ep_masks.open_allowed.argmax() if ep_masks.open_allowed.any() else 0
                    y_grid = rasterize_from_padded_to_grid(p_pad, t_pad, d_pad, lengths, L=L_train, fallback_idx=fallback_idx)
                    y_grid = merge_primary_slivers(y_grid, is_primary, tau_bins)
                    nll = crf.nll(theta, y_grid, endpoint_mask=endpoint_mask_train)
                elif crf_mode == "semi":
                    y_segs = segments_from_padded_to_grid(p_pad, t_pad, d_pad, lengths, L=L_train)
                    nll = crf.nll(
                        theta=theta,
                        gold_segments=y_segs,
                        dur_logprob=dur_logprob,                 # [P, Dmax_bins] on TRAIN grid
                        endpoint_mask=endpoint_mask_train,       # [L,P] on TRAIN grid
                    )
                
                kl  = kl_gaussian_standard(mu, logvar, reduction="mean")
                loss = nll + beta_weight * kl
                total_va += float(loss.item()); nll_va += float(nll.item()); kl_va += float(kl.item()); n_batches += 1

        avg_tr = total_tr / max(len(dl_tr), 1)
        avg_va = total_va / max(n_batches, 1)
        if epoch % 10 == 0 or epoch == epochs:
            click.echo(f"[{epoch:03d}] train={avg_tr:.4f}  val={avg_va:.4f}  (nll={nll_va/max(n_batches,1):.4f}, kl={kl_va/max(n_batches,1):.4f}, beta={beta_weight:.3f})")

        history["epoch"].append(epoch)
        history["train_total"].append(avg_tr)
        history["val_total"].append(avg_va)
        history["nll"].append(nll_va / max(n_batches, 1))
        history["kl"].append(kl_va / max(n_batches, 1))

        # --- checkpoint ---
        ckpt = {
            "epoch": epoch,
            "model_state": {"pds": pds.state_dict(), "enc": enc.state_dict(), "dec": dec.state_dict(), "crf": crf.state_dict()},
            "priors": priors,
            "purposes": purposes,
            "purpose_to_idx": {p: i for i, p in enumerate(purposes)},
            "configs": {
                "time": vars(TimeConfig()),
                "basis": vars(BasisConfig()),
                "pep": vars(PurposeEmbeddingConfig()),
                "dec": vars(DecoderConfig()),
                "vae": vars(VAEConfig()),
                "crf": vars(CRFConfig()),
            },
        }
        if avg_va < best_val_total:
            best_val_total = avg_va
            torch.save(ckpt, outdir / "ckpt_best.pt")
            click.echo(f"Best val so far: {best_val_total:.4f} at epoch {epoch}")

        if epoch == epochs:
            torch.save(ckpt, outdir / "ckpt_final.pt")
            click.echo(f"Final val: {avg_va:.4f} (best {best_val_total:.4f})")

    # Save history to CSV
    pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)
