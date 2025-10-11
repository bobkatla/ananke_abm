import click
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# --- configs & model pieces ---
from ananke_abm.models.traj_syn.configs import (
    TimeConfig,
    BasisConfig,
    PurposeEmbeddingConfig,
    DecoderConfig,
    VAEConfig,
    CRFConfig,
    LossBalanceConfig,
    PairwiseConfig,
)
from ananke_abm.models.traj_syn.eval.pairwise_time_bilinear import TimeVaryingPairwise
from ananke_abm.models.traj_syn.core.pds_loader import derive_priors_from_activities
from ananke_abm.models.traj_syn.vae.purpose_space import PurposeDistributionSpace
from ananke_abm.models.traj_syn.core.utils_bases import make_alloc_grid, segments_from_padded_to_grid
from ananke_abm.models.traj_syn.core.rasterize import rasterize_from_padded_to_grid
from ananke_abm.models.traj_syn.vae.decoder_timefield import TimeFieldDecoder
from ananke_abm.models.traj_syn.vae.encoder import TrajEncoderGRU, kl_gaussian_standard
from ananke_abm.models.traj_syn.crf.crf_linear import LinearChainCRF
from ananke_abm.models.traj_syn.crf.crf_semi import SemiMarkovCRF, build_duration_logprob_table
from ananke_abm.models.traj_syn.core.train_masks import build_endpoint_mask, endpoint_time_mask
from ananke_abm.models.traj_syn.core.data_utils.randomness import set_seed
from ananke_abm.models.traj_syn.core.data_utils.sanitize import sanitize_theta
from ananke_abm.models.traj_syn.core.data_utils.ScheduleDataset import ScheduleDataset, collate_fn


K_TIME_TOKEN_CLOCK = 4
K_TIME_TOKEN_ALLOC = 0
K_DUR_TOKEN = 4
GRU_HIDDEN = 128
NUM_LAYERS = 2
DROPOUT = 0.2
BIDIRECTIONAL = False


def split_indices(n: int, val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
    """Split n items into train/val indices."""
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = int(round(n * val_ratio))
    return idx[n_val:], idx[:n_val]

# -------------------
# training script
# -------------------
def build_phi_features(purposes: List[str], priors: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """Build standardized feature matrix per purpose and return tensor on device."""
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

def build_allocation_and_endpoint(
    priors: Dict[str, Any],
    purp: pd.DataFrame,
    purposes: List[str],
    pds: PurposeDistributionSpace,
    time_cfg: TimeConfig,
    crf_mode: str,
    crf_cfg: CRFConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """Create allocation grids, duration tables (if semi), and endpoint masks."""
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
            )  # [P, L]

    L_train = loglam_grids["train"].shape[1]
    L_eval  = loglam_grids["eval"].shape[1]

    _, t_alloc01_train = make_alloc_grid(
        T_alloc_minutes=time_cfg.ALLOCATION_HORIZON_MINS,
        step_minutes=time_cfg.TRAIN_GRID_MINS,
        device=device,
    )
    _,  t_alloc01_eval  = make_alloc_grid(
        T_alloc_minutes=time_cfg.ALLOCATION_HORIZON_MINS,
        step_minutes=time_cfg.VALID_GRID_MINS,
        device=device,
    )

    dur_logprob_train = None
    dur_logprob_eval = None
    if crf_mode == "semi":
        dur_logprob_train = build_duration_logprob_table(
            priors=priors,
            purposes=purposes,
            step_minutes=time_cfg.TRAIN_GRID_MINS,
            T_alloc_minutes=time_cfg.ALLOCATION_HORIZON_MINS,
            Dmax_minutes=getattr(crf_cfg, "semi_Dmax_minutes", 300),
            device=device,
        )  # [P, Dmax_bins_train]
        assert dur_logprob_train.shape[0] == len(purposes)

        dur_logprob_eval = build_duration_logprob_table(
            priors=priors,
            purposes=purposes,
            step_minutes=time_cfg.VALID_GRID_MINS,
            T_alloc_minutes=time_cfg.ALLOCATION_HORIZON_MINS,
            Dmax_minutes=getattr(crf_cfg, "semi_Dmax_minutes", 300),
            device=device,
        )  # [P, Dmax_bins_eval]
        assert dur_logprob_eval.shape[0] == len(purposes)

    # --- unified endpoint mask ---
    ep_masks = build_endpoint_mask(purp, purposes, can_open_col="can_open_day", can_close_col="can_close_day")
    endpoint_mask_train = endpoint_time_mask(
        ep_masks.open_allowed, ep_masks.close_allowed,
        L_train, step_mins=time_cfg.TRAIN_GRID_MINS, device=device
    )  # [L_train, P]
    assert endpoint_mask_train.shape == (L_train, len(purposes))

    endpoint_mask_eval = endpoint_time_mask(
        ep_masks.open_allowed, ep_masks.close_allowed,
        L_eval, step_mins=time_cfg.VALID_GRID_MINS, device=device
    )  # [L_eval, P]
    assert endpoint_mask_eval.shape == (L_eval, len(purposes))

    return {
        "loglam_grids": loglam_grids,
        "L_train": L_train,
        "L_eval": L_eval,
        "t_alloc01_train": t_alloc01_train,
        "t_alloc01_eval": t_alloc01_eval,
        "dur_logprob_train": dur_logprob_train,
        "dur_logprob_eval": dur_logprob_eval,
        "endpoint_mask_train": endpoint_mask_train,
        "endpoint_mask_eval": endpoint_mask_eval,
    }

def compute_nll_and_weight_linear(
    crf: LinearChainCRF,
    theta: torch.Tensor,
    y_grid: torch.Tensor,
    endpoint_mask: torch.Tensor,
    pairwise_logits: Optional[torch.Tensor],
    loss_w_vec: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    nll = crf.nll(theta, y_grid, endpoint_mask=endpoint_mask, pairwise_logits=pairwise_logits) / max(theta.shape[-1], 1)
    if loss_w_vec is None:
        return nll, torch.tensor(1.0, device=theta.device)
    with torch.no_grad():
        w_tokens = loss_w_vec[y_grid]  # [B,L]
        w_factor = w_tokens.mean()     # scalar
    return nll * w_factor, w_factor

def compute_nll_and_weight_semi(
    crf: SemiMarkovCRF,
    theta: torch.Tensor,
    y_segs: Any,
    dur_logprob: torch.Tensor,
    endpoint_mask: torch.Tensor,
    loss_w_vec: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    nll = crf.nll(
        theta=theta,
        gold_segments=y_segs,
        dur_logprob=dur_logprob,
        endpoint_mask=endpoint_mask,
    ) / max(theta.shape[-1], 1)
    if loss_w_vec is None:
        return nll, torch.tensor(1.0, device=theta.device)
    with torch.no_grad():
        tot_w, tot_cnt = 0.0, 0
        for segs in y_segs:
            for (p_idx, t0_bin, d_bins) in segs:
                tot_w += float(loss_w_vec[p_idx])
                tot_cnt += 1
        w_factor = torch.tensor(tot_w / max(tot_cnt, 1), device=theta.device, dtype=torch.float32)
    return nll * w_factor, w_factor

def build_checkpoint(
    pds: PurposeDistributionSpace,
    enc: TrajEncoderGRU,
    dec: TimeFieldDecoder,
    crf: Union[LinearChainCRF, SemiMarkovCRF],
    pairwise_mod: Optional[TimeVaryingPairwise],
    priors: Dict[str, Any],
    purposes: List[str],
    time_cfg: TimeConfig,
    basis_cfg: BasisConfig,
    pep_cfg: PurposeEmbeddingConfig,
    dec_cfg: DecoderConfig,
    vae_cfg: VAEConfig,
    crf_cfg: CRFConfig,
    pair_cfg: PairwiseConfig,
    epoch: int,
) -> Dict[str, Any]:
    ckpt = {
        "epoch": epoch,
        "model_state": {
            "pds": pds.state_dict(),
            "enc": enc.state_dict(),
            "dec": dec.state_dict(),
            "crf": crf.state_dict(),
            "pairwise": (pairwise_mod.state_dict() if pairwise_mod is not None else None),
        },
        "priors": priors,
        "purposes": purposes,
        "purpose_to_idx": {p: i for i, p in enumerate(purposes)},
        "configs": {
            "time": vars(time_cfg),
            "basis": vars(basis_cfg),
            "pep": vars(pep_cfg),
            "dec": vars(dec_cfg),
            "vae": vars(vae_cfg),
            "crf": vars(crf_cfg),
            "pairwise": vars(pair_cfg),
            # persist encoder arch so validate.py can rebuild the same module
            "enc_arch": {
                "d_p": pep_cfg.d_p,
                "K_time_token_clock": K_TIME_TOKEN_CLOCK,
                "K_time_token_alloc": K_TIME_TOKEN_ALLOC,
                "K_dur_token": K_DUR_TOKEN,
                "m_latent": vae_cfg.latent_dim,
                "gru_hidden": GRU_HIDDEN,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "bidirectional": BIDIRECTIONAL,
                "use_token_resmlp": True,
                "token_resmlp_hidden": 256,
                "use_residual_gru": True,
                "use_attn_pool": True,
            },
        },
    }
    return ckpt

def build_models_and_optimizer(
    pds: PurposeDistributionSpace,
    P: int,
    idx2purpose: List[str],
    alpha_init_per_purpose: Dict[str, float],
    alpha_l2_strength: float,
    time_cfg: TimeConfig,
    basis_cfg: BasisConfig,
    pep_cfg: PurposeEmbeddingConfig,
    dec_cfg: DecoderConfig,
    vae_cfg: VAEConfig,
    crf_cfg: CRFConfig,
    pair_cfg: PairwiseConfig,
    device: torch.device,
    t_alloc01_eval: torch.Tensor,
    lr: float,
    crf_mode: str,
) -> Tuple[
    TimeFieldDecoder,
    TrajEncoderGRU,
    Union[LinearChainCRF, SemiMarkovCRF],
    Optional[TimeVaryingPairwise],
    Optional[torch.Tensor],
    List[nn.Parameter],
    optim.Optimizer,
]:
    """Construct decoder, encoder, CRF, optional pairwise module, and optimizer."""
    dec_local = TimeFieldDecoder(
        P=P,
        m_latent=vae_cfg.latent_dim,
        d_p=pep_cfg.d_p,
        K_decoder_time=basis_cfg.K_decoder_time,
        alpha_prior=dec_cfg.alpha_prior,
        time_cfg=vars(time_cfg),
        idx2purpose=idx2purpose,
        alpha_init_per_purpose=alpha_init_per_purpose,
        alpha_l2=alpha_l2_strength,
        coeff_l2_global=dec_cfg.reg_cfg.coeff_l2_global,
        coeff_l2_per_purpose=dec_cfg.reg_cfg.coeff_l2_per_purpose,
    ).to(device)

    enc_local = TrajEncoderGRU(
        d_p=pep_cfg.d_p,
        K_time_token_clock=K_TIME_TOKEN_CLOCK,
        K_time_token_alloc=K_TIME_TOKEN_ALLOC,
        K_dur_token=K_DUR_TOKEN,
        m_latent=vae_cfg.latent_dim,
        bidirectional=BIDIRECTIONAL,
        gru_hidden=GRU_HIDDEN,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    if crf_mode == "linear":
        crf_local = LinearChainCRF(
            P=P,
            eta=crf_cfg.eta,
            learn_eta=crf_cfg.learn_eta,
            transition_mask=None,
        ).to(device)
    else:
        crf_local = SemiMarkovCRF(
            P=P,
            eta=crf_cfg.eta,
            learn_eta=crf_cfg.learn_eta,
        ).to(device)

    pairwise_mod_local = None
    pairwise_eval_local = None
    if pair_cfg.enabled and crf_mode == "linear":
        pairwise_mod_local = TimeVaryingPairwise(
            P=P, rank=pair_cfg.rank, K_clock=pair_cfg.K_clock, scale=pair_cfg.scale
        ).to(device)
        with torch.no_grad():
            pairwise_mod_local.U.mul_(0.1)
            pairwise_mod_local.V.mul_(0.1)
            pairwise_mod_local.W.mul_(0.1)
        with torch.no_grad():
            pairwise_eval_local = pairwise_mod_local(t_alloc01_eval)

    params_local = list(pds.parameters()) + list(enc_local.parameters()) + list(dec_local.parameters())
    if crf_cfg.learn_eta:
        params_local += list(crf_local.parameters())
    if pairwise_mod_local is not None:
        params_local += list(pairwise_mod_local.parameters())
    opt_local = optim.Adam(params_local, lr=lr, weight_decay=1e-4)
    return dec_local, enc_local, crf_local, pairwise_mod_local, pairwise_eval_local, params_local, opt_local

def train_one_epoch(
    dl_tr: torch.utils.data.DataLoader,
    pds: PurposeDistributionSpace,
    enc: TrajEncoderGRU,
    dec: TimeFieldDecoder,
    crf: Union[LinearChainCRF, SemiMarkovCRF],
    pairwise_mod: Optional[TimeVaryingPairwise],
    loglam_grids: Dict[str, torch.Tensor],
    endpoint_mask_train: torch.Tensor,
    loss_w_vec: torch.Tensor,
    time_cfg: TimeConfig,
    L_train: int,
    crf_mode: str,
    device: torch.device,
    beta_weight: float,
    opt: optim.Optimizer,
    params: List[nn.Parameter],
    t_alloc01_train: torch.Tensor,
    dur_logprob_train: Optional[torch.Tensor],
    learn_eta: bool,
) -> Tuple[float, float, float, int]:
    enc.train()
    dec.train()
    pds.train()
    crf.train()
    total_tr = 0.0
    nll_tr = 0.0
    kl_tr = 0.0
    n_batches_train = 0
    for i, (p_pad, t_pad, d_pad, lengths) in enumerate(dl_tr):
        p_pad = p_pad.to(device)
        t_pad = t_pad.to(device)
        d_pad = d_pad.to(device)
        e_p = pds()  # [P, d_p]
        z, s, mu, logvar = enc(
            p_pad, t_pad, d_pad, lengths, e_p,
            T_alloc_minutes=time_cfg.ALLOCATION_HORIZON_MINS,
            T_clock_minutes=time_cfg.T_clock_minutes,
            sample=True,
        )
        if torch.isnan(z).any():
            click.echo(f"NaN detected in z at batch {i}. Norms: {torch.linalg.norm(s, dim=-1)}")
            break
        theta = dec.utilities_on_grid(
            z, e_p, loglam_grids["train"], grid_type="train",
            endpoint_mask=None,
        )  # [B,P,L]
        theta = sanitize_theta(theta)
        B, P, Lg = theta.shape
        assert endpoint_mask_train.shape == (Lg, P)
        if torch.isnan(theta).any():
            click.echo(f"NaN detected in theta at batch {i}.")
            break
        if crf_mode == "linear":
            fallback_idx = 0
            y_grid = rasterize_from_padded_to_grid(p_pad, t_pad, d_pad, lengths, L=L_train, fallback_idx=fallback_idx)  # [B,L]
            # Compute pairwise logits inside the loop so each batch has a fresh graph
            pairwise_train = pairwise_mod(t_alloc01_train) if pairwise_mod is not None else None
            nll, _ = compute_nll_and_weight_linear(crf, theta, y_grid, endpoint_mask_train, pairwise_train, loss_w_vec)
        else:
            y_segs = segments_from_padded_to_grid(p_pad, t_pad, d_pad, lengths, L=L_train)
            nll, _ = compute_nll_and_weight_semi(crf, theta, y_segs, dur_logprob_train, endpoint_mask_train, loss_w_vec)
        kl  = kl_gaussian_standard(mu, logvar, reduction="mean")
        loss = nll + beta_weight * kl
        loss = loss + dec.regularization_loss()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if learn_eta and hasattr(crf, "log_eta") and crf.log_eta is not None:
            with torch.no_grad():
                crf.log_eta.data.clamp_(min=np.log(0.1), max=np.log(1.5))
        total_tr += float(loss.item())
        nll_tr += float(nll.item())
        kl_tr += float(kl.item())
        n_batches_train += 1
    return total_tr, nll_tr, kl_tr, n_batches_train

def validate_one_epoch(
    dl_va: torch.utils.data.DataLoader,
    pds: PurposeDistributionSpace,
    enc: TrajEncoderGRU,
    dec: TimeFieldDecoder,
    crf: Union[LinearChainCRF, SemiMarkovCRF],
    pairwise_eval: Optional[torch.Tensor],
    loglam_grids: Dict[str, torch.Tensor],
    endpoint_mask_eval: torch.Tensor,
    time_cfg: TimeConfig,
    L_eval: int,
    crf_mode: str,
    device: torch.device,
    beta_weight: float,
    dur_logprob_eval: Optional[torch.Tensor],
) -> Tuple[float, float, float, int]:
    enc.eval()
    dec.eval()
    pds.eval()
    crf.eval()
    total_va = 0.0
    nll_va = 0.0
    kl_va = 0.0
    n_batches_val = 0
    with torch.no_grad():
        e_p = pds()
        for p_pad, t_pad, d_pad, lengths in dl_va:
            p_pad = p_pad.to(device)
            t_pad = t_pad.to(device)
            d_pad = d_pad.to(device)
            z, s, mu, logvar = enc(
                p_pad, t_pad, d_pad, lengths, e_p,
                T_alloc_minutes=time_cfg.ALLOCATION_HORIZON_MINS,
                T_clock_minutes=time_cfg.T_clock_minutes,
                sample=False,  # deterministic μ at val
            )
            theta = dec.utilities_on_grid(
                z, e_p, loglam_grids["eval"], grid_type="eval",
                endpoint_mask=None,
            )
            theta = sanitize_theta(theta)
            B, P, Lg = theta.shape
            assert endpoint_mask_eval.shape == (Lg, P)
            if crf_mode == "linear":
                fallback_idx = 0
                y_grid = rasterize_from_padded_to_grid(p_pad, t_pad, d_pad, lengths, L=L_eval, fallback_idx=fallback_idx)
                nll = crf.nll(theta, y_grid, endpoint_mask=endpoint_mask_eval, pairwise_logits=pairwise_eval) / max(theta.shape[-1], 1)
            else:
                y_segs = segments_from_padded_to_grid(p_pad, t_pad, d_pad, lengths, L=L_eval)
                nll = crf.nll(
                    theta=theta,
                    gold_segments=y_segs,
                    dur_logprob=dur_logprob_eval,
                    endpoint_mask=endpoint_mask_eval,
                ) / max(theta.shape[-1], 1)
            kl  = kl_gaussian_standard(mu, logvar, reduction="mean")
            loss = nll + beta_weight * kl
            loss = loss + dec.regularization_loss()
            total_va += float(loss.item())
            nll_va += float(nll.item())
            kl_va += float(kl.item())
            n_batches_val += 1
    return total_va, nll_va, kl_va, n_batches_val

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
) -> None:
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
    loss_cfg  = LossBalanceConfig()
    pair_cfg = PairwiseConfig()

    # --- data & priors (24h clock prior; durations by allocation) ---
    acts = pd.read_csv(activities_csv)
    purp = pd.read_csv(purposes_csv)
    purposes = purp["purpose"].tolist()
    idx2purpose = purposes[:]

    alpha_init_per_purpose = {
        "Home": 1.6,
        "Work": 1.1,
        "Education": 1.1,
        "Shopping": 0.9,
        "Social": 0.8,
        "Accompanying": 0.9,
        "Other": 0.7,
    }
    alpha_l2_strength = 1e-3

    priors = derive_priors_from_activities(
        acts,
        purp,
        T_alloc_minutes=time_cfg.ALLOCATION_HORIZON_MINS,
        K_clock_prior=basis_cfg.K_clock_prior,
        T_clock_minutes=time_cfg.T_clock_minutes,
    )
    assert len(priors) == len(purposes)

    phi_t = build_phi_features(purposes, priors, device)

    # --- PDS (embeddings + clock prior accessors) ---
    pds = PurposeDistributionSpace(phi_t, d_p=pep_cfg.d_p, hidden=pep_cfg.hidden).to(device)
    pds.set_clock_prior_K(basis_cfg.K_clock_prior)

    # --- allocation grids, duration tables (if semi), and endpoint masks ---
    alloc = build_allocation_and_endpoint(priors, purp, purposes, pds, time_cfg, crf_mode, crf_cfg, device)
    loglam_grids = alloc["loglam_grids"]
    L_train = alloc["L_train"]
    L_eval = alloc["L_eval"]
    t_alloc01_train = alloc["t_alloc01_train"]
    t_alloc01_eval = alloc["t_alloc01_eval"]
    dur_logprob_train = alloc["dur_logprob_train"]
    dur_logprob_eval = alloc["dur_logprob_eval"]
    endpoint_mask_train = alloc["endpoint_mask_train"]
    endpoint_mask_eval = alloc["endpoint_mask_eval"]

    # --- models ---
    P = len(purposes)

    dec, enc, crf, pairwise_mod, pairwise_eval, params, opt = build_models_and_optimizer(
        pds,
        P,
        idx2purpose,
        alpha_init_per_purpose,
        alpha_l2_strength,
        time_cfg,
        basis_cfg,
        pep_cfg,
        dec_cfg,
        vae_cfg,
        crf_cfg,
        pair_cfg,
        device,
        t_alloc01_eval,
        lr,
        crf_mode,
    )

    # --- dataset / loaders ---
    ds = ScheduleDataset(activities_csv, T_alloc_minutes=time_cfg.ALLOCATION_HORIZON_MINS)
    tr_idx, va_idx = split_indices(len(ds), val_ratio=val_ratio, seed=42)
    tr_subset = torch.utils.data.Subset(ds, tr_idx)
    va_subset = torch.utils.data.Subset(ds, va_idx)

    purpose_to_idx = {p: i for i, p in enumerate(purposes)}
    loss_w_vec = torch.tensor(
        [float(loss_cfg.loss_weights_per_purpose.get(p, 1.0)) for p in purposes],
        dtype=torch.float32, device=device
    )  # [P]

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
    best_check_loss = float("inf")

    def beta_at_epoch(ep: int) -> float:
        if vae_cfg.kl_anneal_end and ep < vae_cfg.kl_anneal_end:
            # optional linear warmup; start at beta * (ep / end)
            return vae_cfg.beta * (ep / max(1, vae_cfg.kl_anneal_end))
        return vae_cfg.beta

    for epoch in range(1, epochs + 1):
        beta_weight = beta_at_epoch(epoch)

        total_tr, nll_tr, kl_tr, n_batches_train = train_one_epoch(
            dl_tr, pds, enc, dec, crf, pairwise_mod, loglam_grids, endpoint_mask_train,
            loss_w_vec, time_cfg, L_train, crf_mode, device, beta_weight, opt, params,
            t_alloc01_train, dur_logprob_train, crf_cfg.learn_eta,
        )

        total_va, nll_va, kl_va, n_batches_val = validate_one_epoch(
            dl_va, pds, enc, dec, crf, pairwise_eval, loglam_grids, endpoint_mask_eval,
            time_cfg, L_eval, crf_mode, device, beta_weight, dur_logprob_eval,
        )

        avg_tr = total_tr / max(n_batches_train, 1)
        avg_va = total_va / max(n_batches_val, 1)
        if epoch % 10 == 0 or epoch == epochs:
            click.echo(f"[{epoch:03d}] train={avg_tr:.4f}  val={avg_va:.4f}  (nll={nll_va/max(n_batches_val,1):.4f}, kl={kl_va/max(n_batches_val,1):.4f}, beta={beta_weight:.3f})")
            with torch.no_grad():
                a = dec.alpha.detach().cpu().tolist()
                alpha_log = {p: round(a[i], 3) for i, p in enumerate(idx2purpose)}
            click.echo(f"alpha per-purpose: {alpha_log}")


        history["epoch"].append(epoch)
        history["train_total"].append(avg_tr)
        history["val_total"].append(avg_va)
        history["nll"].append(nll_va / max(n_batches_val, 1))
        history["kl"].append(kl_va / max(n_batches_val, 1))

        # --- checkpoint ---
        ckpt = build_checkpoint(
            pds, enc, dec, crf, pairwise_mod, priors, purposes,
            time_cfg, basis_cfg, pep_cfg, dec_cfg, vae_cfg, crf_cfg, pair_cfg, epoch
        )

        check_loss = (avg_tr + avg_va * 2) / 3  # weighted toward val
        if check_loss < best_check_loss:
            best_check_loss = check_loss
            torch.save(ckpt, outdir / "ckpt_best.pt")
            click.echo(f"Best check so far: {best_check_loss:.4f} at epoch {epoch}")

        if epoch == epochs:
            torch.save(ckpt, outdir / "ckpt_final.pt")
            click.echo(f"Final val: {avg_va:.4f} (best {best_check_loss:.4f})")

    # Save history to CSV
    pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)
