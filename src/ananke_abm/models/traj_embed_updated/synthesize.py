from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import click

from ananke_abm.models.traj_embed_updated.configs import DecoderConfig
from ananke_abm.models.traj_embed_updated.model.purpose_space import PurposeDistributionSpace
from ananke_abm.models.traj_embed_updated.model.utils_bases import make_alloc_grid
from ananke_abm.models.traj_embed_updated.model.decoder_timefield import TimeFieldDecoder
from ananke_abm.models.traj_embed_updated.model.crf_linear import LinearChainCRF
from ananke_abm.models.traj_embed_updated.model.crf_semi import (
    SemiMarkovCRF, build_duration_logprob_table
)
from ananke_abm.models.traj_embed_updated.model.train_masks import build_endpoint_mask, endpoint_time_mask
from ananke_abm.models.traj_embed_updated.model.pairwise_time_bilinear import TimeVaryingPairwise


def set_seed(seed: int = 0):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def sanitize_theta(theta: torch.Tensor) -> torch.Tensor:
    theta_max = torch.max(theta, dim=1, keepdim=True).values
    theta_stable = theta - theta_max
    return torch.clamp(theta_stable, min=-30.0, max=30.0)


def decoded_to_activities_df(decoded, purposes, T_minutes: int,
                             start_persid: int = 0, prefix: str = "gen") -> pd.DataFrame:
    rows = []
    for s_idx, segs in enumerate(decoded):
        persid = f"{prefix}_{start_persid + s_idx:06d}"
        if not segs:
            rows.append({"persid": persid, "stopno": 1, "purpose": "Home",
                         "startime": 0, "total_duration": int(T_minutes)})
            continue

        d = np.array([max(0.0, float(dur)) for (_, _, dur) in segs], dtype=np.float64)
        if d.sum() <= 0:
            d = np.ones_like(d) / len(d)
        d = d / d.sum()

        dur_m = np.rint(d * T_minutes).astype(int)
        delta = int(T_minutes - dur_m.sum())
        dur_m[-1] += delta
        start_m = np.concatenate([[0], np.cumsum(dur_m[:-1])])

        for stopno, ((p_idx, _t0, _d), st, du) in enumerate(zip(segs, start_m, dur_m), start=1):
            if du <= 0:
                continue
            rows.append({
                "persid": persid,
                "stopno": stopno,
                "purpose": purposes[p_idx],
                "startime": int(st),
                "total_duration": int(du),
            })

    return pd.DataFrame(rows, columns=["persid", "stopno", "purpose", "startime", "total_duration"])


def synthesize(ckpt: str,
         purposes_csv: str,
         activities_csv: str,
         num_gen: int,
         batch_size: int,
         eval_step_minutes: int,
         crf_mode: str,
         gen_csv: str,
         seed: int,
         device: str):
    """
    Load checkpoint, sample synthetic schedules, decode with CRF, and write gen_csv.
    No evaluation/metrics are performed here.
    """
    click.echo(f"Synthesizing {num_gen} trajectories...")
    click.echo(f"Using device: {device}")
    click.echo(f"Using eval step minutes: {eval_step_minutes}")
    click.echo(f"Using CRF mode: {crf_mode}")
    set_seed(seed)

    device_t = torch.device(device)

    # --- load checkpoint & configs ---
    ck = torch.load(ckpt, map_location=device_t, weights_only=False)
    priors = ck["priors"]
    purposes: List[str] = ck["purposes"]

    cfg_basis = ck.get("configs", {}).get("basis", {})
    cfg_pep   = ck.get("configs", {}).get("pep", {})
    cfg_dec   = ck.get("configs", {}).get("dec", {})
    cfg_vae   = ck.get("configs", {}).get("vae", {"latent_dim": DecoderConfig.m_latent})
    cfg_crf   = ck.get("configs", {}).get("crf", {"eta": 4.0, "learn_eta": False})
    cfg_pair = ck.get("configs", {}).get("pairwise", {"enabled": False, "rank": 2, "K_clock": 6, "scale": 1.0})

    time_cfg_from_ckpt = ck.get("configs", {}).get("time", {})
    time_cfg = {
        "ALLOCATION_HORIZON_MINS": int(time_cfg_from_ckpt.get("ALLOCATION_HORIZON_MINS", 1800)),
        "TRAIN_GRID_MINS": int(time_cfg_from_ckpt.get("TRAIN_GRID_MINS", 10)),
        "VALID_GRID_MINS": int(eval_step_minutes),
        "T_clock_minutes": int(time_cfg_from_ckpt.get("T_clock_minutes", 1440)),
    }

    T_alloc_minutes = time_cfg["ALLOCATION_HORIZON_MINS"]
    T_clock_minutes = time_cfg["T_clock_minutes"]
    K_clock_prior   = int(cfg_basis.get("K_clock_prior", 6))
    K_decoder_time  = int(cfg_basis.get("K_decoder_time", 8))
    d_p             = int(cfg_pep.get("d_p", 16))
    dec_alpha_prior = float(cfg_dec.get("alpha_prior", 1.0))
    latent_dim      = int(cfg_vae.get("latent_dim", 16))
    crf_eta         = float(cfg_crf.get("eta", 4.0))
    crf_learn_eta   = bool(cfg_crf.get("learn_eta", False))

    # --- PDS features / model init ---
    rows = []
    for p in purposes:
        pr = priors[p]
        is_prim = float(getattr(pr, "is_primary_ooh", 0.0))
        rows.append(np.concatenate([pr.time_fourier, [pr.mu_t, pr.sigma_t, pr.mu_d, pr.sigma_d, is_prim]]).astype("float32"))
    phi = np.stack(rows, axis=0)
    phi = (phi - phi.mean(0, keepdims=True)) / (phi.std(0, keepdims=True) + 1e-6)
    phi_t = torch.tensor(phi, dtype=torch.float32, device=device_t)

    pds = PurposeDistributionSpace(phi_t, d_p=d_p, hidden=64).to(device_t)
    pds.set_clock_prior_K(K_clock_prior)

    P = len(purposes)
    dec = TimeFieldDecoder(
        P=P, m_latent=latent_dim, d_p=d_p,
        K_decoder_time=K_decoder_time, alpha_prior=dec_alpha_prior,
        time_cfg=time_cfg
    ).to(device_t)

    # Load states, ensuring Phi_* are purged to rebuild on new eval grid
    pds.load_state_dict(ck["model_state"]["pds"], strict=True)
    decoder_state_dict = ck["model_state"]["dec"].copy()
    for key in list(decoder_state_dict.keys()):
        if key.startswith("Phi_"):
            del decoder_state_dict[key]
    dec.load_state_dict(decoder_state_dict, strict=False)

    lin_crf  = LinearChainCRF(P=P, eta=crf_eta, learn_eta=crf_learn_eta).to(device_t)
    semi_crf = SemiMarkovCRF(P=P, eta=crf_eta, learn_eta=crf_learn_eta).to(device_t)
    if "crf" in ck["model_state"]:
        try:
            lin_crf.load_state_dict(ck["model_state"]["crf"], strict=False)
            semi_crf.load_state_dict(ck["model_state"]["crf"], strict=False)
        except Exception:
            pass

    pds.eval(); dec.eval(); lin_crf.eval(); semi_crf.eval()

    # --- Time-varying pairwise (if present/enabled) ---
    pairwise_mod = None
    pairwise_eval = None
    if bool(cfg_pair.get("enabled", False)) and crf_mode == "linear":
        pairwise_mod = TimeVaryingPairwise(
            P=P,
            rank=int(cfg_pair.get("rank", 2)),
            K_clock=int(cfg_pair.get("K_clock", 6)),
            scale=float(cfg_pair.get("scale", 1.0)),
        ).to(device_t)
        # load weights if saved
        pw_state = ck.get("model_state", {}).get("pairwise", None)
        if pw_state is not None:
            try:
                pairwise_mod.load_state_dict(pw_state, strict=True)
            except Exception:
                pass
        pairwise_mod.eval()

    # --- Grid and priors ---
    t_alloc_minutes_eval, t_alloc01_eval = make_alloc_grid(
        T_alloc_minutes=T_alloc_minutes,
        step_minutes=int(eval_step_minutes),
        device=device_t,
        dtype=torch.float32,
    )
    L_eval = t_alloc01_eval.numel()
    with torch.no_grad():
        loglam_eval = pds.lambda_log_on_alloc_grid(t_alloc_minutes_eval, T_clock_minutes=T_clock_minutes)  # [P,L]
    
    with torch.no_grad():
        pairwise_eval = pairwise_mod(t_alloc01_eval) if pairwise_mod is not None else None  # [L_eval,P,P] or None

    # --- Endpoint mask ---
    purp_df = pd.read_csv(purposes_csv)
    ep_masks = build_endpoint_mask(purp_df, purposes, can_open_col="can_open_day", can_close_col="can_close_day")
    endpoint_mask_eval = endpoint_time_mask(
        ep_masks.open_allowed, ep_masks.close_allowed, L_eval, step_mins=eval_step_minutes, device=device_t
    )  # [L,P]

    # --- Duration prior table (semi only) ---
    if crf_mode == "semi":
        dur_logprob = build_duration_logprob_table(
            priors=priors,
            purposes=purposes,
            step_minutes=int(eval_step_minutes),
            T_alloc_minutes=T_alloc_minutes,
            Dmax_minutes=int(300),  # default used in training config
            device=device_t,
        )  # [P, Dmax_bins]

    # --- Synthesis ---
    with torch.no_grad():
        total = int(num_gen)
        batch = int(batch_size)
        decoded_gen = []
        for start in range(0, total, batch):
            cur = min(batch, total - start)
            s = torch.randn(cur, latent_dim, device=device_t)
            z_samp = s / (s.norm(dim=-1, keepdim=True) + 1e-8)

            e_p = pds()
            theta_gen = dec.utilities_on_grid(
                z_samp, e_p, loglam_eval,
                grid_type="eval",
                endpoint_mask=None
            )
            theta_gen = sanitize_theta(theta_gen)

            if crf_mode == "linear":
                y_hat_gen = lin_crf.viterbi(
                    theta_gen, endpoint_mask=endpoint_mask_eval, pairwise_logits=pairwise_eval
                )  # [cur,L]
                # Convert labels to segments using normalized grid positions as in validate
                t = t_alloc01_eval.detach().cpu().numpy()
                for b in range(y_hat_gen.shape[0]):
                    yb = y_hat_gen[b].detach().cpu().numpy()
                    if L_eval == 0:
                        decoded_gen.append([])
                        continue
                    segs = []
                    start_i = 0
                    for i in range(1, L_eval):
                        if yb[i] != yb[i - 1]:
                            p = int(yb[i - 1])
                            t0 = float(t[start_i])
                            t1 = float(t[i])
                            segs.append((p, t0, max(t1 - t0, 0.0)))
                            start_i = i
                    p = int(yb[-1])
                    t0 = float(t[start_i])
                    t1 = float(t[-1])
                    segs.append((p, t0, max(t1 - t0, 0.0)))
                    decoded_gen.append(segs)
            else:
                y_hat_segments = semi_crf.viterbi(
                    theta=theta_gen,
                    dur_logprob=dur_logprob,
                    endpoint_mask=endpoint_mask_eval,
                    Dmax_bins=dur_logprob.shape[1],
                )
                for segs in y_hat_segments:
                    out = []
                    for (p, s_bin, d_bins) in segs:
                        t0 = float(t_alloc01_eval[s_bin].item())
                        t1 = float(t_alloc01_eval[min(s_bin + d_bins, L_eval - 1)].item())
                        out.append((p, t0, max(t1 - t0, 0.0)))
                    decoded_gen.append(out)

    Tm = T_alloc_minutes
    gen_df = decoded_to_activities_df(decoded_gen, purposes, Tm, start_persid=0, prefix="gen")
    Path(gen_csv).parent.mkdir(parents=True, exist_ok=True)
    gen_df.to_csv(gen_csv, index=False)
    click.echo(f"Wrote generated activities to: {gen_csv}")
