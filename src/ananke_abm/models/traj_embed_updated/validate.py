from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

# --- configs & model pieces ---
from ananke_abm.models.traj_embed_updated.configs import DecoderConfig
from ananke_abm.models.traj_embed_updated.model.purpose_space import PurposeDistributionSpace
from ananke_abm.models.traj_embed_updated.model.utils_bases import make_alloc_grid
from ananke_abm.models.traj_embed_updated.model.rasterize import rasterize_from_padded_to_grid
from ananke_abm.models.traj_embed_updated.model.decoder_timefield import TimeFieldDecoder
from ananke_abm.models.traj_embed_updated.model.encoder import TrajEncoderGRU
from ananke_abm.models.traj_embed_updated.model.crf_linear import LinearChainCRF
from ananke_abm.models.traj_embed_updated.model.train_masks import build_endpoint_mask, endpoint_time_mask


# -------------------
# Small utilities
# -------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def collate_fn(batch, purpose_to_idx):
    p_lists, t_lists, d_lists, lens = [], [], [], []
    for seq in batch:
        p_idx = [purpose_to_idx[p] for p, _, _ in seq]
        t0 = [t for _, t, _ in seq]
        dd = [d for _, _, d in seq]
        p_lists.append(torch.tensor(p_idx, dtype=torch.long))
        t_lists.append(torch.tensor(t0, dtype=torch.float32))
        d_lists.append(torch.tensor(dd, dtype=torch.float32))
        lens.append(len(seq))
    p_pad = torch.nn.utils.rnn.pad_sequence(p_lists, batch_first=True, padding_value=0)
    t_pad = torch.nn.utils.rnn.pad_sequence(t_lists, batch_first=True, padding_value=0.0)
    d_pad = torch.nn.utils.rnn.pad_sequence(d_lists, batch_first=True, padding_value=0.0)
    return p_pad, t_pad, d_pad, lens


def labels_to_segments(y_hat: torch.Tensor, t_alloc01: torch.Tensor) -> List[List[Tuple[int, float, float]]]:
    """
    y_hat: [B, L] long labels
    t_alloc01: [L] normalized grid positions in [0,1]
    Returns: list over batch; each item is list[(p_idx, t0_norm, d_norm)]
    """
    B, L = y_hat.shape[0], y_hat.shape[1]
    t = t_alloc01.detach().cpu().numpy()
    out = []
    for b in range(B):
        yb = y_hat[b].detach().cpu().numpy()
        if L == 0:
            out.append([])
            continue
        segs = []
        start = 0
        for i in range(1, L):
            if yb[i] != yb[i - 1]:
                p = int(yb[i - 1])
                t0 = float(t[start])
                t1 = float(t[i])
                segs.append((p, t0, max(t1 - t0, 0.0)))
                start = i
        p = int(yb[-1])
        t0 = float(t[start])
        t1 = float(t[-1])
        segs.append((p, t0, max(t1 - t0, 0.0)))
        out.append(segs)
    return out


def decoded_to_activities_df(decoded, purposes, T_minutes: int,
                             start_persid: int = 0, prefix: str = "gen") -> pd.DataFrame:
    """
    Turn a list[List[(p_idx, t0, d)]] into a long DataFrame with columns:
    persid, stopno, purpose, startime, total_duration.
    Durations sum to T_minutes per persid; start times are cumulative.
    """
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


def build_truth_sets(activities_csv: str):
    acts = pd.read_csv(activities_csv)
    acts = acts.sort_values(["persid", "startime", "stopno"])
    seqs = acts.groupby("persid")["purpose"].apply(list)

    full_seqs = set(tuple(s) for s in seqs.values)

    bigrams = set()
    for s in seqs.values:
        if len(s) >= 2:
            bigrams.update(zip(s[:-1], s[1:]))
    return full_seqs, bigrams


def validate_sequences(gen_df: pd.DataFrame,
                       full_seqs: set,
                       bigrams: set,
                       home_label: str = "Home") -> pd.DataFrame:
    out = []
    for pid, g in gen_df.sort_values(["persid", "stopno"]).groupby("persid"):
        seq = g["purpose"].tolist()
        start_home = (len(seq) > 0 and seq[0] == home_label)
        end_home   = (len(seq) > 0 and seq[-1] == home_label)
        seq_exists = tuple(seq) in full_seqs
        if len(seq) <= 1:
            all_pairs_ok = True
        else:
            pairs = list(zip(seq[:-1], seq[1:]))
            all_pairs_ok = all(pair in bigrams for pair in pairs)

        if not (start_home and end_home):
            confidence = "NO"
        elif seq_exists:
            confidence = "OK"
        elif all_pairs_ok:
            confidence = "MODERATE"
        else:
            confidence = "MAYBE"

        out.append({
            "persid": pid,
            "n_stops": len(seq),
            "start_is_home": start_home,
            "end_is_home": end_home,
            "sequence_exists_in_seed": seq_exists,
            "all_adjacent_pairs_in_seed": all_pairs_ok,
            "confidence": confidence,
            "sequence": " > ".join(seq),
        })

    cols = ["persid","n_stops","start_is_home","end_is_home",
            "sequence_exists_in_seed","all_adjacent_pairs_in_seed","confidence","sequence"]
    return pd.DataFrame(out, columns=cols)


# -------------------
# Validation script
# -------------------
def gen_n_val_traj(
    ckpt: str,
    activities_csv: str,
    purposes_csv: str,
    batch_size: int,
    num_gen: int,
    gen_prefix: str,
    gen_csv: str,
    val_csv: str,
    eval_step_minutes: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Validating {num_gen} trajectories...")
    print(f"Using device: {device}")
    print(f"Using eval step minutes: {eval_step_minutes}")
    print(f"Using batch size: {batch_size}")
    print(f"Output gen csv: {gen_csv}")
    print(f"Output val csv: {val_csv}")
    set_seed(42)

    device = torch.device(device)

    # --- load checkpoint & configs (with robust fallbacks) ---
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    priors = ck["priors"]
    purposes = ck["purposes"]
    purpose_to_idx = ck["purpose_to_idx"]

    # Fallbacks in case older ckpts don't have new sections
    cfg_basis = ck.get("configs", {}).get("basis", {})
    cfg_pep = ck.get("configs", {}).get("pep", {})
    cfg_dec = ck.get("configs", {}).get("dec", {})
    cfg_vae = ck.get("configs", {}).get("vae", {"latent_dim": DecoderConfig.m_latent})
    cfg_crf = ck.get("configs", {}).get("crf", {"eta": 4.0, "learn_eta": False})

    # Create a time_cfg dict for decoder init, respecting CLI override for eval grid
    time_cfg_from_ckpt = ck.get("configs", {}).get("time", {})
    time_cfg = {
        "ALLOCATION_HORIZON_MINS": int(time_cfg_from_ckpt.get("ALLOCATION_HORIZON_MINS", 1800)),
        "TRAIN_GRID_MINS": int(time_cfg_from_ckpt.get("TRAIN_GRID_MINS", 10)),
        "VALID_GRID_MINS": eval_step_minutes,
    }

    T_alloc_minutes = time_cfg["ALLOCATION_HORIZON_MINS"]
    T_clock_minutes = int(time_cfg_from_ckpt.get("T_clock_minutes", 1440))
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
        rows.append(np.concatenate([pr.time_fourier, [pr.mu_t, pr.sigma_t, pr.mu_d, pr.sigma_d, pr.is_primary_ooh]]).astype("float32"))
    phi = np.stack(rows, axis=0)
    phi = (phi - phi.mean(0, keepdims=True)) / (phi.std(0, keepdims=True) + 1e-6)
    phi_t = torch.tensor(phi, dtype=torch.float32, device=device)

    pds = PurposeDistributionSpace(phi_t, d_p=d_p, hidden=64).to(device)
    pds.set_clock_prior_K(K_clock_prior)

    # --- models ---
    P = len(purposes)
    dec = TimeFieldDecoder(
        P=P, m_latent=latent_dim, d_p=d_p,
        K_decoder_time=K_decoder_time, alpha_prior=dec_alpha_prior,
        time_cfg=time_cfg
    ).to(device)
    enc = TrajEncoderGRU(d_p=d_p, K_time_token_clock=4, K_time_token_alloc=0, K_dur_token=4,
                         m_latent=latent_dim, gru_hidden=64, num_layers=1).to(device)
    crf = LinearChainCRF(P=P, eta=crf_eta, learn_eta=crf_learn_eta).to(device)

    # Load states (be lenient with older ckpts)
    pds.load_state_dict(ck["model_state"]["pds"], strict=True)
    enc.load_state_dict(ck["model_state"]["enc"], strict=False)
    
    # Remove cached Phi buffers from the state dict to allow for different validation grid sizes
    decoder_state_dict = ck["model_state"]["dec"]
    for key in list(decoder_state_dict.keys()):
        if key.startswith("Phi_"):
            del decoder_state_dict[key]
    dec.load_state_dict(decoder_state_dict, strict=False)

    if "crf" in ck["model_state"]:
        crf.load_state_dict(ck["model_state"]["crf"], strict=False)

    pds.eval(); enc.eval(); dec.eval(); crf.eval()

    # --- grids & priors on grid ---
    t_alloc_minutes_eval, t_alloc01_eval = make_alloc_grid(
        T_alloc_minutes=T_alloc_minutes,
        step_minutes=int(eval_step_minutes),
        device=device,
        dtype=torch.float32,
    )
    L_eval = t_alloc01_eval.numel()
    with torch.no_grad():
        loglam_eval = pds.lambda_log_on_alloc_grid(t_alloc_minutes_eval, T_clock_minutes=T_clock_minutes)  # [P,L]

    # --- endpoint masks ---
    purp_df = pd.read_csv(purposes_csv)
    ep_masks = build_endpoint_mask(purp_df, purposes, can_open_col="can_open_day", can_close_col="can_close_day")
    endpoint_mask_eval = endpoint_time_mask(ep_masks.open_allowed, ep_masks.close_allowed, L_eval, step_mins=eval_step_minutes, device=device)  # [L,P]

    # --- data loader ---
    ds = ScheduleDataset(activities_csv, T_alloc_minutes=T_alloc_minutes)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, purpose_to_idx),
        num_workers=0, pin_memory=False
    )

    # --------- Validation: NLL on grid + a few μ-recon decodes ----------
    totals = []
    nlls = []
    decoded_preview = []
    with torch.no_grad():
        e_p = pds()
        for p_pad, t_pad, d_pad, lengths in dl:
            p_pad = p_pad.to(device); t_pad = t_pad.to(device); d_pad = d_pad.to(device)

            # encode with μ (deterministic at validation preview)
            z, s, mu, logvar = enc(
                p_pad, t_pad, d_pad, lengths, e_p,
                T_alloc_minutes=T_alloc_minutes, T_clock_minutes=T_clock_minutes,
                sample=False
            )

            theta = dec.utilities_on_grid(
                z, e_p, loglam_eval,
                grid_type="eval",
                endpoint_mask=endpoint_mask_eval
            )
            fallback_idx = 0 #ep_masks.open_allowed.argmax() if ep_masks.open_allowed.any() else 0
            y_grid = rasterize_from_padded_to_grid(p_pad, t_pad, d_pad, lengths, L=L_eval, fallback_idx=fallback_idx)

            # NLL on the grid
            nll = crf.nll(theta, y_grid, endpoint_mask=endpoint_mask_eval)
            nlls.append(float(nll.item()))
            totals.append(float(nll.item()))

            assert endpoint_mask_eval.shape == (theta.shape[-1], theta.shape[1])

            # Viterbi for deterministic decode preview (take first few only)
            y_hat = crf.viterbi(theta, endpoint_mask=endpoint_mask_eval)  # [B,L]
            tail_bins = int(round(60 / time_cfg["VALID_GRID_MINS"]))
            bad = ~endpoint_mask_eval[-tail_bins:, :].gather(1, y_hat[:, -tail_bins:].T).T  # [B, tail_bins]
            # print("Tail violations per batch:", bad.any(dim=1).float().mean().item())
            assert bad.any(dim=1).float().mean().item() == 0, "Tail violations found"
            decoded_preview.extend(labels_to_segments(y_hat, t_alloc01_eval))

    print(f"Validation (grid NLL): nll={np.mean(nlls):.4f}")

    # Show a few reconstructed samples (first 3)
    Tm = T_alloc_minutes
    # print("=== Reconstructions (μ → Viterbi) ===")
    # for b, segs in enumerate(decoded_preview[:3]):
    #     pretty = " | ".join(f"{purposes[p]} @ {int(t0*Tm)}m for {int(d*Tm)}m" for (p,t0,d) in segs)
    #     print(f"Rec {b}: {pretty}")

    # --------- Generation from prior ----------
    # Sample s ~ N(0, I), then z = normalize(s), decode with Viterbi.
    with torch.no_grad():
        S = int(num_gen)
        s = torch.randn(S, latent_dim, device=device)
        z_samp = s / (s.norm(dim=-1, keepdim=True) + 1e-8)

        e_p = pds()
        theta_gen = dec.utilities_on_grid(
            z_samp, e_p, loglam_eval,
            grid_type="eval",
            endpoint_mask=endpoint_mask_eval
        )
        y_hat_gen = crf.viterbi(theta_gen, endpoint_mask=endpoint_mask_eval)  # [S,L]
        decoded_gen = labels_to_segments(y_hat_gen, t_alloc01_eval)

    gen_df = decoded_to_activities_df(decoded_gen, purposes, Tm, start_persid=0, prefix=gen_prefix)
    if gen_csv:
        Path(gen_csv).parent.mkdir(parents=True, exist_ok=True)
        gen_df.to_csv(gen_csv, index=False)
        print(f"Wrote generated activities to: {gen_csv}")

    # --------- Per-trajectory validation on generated CSV ----------
    full_seqs, bigrams = build_truth_sets(activities_csv)
    home_label = "Home"
    val_df = validate_sequences(gen_df, full_seqs, bigrams, home_label=home_label)
    if val_csv:
        Path(val_csv).parent.mkdir(parents=True, exist_ok=True)
        val_df.to_csv(val_csv, index=False)
        print(f"Wrote per-trajectory validation to: {val_csv}")

    print("=== Generated sample sequences (first 10 check) ===")
    print(val_df.head(10).to_string(index=False))

    