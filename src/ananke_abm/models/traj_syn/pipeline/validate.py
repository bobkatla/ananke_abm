# ananke_abm/models/traj_embed_updated/validate.py
from pathlib import Path
from typing import List, Tuple, Optional
import os
import tempfile

import numpy as np
import pandas as pd
import torch
import click
import json

# --- configs & model pieces ---
from ananke_abm.models.traj_syn.configs import DecoderConfig
from ananke_abm.models.traj_syn.vae.purpose_space import PurposeDistributionSpace
from ananke_abm.models.traj_syn.core.utils_bases import make_alloc_grid
from ananke_abm.models.traj_syn.core.rasterize import rasterize_from_padded_to_grid
from ananke_abm.models.traj_syn.vae.decoder_timefield import TimeFieldDecoder
from ananke_abm.models.traj_syn.vae.encoder import TrajEncoderGRU
from ananke_abm.models.traj_syn.crf.crf_linear import LinearChainCRF
from ananke_abm.models.traj_syn.crf.crf_semi import (
    SemiMarkovCRF, build_duration_logprob_table
)
from ananke_abm.models.traj_syn.core.train_masks import build_endpoint_mask, endpoint_time_mask
from ananke_abm.models.traj_syn.eval.eval_utils import (
    activities_csv_to_segments,
    summarize as summarize_metrics,
)
from ananke_abm.models.traj_syn.pipeline.synthesize import synthesize, set_seed, sanitize_theta
from ananke_abm.models.traj_syn.eval.pairwise_time_bilinear import TimeVaryingPairwise


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
    B, L = y_hat.shape
    t = t_alloc01.detach().cpu().numpy()
    out: List[List[Tuple[int, float, float]]] = []
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    crf_mode: str = "linear",        # "linear" | "semi"
    semi_Dmax_minutes: int = 300,    # only used when crf_mode="semi"
    summary_json: Optional[str] = None,
    use_samples: Optional[str] = None,
):
    """
    Validate & generate using the trained model.
    - Uses the same emissions as training (decoder utilities + clock prior).
    - CRF constraints are applied only inside CRF (no double-masking).
    - crf_mode chooses Linear or Semi-Markov decoding & NLL.
    """
    click.echo(f"Validating {num_gen} trajectories...")
    click.echo(f"Using device: {device}")
    click.echo(f"Using eval step minutes: {eval_step_minutes}")
    click.echo(f"Using CRF mode: {crf_mode}")
    set_seed(42)

    device = torch.device(device)

    # If evaluating pre-generated samples, short-circuit heavy model setup
    if use_samples:
        click.echo("Using pre-generated samples for evaluation (skipping model decode).")
        gen_df = pd.read_csv(use_samples)

        # Per-trajectory validation on provided samples
        full_seqs, bigrams = build_truth_sets(activities_csv)
        home_label = "Home"
        val_df = validate_sequences(gen_df, full_seqs, bigrams, home_label=home_label)
        if val_csv:
            Path(val_csv).parent.mkdir(parents=True, exist_ok=True)
            val_df.to_csv(val_csv, index=False)
            click.echo(f"Wrote per-trajectory validation to: {val_csv}")

        # Summary JSON
        real_segments = activities_csv_to_segments(activities_csv)
        syn_segments = activities_csv_to_segments(use_samples)
        # Determine purposes from union of both sets
        purposes: List[str] = sorted({s["purpose"] for person in (real_segments + syn_segments) for s in person})
        summary = summarize_metrics(real_segments, syn_segments, purposes, step_minutes=int(eval_step_minutes))
        if summary_json:
            Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
            with open(summary_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            click.echo(f"Wrote summary JSON to: {summary_json}")
        return

    # --- load checkpoint & configs (with robust fallbacks) ---
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    priors = ck["priors"]
    purposes = ck["purposes"]
    purpose_to_idx = ck["purpose_to_idx"]

    # Fallbacks in case older ckpts don't have new sections
    cfg_basis = ck.get("configs", {}).get("basis", {})
    cfg_pep   = ck.get("configs", {}).get("pep", {})
    cfg_dec   = ck.get("configs", {}).get("dec", {})
    cfg_vae   = ck.get("configs", {}).get("vae", {"latent_dim": DecoderConfig.m_latent})
    cfg_crf   = ck.get("configs", {}).get("crf", {"eta": 4.0, "learn_eta": False})
    cfg_pair = ck.get("configs", {}).get("pairwise", {"enabled": False, "rank": 2, "K_clock": 6, "scale": 1.0})

    # Create a time_cfg dict for decoder init, respecting CLI override for eval grid
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
        # if your priors include is_primary_ooh, keep it; else default 0.0
        is_prim = float(getattr(pr, "is_primary_ooh", 0.0))
        rows.append(np.concatenate([pr.time_fourier, [pr.mu_t, pr.sigma_t, pr.mu_d, pr.sigma_d, is_prim]]).astype("float32"))
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
    cfgs = ck.get("configs", {})
    enc_arch = cfgs.get("enc_arch", {})

    # Fallbacks keep legacy ckpts working
    enc_kwargs = dict(
        d_p=enc_arch.get("d_p", d_p),
        K_time_token_clock=enc_arch.get("K_time_token_clock", 4),
        K_time_token_alloc=enc_arch.get("K_time_token_alloc", 0),
        K_dur_token=enc_arch.get("K_dur_token", 4),
        m_latent=enc_arch.get("m_latent", latent_dim),
        gru_hidden=enc_arch.get("gru_hidden", 64),
        num_layers=enc_arch.get("num_layers", 1),
        dropout=enc_arch.get("dropout", 0.0),
        bidirectional=enc_arch.get("bidirectional", False),
        use_token_resmlp=enc_arch.get("use_token_resmlp", False),
        token_resmlp_hidden=enc_arch.get("token_resmlp_hidden", 128),
        use_residual_gru=enc_arch.get("use_residual_gru", True),
        use_attn_pool=enc_arch.get("use_attn_pool", False),
    )
    enc = TrajEncoderGRU(**enc_kwargs).to(device)
    enc.load_state_dict(ck["model_state"]["enc"], strict=True)  # strict=True now works

    # Both CRFs available; we choose by crf_mode below
    lin_crf  = LinearChainCRF(P=P, eta=crf_eta, learn_eta=crf_learn_eta).to(device)
    semi_crf = SemiMarkovCRF(P=P, eta=crf_eta, learn_eta=crf_learn_eta).to(device)

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
        # same state dict for both; it may only hold eta for linear; semi ignores extras
        try:
            lin_crf.load_state_dict(ck["model_state"]["crf"], strict=False)
            semi_crf.load_state_dict(ck["model_state"]["crf"], strict=False)
        except Exception:
            pass

    pds.eval()
    enc.eval()
    dec.eval()
    lin_crf.eval()
    semi_crf.eval()

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

    # --- time-varying pairwise on eval grid (if present) ---
    pairwise_mod = None
    pairwise_eval = None
    if bool(cfg_pair.get("enabled", False)) and crf_mode == "linear":
        pairwise_mod = TimeVaryingPairwise(
            P=len(purposes),
            rank=int(cfg_pair.get("rank", 2)),
            K_clock=int(cfg_pair.get("K_clock", 6)),
            scale=float(cfg_pair.get("scale", 1.0)),
        ).to(device)
        # load weights if saved
        pw_state = ck.get("model_state", {}).get("pairwise", None)
        if pw_state is not None:
            try:
                pairwise_mod.load_state_dict(pw_state, strict=True)
            except Exception:
                pass
        pairwise_mod.eval()
        with torch.no_grad():
            pairwise_eval = pairwise_mod(t_alloc01_eval)  # [L_eval, P, P]

    # --- endpoint masks ---
    purp_df = pd.read_csv(purposes_csv)
    ep_masks = build_endpoint_mask(purp_df, purposes, can_open_col="can_open_day", can_close_col="can_close_day")
    endpoint_mask_eval = endpoint_time_mask(
        ep_masks.open_allowed, ep_masks.close_allowed, L_eval, step_mins=eval_step_minutes, device=device
    )  # [L,P]

    # --- duration prior table (semi only) ---
    if crf_mode == "semi":
        dur_logprob = build_duration_logprob_table(
            priors=priors,
            purposes=purposes,
            step_minutes=int(eval_step_minutes),
            T_alloc_minutes=T_alloc_minutes,
            Dmax_minutes=int(semi_Dmax_minutes),
            device=device,
        )  # [P, Dmax_bins]

    # --- data loader ---
    ds = ScheduleDataset(activities_csv, T_alloc_minutes=T_alloc_minutes)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, purpose_to_idx),
        num_workers=0, pin_memory=False
    )

    # --------- Validation: NLL on grid + a few μ-recon decodes ----------
    nlls = []
    decoded_preview = []
    with torch.no_grad():
        e_p = pds()
        for p_pad, t_pad, d_pad, lengths in dl:
            p_pad = p_pad.to(device)
            t_pad = t_pad.to(device)
            d_pad = d_pad.to(device)

            # encode with μ (deterministic at validation preview)
            z, s, mu, logvar = enc(
                p_pad, t_pad, d_pad, lengths, e_p,
                T_alloc_minutes=T_alloc_minutes, T_clock_minutes=T_clock_minutes,
                sample=False
            )

            # emissions on eval grid; CRF will handle constraints (no endpoint mask here)
            theta = dec.utilities_on_grid(
                z, e_p, loglam_eval,
                grid_type="eval",
                endpoint_mask=None
            )
            theta = sanitize_theta(theta)

            if crf_mode == "linear":
                # rasterize labels to grid for NLL
                fallback_idx = 0
                y_grid = rasterize_from_padded_to_grid(p_pad, t_pad, d_pad, lengths, L=L_eval, fallback_idx=fallback_idx)
                nll = (lin_crf.nll(theta, y_grid, endpoint_mask=endpoint_mask_eval, pairwise_logits=pairwise_eval)) / max(theta.shape[-1], 1)
                y_hat = lin_crf.viterbi(theta, endpoint_mask=endpoint_mask_eval, pairwise_logits=pairwise_eval)
                decoded_preview.extend(labels_to_segments(y_hat, t_alloc01_eval))
            else:
                # build gold segments for NLL under semi-CRF
                # (re-use rasterized labels to extract segments)
                fallback_idx = 0
                y_grid = rasterize_from_padded_to_grid(p_pad, t_pad, d_pad, lengths, L=L_eval, fallback_idx=fallback_idx)
                # convert dense labels -> segments (p, s, d) in bin indices
                segs_b: List[List[Tuple[int,int,int]]] = []
                for b in range(y_grid.shape[0]):
                    yb = y_grid[b].cpu().tolist()
                    if len(yb) == 0:
                        segs_b.append([])
                        continue
                    cur = yb[0]
                    s = 0
                    segs = []
                    for i in range(1, len(yb)):
                        if yb[i] != cur:
                            segs.append((int(cur), int(s), int(i - s)))
                            cur = yb[i]
                            s = i
                    segs.append((int(cur), int(s), int(len(yb) - s)))
                    segs_b.append(segs)

                nll = (semi_crf.nll(
                    theta=theta,
                    gold_segments=segs_b,
                    dur_logprob=dur_logprob,
                    endpoint_mask=endpoint_mask_eval,
                )) / max(theta.shape[-1], 1)
                # semi-CRF Viterbi for preview
                y_hat_segments = semi_crf.viterbi(
                    theta=theta,
                    dur_logprob=dur_logprob,
                    endpoint_mask=endpoint_mask_eval,
                    Dmax_bins=dur_logprob.shape[1],
                )
                # convert segments -> (p, t0_norm, d_norm)
                for segs in y_hat_segments:
                    out = []
                    for (p, s_bin, d_bins) in segs:
                        t0 = float(t_alloc01_eval[s_bin].item())
                        t1 = float(t_alloc01_eval[min(s_bin + d_bins, L_eval - 1)].item())
                        out.append((p, t0, max(t1 - t0, 0.0)))
                    decoded_preview.append(out)

            nlls.append(float(nll.item()))

    click.echo(f"Validation (grid NLL): nll={np.mean(nlls):.4f}")

    # --------- Synthesize (via helper) ----------
    # Use the shared synthesize() to generate samples. If no gen_csv provided, write to a temp file.
    tmp_path = None
    out_csv = gen_csv
    if out_csv is None or len(str(out_csv)) == 0:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp_path = tmp.name
        tmp.close()
        out_csv = tmp_path

    synthesize(
        ckpt=ckpt,
        purposes_csv=purposes_csv,
        activities_csv=activities_csv,
        num_gen=num_gen,
        batch_size=batch_size,
        eval_step_minutes=eval_step_minutes,
        crf_mode=crf_mode,
        gen_csv=out_csv,
        seed=42,
        device=str(device),
    )

    gen_df = pd.read_csv(out_csv)
    if tmp_path is not None:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # --------- Per-trajectory validation on generated CSV ----------
    full_seqs, bigrams = build_truth_sets(activities_csv)
    home_label = "Home"
    val_df = validate_sequences(gen_df, full_seqs, bigrams, home_label=home_label)
    if val_csv:
        Path(val_csv).parent.mkdir(parents=True, exist_ok=True)
        val_df.to_csv(val_csv, index=False)
        click.echo(f"Wrote per-trajectory validation to: {val_csv}")

    click.echo("=== Generated sample sequences (first 10 check) ===")
    click.echo(val_df.head(10).to_string(index=False))

    # --------- Summary JSON ----------
    real_segments = activities_csv_to_segments(activities_csv)
    # Convert in-memory gen_df into segments list
    syn_segments = []
    for _, g in gen_df.sort_values(["persid", "stopno"]).groupby("persid"):
        person = []
        for _i, r in g.iterrows():
            person.append({
                "purpose": str(r["purpose"]),
                "startime": int(r["startime"]),
                "total_duration": int(r["total_duration"]),
            })
        syn_segments.append(person)

    summary = summarize_metrics(real_segments, syn_segments, purposes, step_minutes=int(eval_step_minutes))
    if summary_json:
        Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        click.echo(f"Wrote summary JSON to: {summary_json}")
