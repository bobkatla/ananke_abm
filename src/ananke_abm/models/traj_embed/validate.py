import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import torch.nn.functional as F
from ananke_abm.models.traj_embed.configs import TimeConfig, BasisConfig, QuadratureConfig, PurposeEmbeddingConfig, DecoderConfig
from ananke_abm.models.traj_embed.model.purpose_space import PurposeDistributionSpace
from ananke_abm.models.traj_embed.model.decoder_timefield import TimeFieldDecoder
from ananke_abm.models.traj_embed.model.utils_bases import gauss_legendre_nodes, build_time_binned_transition_costs, viterbi_timecost_decode
from ananke_abm.models.traj_embed.train import ScheduleDataset, TrajEncoderGRU, build_masks, collate_fn
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader


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
            # Emit a single 'Home' zero-duration row if truly empty (rare)
            rows.append({"persid": persid, "stopno": 1, "purpose": "Home",
                         "startime": 0, "total_duration": int(T_minutes)})
            continue

        # Use continuous durations, renormalize to exactly T and build cumulative starts.
        d = np.array([max(0.0, float(dur)) for (_, _, dur) in segs], dtype=np.float64)
        if d.sum() <= 0:
            d = np.ones_like(d) / len(d)
        d = d / d.sum()

        dur_m = np.rint(d * T_minutes).astype(int)
        delta = int(T_minutes - dur_m.sum())
        dur_m[-1] += delta  # ensure exact total

        start_m = np.concatenate([[0], np.cumsum(dur_m[:-1])])

        for stopno, ((p_idx, _t0, _d), st, du) in enumerate(zip(segs, start_m, dur_m), start=1):
            if du <= 0:
                continue  # drop pathological zero/neg durations after rounding
            rows.append({
                "persid": persid,
                "stopno": stopno,
                "purpose": purposes[p_idx],
                "startime": int(st),
                "total_duration": int(du),
            })

    return pd.DataFrame(rows, columns=["persid", "stopno", "purpose", "startime", "total_duration"])


def build_truth_sets(activities_csv: str):
    """
    From the seed activities CSV, compute:
      - full_seqs: set of tuples of full purpose sequences per persid
      - bigrams:   set of all consecutive (a_i, a_{i+1}) pairs observed anywhere
    """
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
    """
    Produce one row per generated persid with the requested checks:
      1) start at home
      2) end at home
      3) full sequence exists in seed
      4) all adjacent pairs exist in seed
      5) confidence bucket: NO / OK / MODERATE / MAYBE
    """
    out = []
    for pid, g in gen_df.sort_values(["persid", "stopno"]).groupby("persid"):
        seq = g["purpose"].tolist()

        start_home = (len(seq) > 0 and seq[0] == home_label)
        end_home   = (len(seq) > 0 and seq[-1] == home_label)

        seq_exists = tuple(seq) in full_seqs

        if len(seq) <= 1:
            all_pairs_ok = True  # vacuously true
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

def _spd_cov(Z: torch.Tensor, shrink: float = 0.1, eps: float = 1e-5):
    N, m = Z.shape
    mu = Z.mean(0, keepdim=True) if N > 0 else torch.zeros(1, m)
    X = Z - mu
    S = (X.T @ X) / max(N-1, 1) if N > 1 else torch.zeros(m, m)
    # shrinkage toward spherical
    trace = torch.trace(S)
    avg_var = (trace / m) if m > 0 else torch.tensor(1.0)
    S = (1.0 - shrink) * S + shrink * avg_var * torch.eye(m)
    try:
        evals, evecs = torch.linalg.eigh(S)
        evals = torch.clamp(evals, min=eps)
        S_pd = (evecs * evals) @ evecs.T
    except Exception:
        var = torch.clamp(torch.var(Z, dim=0, unbiased=True) if N > 1 else torch.ones(m), min=eps)
        S_pd = torch.diag(var)
    return S_pd, mu.squeeze(0)

def sample_and_decode(enc, pds, dec, ds, purposes, masks_t, device,
                      num_samples: int = 5, L: int = 360, seed: int = 0, shrink: float = 0.1, C_t=None):
    torch.manual_seed(seed)
    purpose_to_idx = {p:i for i,p in enumerate(purposes)}

    # --- embed whole dataset -> collect R (unnormalized latents) ---
    def _collate(batch): return collate_fn(batch, purpose_to_idx)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=_collate)

    R = []
    with torch.no_grad():
        e_p = pds()
        for p_pad, t_pad, d_pad, lengths, *_ in dl:
            p_pad = p_pad.to(device); t_pad = t_pad.to(device); d_pad = d_pad.to(device)
            z, r = enc(p_pad, t_pad, d_pad, lengths, e_p)
            R.append(r.cpu())
    if not R:
        return []
    R = torch.cat(R, dim=0)  # (N, d_r) on CPU

    # --- fit Gaussian on R ---
    Sigma, mu = _spd_cov(R, shrink=shrink)
    dist = MultivariateNormal(loc=mu, covariance_matrix=Sigma)
    r_samp = dist.sample((num_samples,))  # CPU, shape (S, d_r)

    # --- map r* to z* correctly before decoding ---
    # dec.m is the expected latent dim; enc.proj maps pre-proj features -> latent
    need_projection = R.shape[1] != dec.m
    if need_projection:
        if not hasattr(enc, "proj"):
            raise RuntimeError(f"Sampled r has dim {R.shape[1]} but decoder expects {dec.m}, and enc.proj is missing.")
        z_samp = F.normalize(enc.proj(r_samp.to(device)), p=2, dim=-1)
    else:
        z_samp = F.normalize(r_samp.to(device), p=2, dim=-1)

    # --- decode deterministically on a dense grid ---
    t_dense = torch.linspace(0, 1, L, device=device)
    with torch.no_grad():
        e_p = pds()
        loglam_dense = pds.lambda_log(t_dense)
        u_dense = dec.utilities(z_samp, e_p, t_dense, loglam_dense, masks=masks_t)
        # decoded = dec.argmax_decode(u_dense, t_dense)
        decoded = viterbi_timecost_decode(u_dense, t_dense, C_t, switch_cost=0.02)
    return decoded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--activities_csv", type=str, default="/mnt/data/small_activities_homebound_wd.csv")
    ap.add_argument("--purposes_csv", type=str, default="/mnt/data/purposes.csv")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_gen", type=int, default=20, help="number of samples to generate from p(r)")
    ap.add_argument("--gen_prefix", type=str, default="gen", help="prefix for generated persid")
    ap.add_argument("--gen_csv", type=str, default=None, help="optional path to save generated activities CSV")
    ap.add_argument("--val_csv", type=str, default=None, help="optional path to save per-trajectory validation CSV")
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

    # PDS features / model init
    rows = []
    for p in purposes:
        pr = priors[p]
        rows.append(np.concatenate([pr.time_fourier, [pr.mu_t, pr.sigma_t, pr.mu_d, pr.sigma_d]]).astype("float32"))
    phi = np.stack(rows, axis=0); phi = (phi - phi.mean(0, keepdims=True)) / (phi.std(0, keepdims=True) + 1e-6)
    phi_t = torch.tensor(phi, dtype=torch.float32, device=args.device)

    pds = PurposeDistributionSpace(phi_t, d_p=pep_cfg.d_p, hidden=pep_cfg.hidden).to(args.device)
    pds.set_time_prior_K(basis_cfg.K_time_prior)
    e_p = pds()

    dec = TimeFieldDecoder(P=len(purposes), m_latent=dec_cfg.m_latent, d_p=pep_cfg.d_p,
                           K_decoder_time=basis_cfg.K_decoder_time, alpha_prior=dec_cfg.alpha_prior).to(args.device)
    enc = TrajEncoderGRU(d_p=pep_cfg.d_p, K_time_token=4, K_dur_token=4, m_latent=dec_cfg.m_latent).to(args.device)

    pds.load_state_dict(ck["model_state"]["pds"])
    enc.load_state_dict(ck["model_state"]["enc"])
    dec.load_state_dict(ck["model_state"]["dec"])
    pds.eval(); enc.eval(); dec.eval()

    purp_df = pd.read_csv(args.purposes_csv)
    ds = ScheduleDataset(args.activities_csv, time_cfg.T_minutes)  # reads input schema with persid, purpose, startime, total_duration
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                     collate_fn=lambda batch: collate_fn(batch, purpose_to_idx))

    # Quadrature for validation losses
    t_q, w_q = gauss_legendre_nodes(quad_cfg.Q_nodes_val, dtype=torch.float32, device=args.device)
    loglam = pds.lambda_log(t_q)

    masks = build_masks(purp_df, purposes)  # includes home_idx for "Home" if present
    masks_t = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in masks.items()}

    C_t = build_time_binned_transition_costs(args.activities_csv, purposes, nbins=24, device=args.device)

    # --------- Validation metrics (unchanged) ----------
    totals = []; ces=[]; emds=[]; tvs=[]; last_z=None
    with torch.no_grad():
        e_p = pds()
        for p_pad, t_pad, d_pad, lengths in dl:  # collate: p_pad,t_pad,d_pad,lengths
            p_pad = p_pad.to(args.device); t_pad = t_pad.to(args.device); d_pad = d_pad.to(args.device)
            z, r = enc(p_pad, t_pad, d_pad, lengths, e_p)
            # last_z = z
            u = dec.utilities(z, e_p, t_q, loglam, masks=masks_t)
            q = dec.soft_assign(u)

            # ground-truth rasterization from padded tensors (vectorized)
            from ananke_abm.models.traj_embed.model.rasterize import rasterize_from_padded
            y = rasterize_from_padded(p_pad, t_pad, d_pad, lengths, len(purposes), t_q).to(args.device)

            ce = dec.ce_loss(q, y, w_q)
            emd = dec.emd1d_loss(q, y, w_q)
            tv = dec.tv_loss(q, w_q)
            total = (dec_cfg.ce_weight*ce +
                     dec_cfg.emd_weight*emd +
                     dec_cfg.tv_weight*tv +
                     dec_cfg.durlen_weight*dec.durlen_loss(
                        q,
                        torch.tensor([priors[p].mu_d for p in purposes], dtype=torch.float32, device=args.device),
                        torch.tensor([priors[p].sigma_d for p in purposes], dtype=torch.float32, device=args.device),
                        w_q))
            totals.append(float(total.item())); ces.append(float(ce.item()))
            emds.append(float(emd.item())); tvs.append(float(tv.item()))

    print(f"Validation: total={np.mean(totals):.4f}  CE={np.mean(ces):.4f}  EMD={np.mean(emds):.4f}  TV={np.mean(tvs):.4f}")

    # --------- Generation -> CSV ----------
    Tm = time_cfg.T_minutes
    L = Tm // 5  # dense grid for decoding preview/generation
    # t_dense = torch.linspace(0, 1, L, device=args.device)
    # loglam_dense = pds.lambda_log(t_dense)

    with torch.no_grad():
        # Generate from latent Gaussian p(r) -> project -> normalize -> decode (your routine)
        decoded_gen = sample_and_decode(enc, pds, dec, ds, purposes, masks_t,
                                        args.device, num_samples=args.num_gen, L=L, seed=0, C_t=C_t)
    gen_df = decoded_to_activities_df(decoded_gen, purposes, Tm, start_persid=0, prefix=args.gen_prefix)

    # Optional save
    if args.gen_csv:
        Path(args.gen_csv).parent.mkdir(parents=True, exist_ok=True)
        gen_df.to_csv(args.gen_csv, index=False)
        print(f"Wrote generated activities to: {args.gen_csv}")

    # --------- Per-trajectory validation -> CSV ----------
    full_seqs, bigrams = build_truth_sets(args.activities_csv)
    home_label = purposes[masks["home_idx"]] if masks.get("home_idx", None) is not None else "Home"
    val_df = validate_sequences(gen_df, full_seqs, bigrams, home_label=home_label)

    if args.val_csv:
        Path(args.val_csv).parent.mkdir(parents=True, exist_ok=True)
        val_df.to_csv(args.val_csv, index=False)
        print(f"Wrote per-trajectory validation to: {args.val_csv}")

    # Optionally print a small sample for quick sanity check
    print(val_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
