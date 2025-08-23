import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from ananke_abm.models.traj_embed.configs import TimeConfig, BasisConfig, QuadratureConfig, PurposeEmbeddingConfig, DecoderConfig
from ananke_abm.models.traj_embed.model.purpose_space import PurposeDistributionSpace
from ananke_abm.models.traj_embed.model.decoder_timefield import TimeFieldDecoder
from ananke_abm.models.traj_embed.model.utils_bases import gauss_legendre_nodes
from ananke_abm.models.traj_embed.train import ScheduleDataset, TrajEncoderGRU, build_masks, collate_fn
from ananke_abm.models.traj_embed.model.rasterize import rasterize_batch

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
    L = 360
    t_dense = torch.linspace(0, 1, L, device=args.device)
    loglam_dense = pds.lambda_log(t_dense)
    with torch.no_grad():
        u_dense = dec.utilities(last_z, e_p, t_dense, loglam_dense, masks=masks_t)  # (B,P,L)
        decoded = dec.argmax_decode(u_dense, t_dense)
        Tm = time_cfg.T_minutes
        def seg_to_str(seg):
            p_idx, t0, d = seg
            return f"{purposes[p_idx]} @ {int(t0*Tm)}m for {int(d*Tm)}m"
        for b, segs in enumerate(decoded[:3]):
            print(f"Decoded sample {b}:")
            print(" | ".join(seg_to_str(s) for s in segs))

if __name__ == "__main__":
    main()
