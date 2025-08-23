import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from pathlib import Path

from ananke_abm.models.traj_embed.configs import TimeConfig, BasisConfig, QuadratureConfig, PurposeEmbeddingConfig, DecoderConfig
from ananke_abm.models.traj_embed.model.pds_loader import derive_priors_from_activities
from ananke_abm.models.traj_embed.model.purpose_space import PurposeDistributionSpace
from ananke_abm.models.traj_embed.model.utils_bases import gauss_legendre_nodes, fourier_time_features
from ananke_abm.models.traj_embed.model.rasterize import rasterize_batch
from ananke_abm.models.traj_embed.model.decoder_timefield import TimeFieldDecoder

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# helper
def build_y_cache(subset, purpose_to_idx, t_q, device):
    cache = []
    for seq in subset:   # seq is list[(p,t0,d), ...]
        y = rasterize_batch([seq], purpose_to_idx, t_q)[0]  # (P,Q)
        cache.append(y)
    return [torch.tensor(y, device=device) if not torch.is_tensor(y) else y.to(device) for y in cache]

class ScheduleDataset(torch.utils.data.Dataset):
    def __init__(self, activities_csv: str, T_minutes: int):
        acts = pd.read_csv(activities_csv)
        self.T = float(T_minutes)
        self.seqs = []
        for _, g in acts.groupby("persid"):
            g = g.sort_values(["startime","stopno"])
            day = [(str(r["purpose"]), float(r["startime"]/self.T), float(r["total_duration"]/self.T)) for _, r in g.iterrows()]
            self.seqs.append(day)
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx): return self.seqs[idx]

def split_indices(n, val_ratio=0.2, seed=42):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = int(round(n*val_ratio))
    return idx[n_val:], idx[:n_val]

class TrajEncoderGRU(nn.Module):
    def __init__(self, d_p:int, K_time_token:int=4, K_dur_token:int=4, m_latent:int=16):
        super().__init__()
        self.d_p = d_p
        self.Kt = K_time_token
        self.Kd = K_dur_token
        in_dim = d_p + (2*self.Kt+1) + (2*self.Kd+1)
        self.gru = nn.GRU(input_size=in_dim, hidden_size=64, num_layers=1, batch_first=True)
        self.proj = nn.Linear(64*2, m_latent)
    def forward(self, p_idx_pad, t_pad, d_pad, lengths, e_p):
        B, Lmax = p_idx_pad.shape
        ep_tok = e_p[p_idx_pad]  # (B,Lmax,d_p)
        t_feat = fourier_time_features(t_pad, self.Kt)
        d_feat = fourier_time_features(d_pad, self.Kd)
        x = torch.cat([ep_tok, t_feat, d_feat], dim=-1)
        lengths_tensor = torch.tensor(lengths, device=x.device, dtype=torch.long)
        packed = pack_padded_sequence(x, lengths_tensor.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, h_last = self.gru(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=Lmax)
        mask = torch.arange(Lmax, device=x.device).unsqueeze(0) < lengths_tensor.unsqueeze(1)
        mean = (out * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        last = h_last.squeeze(0)
        r = torch.cat([last, mean], dim=-1)
        z = self.proj(r)
        z = torch.nn.functional.normalize(z, p=2, dim=-1)
        return z, r

def collate_fn(batch, purpose_to_idx):
    p_lists, t_lists, d_lists, lens = [], [], [], []
    for seq in batch:
        p_idx = [purpose_to_idx[p] for p,_,_ in seq]
        t0 = [t for _,t,_ in seq]
        dd = [d for _,_,d in seq]
        p_lists.append(torch.tensor(p_idx, dtype=torch.long))
        t_lists.append(torch.tensor(t0, dtype=torch.float32))
        d_lists.append(torch.tensor(dd, dtype=torch.float32))
        lens.append(len(seq))
    p_pad = pad_sequence(p_lists, batch_first=True, padding_value=0)
    t_pad = pad_sequence(t_lists, batch_first=True, padding_value=0.0)
    d_pad = pad_sequence(d_lists, batch_first=True, padding_value=0.0)
    return p_pad, t_pad, d_pad, lens

def build_masks(purposes_df: pd.DataFrame, purposes_order: list):
    P = len(purposes_order)
    name_to_idx = {p:i for i,p in enumerate(purposes_order)}
    def parse_bool(x):
        if isinstance(x, (int, float)):
            return int(x) != 0
        if isinstance(x, str):
            x = x.strip().lower()
            if x in ("y","yes","true","t","1"):
                return True
            if x in ("n","no","false","f","0"):
                return False
        return False
    open_allowed = torch.zeros(P, dtype=torch.bool)
    close_allowed = torch.zeros(P, dtype=torch.bool)
    home_idx = name_to_idx.get("Home", None)
    if "can_open_close_day" in purposes_df.columns:
        for _, r in purposes_df.iterrows():
            p = r["purpose"]
            if p not in name_to_idx:
                continue
            allowed = parse_bool(r["can_open_close_day"]) 
            open_allowed[name_to_idx[p]] = allowed
            close_allowed[name_to_idx[p]] = allowed
    else:
        open_allowed[:] = True
        close_allowed[:] = True
    return {"open_allowed": open_allowed, "close_allowed": close_allowed, "home_idx": home_idx, "home_idx_end": home_idx}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activities_csv", type=str, default="/mnt/data/small_activities_homebound_wd.csv")
    ap.add_argument("--purposes_csv", type=str, default="/mnt/data/purposes.csv")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--outdir", type=str, default="./runs")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    set_seed(42)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    time_cfg = TimeConfig()
    basis_cfg = BasisConfig()
    quad_cfg = QuadratureConfig()
    pep_cfg = PurposeEmbeddingConfig()
    dec_cfg = DecoderConfig()

    acts = pd.read_csv(args.activities_csv)
    purp = pd.read_csv(args.purposes_csv)
    
    priors, purposes = derive_priors_from_activities(acts, purp, time_cfg.T_minutes, basis_cfg.K_time_prior)

    rows = []
    for p in purposes:
        pr = priors[p]
        rows.append(np.concatenate([pr.time_fourier, [pr.mu_t, pr.sigma_t, pr.mu_d, pr.sigma_d]]).astype("float32"))
    phi = np.stack(rows, axis=0)
    phi = (phi - phi.mean(0, keepdims=True)) / (phi.std(0, keepdims=True) + 1e-6)
    phi_t = torch.tensor(phi, dtype=torch.float32, device=args.device)

    pds = PurposeDistributionSpace(phi_t, d_p=pep_cfg.d_p, hidden=pep_cfg.hidden).to(args.device)
    pds.set_time_prior_K(basis_cfg.K_time_prior)
    e_p = pds()

    P = len(purposes)
    dec = TimeFieldDecoder(P=P, m_latent=dec_cfg.m_latent, d_p=pep_cfg.d_p, K_decoder_time=basis_cfg.K_decoder_time, alpha_prior=dec_cfg.alpha_prior).to(args.device)
    enc = TrajEncoderGRU(d_p=pep_cfg.d_p, K_time_token=4, K_dur_token=4, m_latent=dec_cfg.m_latent).to(args.device)

    params = list(pds.parameters()) + list(enc.parameters()) + list(dec.parameters())
    opt = optim.Adam(params, lr=args.lr)

    t_q, w_q = gauss_legendre_nodes(quad_cfg.Q_nodes, dtype=torch.float32, device=args.device)

    masks = build_masks(purp, purposes)
    masks_t = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k,v in masks.items()}

    ds = ScheduleDataset(args.activities_csv, time_cfg.T_minutes)
    tr_idx, va_idx = split_indices(len(ds), val_ratio=args.val_ratio, seed=42)
    tr_subset = torch.utils.data.Subset(ds, tr_idx)
    va_subset = torch.utils.data.Subset(ds, va_idx)

    purpose_to_idx = {p:i for i,p in enumerate(purposes)}
    # Create subsets
    tr_subset = torch.utils.data.Subset(ds, tr_idx)
    va_subset = torch.utils.data.Subset(ds, va_idx)

    def loader(subset, shuffle):
        return torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=lambda batch: collate_fn(batch, purpose_to_idx))
    dl_tr = loader(tr_subset, True)
    dl_va = loader(va_subset, False)

    history = {"epoch": [], "train_total": [], "val_total": [], "ce": [], "emd": [], "tv": []}
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    best_val_total = float('inf')

    mu_d = torch.tensor([priors[p].mu_d for p in purposes], dtype=torch.float32, device=args.device)
    sd_d = torch.tensor([priors[p].sigma_d for p in purposes], dtype=torch.float32, device=args.device)

    for epoch in range(1, args.epochs+1):
        enc.train()
        dec.train()
        pds.train()
        total_tr = 0.0
        for p_pad, t_pad, d_pad, lengths in dl_tr:
            e_p = pds()
            # lambda_log uses only buffers; detach to be explicit that it’s constant
            with torch.no_grad():
                loglam = pds.lambda_log(t_q)         # (P, Q)
            p_pad = p_pad.to(args.device); t_pad = t_pad.to(args.device); d_pad = d_pad.to(args.device)
            z, r = enc(p_pad, t_pad, d_pad, lengths, e_p)
            u = dec.utilities(z, e_p, t_q, loglam, masks=masks_t)
            q = dec.soft_assign(u)
            segs = []
            B, Lmax = p_pad.shape
            for b in range(B):
                L = lengths[b]
                seq = []
                for i in range(L):
                    pid = int(p_pad[b,i].item())
                    pstr = purposes[pid]
                    seq.append((pstr, float(t_pad[b,i].item()), float(d_pad[b,i].item())))
                segs.append(seq)
            y = rasterize_batch(segs, purpose_to_idx, t_q).to(args.device)

            ce = dec.ce_loss(q, y, w_q)
            emd = dec.emd1d_loss(q, y, w_q)
            tv = dec.tv_loss(q, w_q)
            durlen = dec.durlen_loss(q, mu_d, sd_d, w_q)

            total = dec_cfg.ce_weight*ce + dec_cfg.emd_weight*emd + dec_cfg.tv_weight*tv + dec_cfg.durlen_weight*durlen
            # Update model parameters
            opt.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total_tr += float(total.item())

        enc.eval()
        dec.eval()
        pds.eval()
        with torch.no_grad():
            e_p = pds()
            loglam = pds.lambda_log(t_q)
            total_va = 0.0
            ce_v = emd_v = tv_v = 0.0
            n_batches=0
            for p_pad, t_pad, d_pad, lengths in dl_va:
                p_pad = p_pad.to(args.device)
                t_pad = t_pad.to(args.device)
                d_pad = d_pad.to(args.device)
                z, r = enc(p_pad, t_pad, d_pad, lengths, e_p)
                u = dec.utilities(z, e_p, t_q, loglam, masks=masks_t)
                q = dec.soft_assign(u)
                segs = []
                B, Lmax = p_pad.shape
                for b in range(B):
                    L = lengths[b]
                    seq = []
                    for i in range(L):
                        pid = int(p_pad[b,i].item())
                        pstr = purposes[pid]
                        seq.append((pstr, float(t_pad[b,i].item()), float(d_pad[b,i].item())))
                    segs.append(seq)
                y = rasterize_batch(segs, purpose_to_idx, t_q).to(args.device)
                ce_l = dec.ce_loss(q, y, w_q)
                emd_l = dec.emd1d_loss(q, y, w_q)
                tv_l = dec.tv_loss(q, w_q)
                mu_d = torch.tensor([priors[p].mu_d for p in purposes], dtype=torch.float32, device=args.device)
                sd_d = torch.tensor([priors[p].sigma_d for p in purposes], dtype=torch.float32, device=args.device)
                total_l = DecoderConfig.ce_weight*ce_l + DecoderConfig.emd_weight*emd_l + DecoderConfig.tv_weight*tv_l + DecoderConfig.durlen_weight*dec.durlen_loss(q, mu_d, sd_d, w_q)
                total_va += float(total_l.item())
                ce_v += float(ce_l.item())
                emd_v += float(emd_l.item())
                tv_v += float(tv_l.item())
                n_batches += 1

        avg_tr = total_tr / max(len(dl_tr),1); avg_va = total_va / max(n_batches,1)
        if epoch % 500 == 0 or epoch == args.epochs:
            print(f"[{epoch:03d}] train={avg_tr:.4f}  val={avg_va:.4f}  (ce={ce_v/max(n_batches,1):.4f}, emd={emd_v/max(n_batches,1):.4f}, tv={tv_v/max(n_batches,1):.4f})")
        history["epoch"].append(epoch)
        history["train_total"].append(avg_tr)
        history["val_total"].append(avg_va)
        history["ce"].append(ce_v/max(n_batches,1))
        history["emd"].append(emd_v/max(n_batches,1))
        history["tv"].append(tv_v/max(n_batches,1))

        ckpt = {
            "epoch": epoch,
            "model_state": {"pds": pds.state_dict(), "enc": enc.state_dict(), "dec": dec.state_dict()},
            "priors": priors, "purposes": purposes, "purpose_to_idx": purpose_to_idx,
            "configs": {"time": vars(time_cfg:=TimeConfig()), "basis": vars(BasisConfig()), "quad": vars(QuadratureConfig()), "pep": vars(PurposeEmbeddingConfig()), "dec": vars(DecoderConfig())},
        }
        if avg_va < best_val_total:
            best_val_total = avg_va
            torch.save(ckpt, Path(args.outdir)/"ckpt_best.pt")
            if epoch > 500:
                print(f"Best validation total: {best_val_total:.4f} at epoch {epoch}")
        if epoch == args.epochs:
            torch.save(ckpt, Path(args.outdir)/"ckpt_final.pt")
            print(f"Final validation total: {avg_va:.4f} at epoch {epoch}")

    # Save history to CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(Path(args.outdir)/"history.csv", index=False)

if __name__ == "__main__":
    main()
