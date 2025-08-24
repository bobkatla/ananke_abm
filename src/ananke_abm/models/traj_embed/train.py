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
from ananke_abm.models.traj_embed.model.utils_bases import gauss_legendre_nodes, fourier_time_features, build_time_binned_transition_costs
from ananke_abm.models.traj_embed.model.rasterize import rasterize_batch, rasterize_from_padded
from ananke_abm.models.traj_embed.model.decoder_timefield import TimeFieldDecoder

def purpose_ce_weights_from_seed_by_time(activities_csv: str, purposes: list[str], eps: float = 1e-6):
    df = pd.read_csv(activities_csv)
    mp = {p:i for i,p in enumerate(purposes)}
    P = len(purposes)
    dur = np.zeros(P, dtype=np.float64)
    for p, g in df.groupby("purpose"):
        if p in mp:
            dur[mp[p]] = g["total_duration"].sum()
    freq = dur / max(1.0, dur.sum())      # time share for each purpose
    w = 1.0 / np.maximum(freq, eps)       # inverse frequency
    w = w / w.mean()                      # normalize around 1.0 (keeps loss scale stable)
    return torch.tensor(w, dtype=torch.float32)

# q: (B,P,Q), y: (B,P,Q), w_q: (Q,), home_idx: int
def nonhome_mass(x, w_q, home_idx):
    # x may be q or y
    xh = x[:, home_idx, :]                             # (B,Q)
    return ((1.0 - xh) * w_q.view(1, -1)).sum(-1)      # (B,)

def nonhome_mass_mse(q, y, w_q, home_idx):
    pred = nonhome_mass(q, w_q, home_idx)
    gt   = nonhome_mass(y, w_q, home_idx)
    return torch.mean((pred - gt)**2)

def last_home_start_times(p_pad, t_pad, lengths, home_idx):
    B, Lmax = p_pad.shape
    tau = t_pad.new_full((B,), 1.0)  # default: end of day (safety)
    for b in range(B):
        L = lengths[b]
        idx = (p_pad[b, :L] == home_idx).nonzero(as_tuple=False)
        if idx.numel():
            tau[b] = t_pad[b, idx[-1, 0]]
    return tau  # (B,)

def terminal_home_loss(q, t_q, w_q, tau, home_idx):
    # q: (B,P,Q); t_q,w_q: (Q,); tau:(B,)
    qh = q[:, home_idx, :]                          # (B,Q)
    after = (t_q.view(1, -1) >= tau.view(-1, 1)).float()
    # Encourage "stay Home" after final-Home start => penalize (1 - q_home)
    return ((1.0 - qh) * after * w_q.view(1,-1)).sum(dim=-1).mean()

def bigram_prior_loss_time(q, t_q, C_t):  # q:(B,P,Q), t_q:(Q,), C_t:(Tbin,P,P)
    nbins, P, _ = C_t.shape
    Q = q.shape[-1]
    if Q < 2: return q.new_tensor(0.0)
    # time at midpoints between nodes
    t_mid = 0.5 * (t_q[:-1] + t_q[1:])                 # (Q-1,)
    bin_idx = torch.clamp((t_mid * nbins).long(), 0, nbins-1)  # (Q-1,)
    C_stack = C_t.index_select(0, bin_idx)             # (Q-1,P,P)

    # q at consecutive steps
    qt = q[:, :, :-1].transpose(1, 2)                  # (B,Q-1,P)
    qn = q[:, :,  1:].transpose(1, 2)                  # (B,Q-1,P)

    # For each step s: (qt[b,s] @ C_stack[s]) · qn[b,s]
    tmp = torch.einsum('bsp,spq->bsq', qt, C_stack)    # (B,Q-1,P)
    expC = (tmp * qn).sum(dim=2)                       # (B,Q-1)
    return expC.mean()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    t_q, w_q = gauss_legendre_nodes(quad_cfg.Q_nodes_train, dtype=torch.float32, device=args.device)
    Phi_q = fourier_time_features(t_q, basis_cfg.K_decoder_time)  # keep on device
    loglam_q = pds.lambda_log(t_q)  # constant; uses only buffers in PDS

    masks = build_masks(purp, purposes)
    masks_t = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k,v in masks.items()}
    home_idx = masks.get("home_idx", None)
    if home_idx is not None:
        with torch.no_grad():
            dec.alpha_per_p.data.fill_(dec_cfg.alpha_prior)
            dec.alpha_per_p.data[home_idx] *= dec_cfg.alpha_home_factor

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
    C_t = build_time_binned_transition_costs(args.activities_csv, purposes, nbins=24, device=args.device)
    ce_w = purpose_ce_weights_from_seed_by_time(args.activities_csv, purposes).to(args.device)

    for epoch in range(1, args.epochs+1):
        enc.train()
        dec.train()
        pds.train()
        total_tr = 0.0
        for p_pad, t_pad, d_pad, lengths in dl_tr:
            e_p = pds()
            # lambda_log uses only buffers; detach to be explicit that it’s constant
            # with torch.no_grad():
            #     loglam = pds.lambda_log(t_q)         # (P, Q)
            p_pad = p_pad.to(args.device); t_pad = t_pad.to(args.device); d_pad = d_pad.to(args.device)
            z, r = enc(p_pad, t_pad, d_pad, lengths, e_p)
            u = dec.utilities(z, e_p, t_q, loglam_q, masks=masks_t, Phi=Phi_q)
            q = dec.soft_assign(u)
            tau = last_home_start_times(p_pad, t_pad, lengths, masks_t["home_idx"])
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
            # y = rasterize_batch(segs, purpose_to_idx, t_q).to(args.device)
            y = rasterize_from_padded(p_pad, t_pad, d_pad, lengths, P, t_q).to(args.device)

            ce = dec.ce_loss(q, y, w_q, class_weights=ce_w)
            emd = dec.emd1d_loss(q, y, w_q)
            tv = dec.tv_loss(q, w_q)
            durlen = dec.durlen_loss(q, mu_d, sd_d, w_q)
            home_loss = terminal_home_loss(q, t_q, w_q, tau, masks_t["home_idx"])
            bigram_prior_loss = bigram_prior_loss_time(q, t_q, C_t)
            nonhome_mse = nonhome_mass_mse(q, y, w_q, masks_t["home_idx"])

            total = dec_cfg.ce_weight*ce \
                + dec_cfg.emd_weight*emd \
                + dec_cfg.tv_weight*tv \
                + dec_cfg.durlen_weight*durlen \
                + dec_cfg.home_weight*home_loss \
                + dec_cfg.bigram_prior_weight*bigram_prior_loss \
                + dec_cfg.nonhome_mse_weight*nonhome_mse
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
            # loglam = pds.lambda_log(t_q)
            total_va = 0.0
            ce_v = emd_v = tv_v = durlen_v = home_v = bigram_prior_v = 0.0
            n_batches=0
            
            for p_pad, t_pad, d_pad, lengths in dl_va:
                p_pad = p_pad.to(args.device)
                t_pad = t_pad.to(args.device)
                d_pad = d_pad.to(args.device)
                z, r = enc(p_pad, t_pad, d_pad, lengths, e_p)
                u = dec.utilities(z, e_p, t_q, loglam_q, masks=masks_t, Phi=Phi_q)
                q = dec.soft_assign(u)
                tau = last_home_start_times(p_pad, t_pad, lengths, masks_t["home_idx"])
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
                # y = rasterize_batch(segs, purpose_to_idx, t_q).to(args.device)
                y = rasterize_from_padded(p_pad, t_pad, d_pad, lengths, P, t_q).to(args.device)
                ce_l = dec.ce_loss(q, y, w_q)
                emd_l = dec.emd1d_loss(q, y, w_q)
                tv_l = dec.tv_loss(q, w_q)
                durlen_l = dec.durlen_loss(q, mu_d, sd_d, w_q)
                home_l = terminal_home_loss(q, t_q, w_q, tau, masks_t["home_idx"])
                bigram_prior_l = bigram_prior_loss_time(q, t_q, C_t)
                nonhome_mse_l = nonhome_mass_mse(q, y, w_q, masks_t["home_idx"])
                total_l = dec_cfg.ce_weight*ce_l \
                    + dec_cfg.emd_weight*emd_l \
                    + dec_cfg.tv_weight*tv_l \
                    + dec_cfg.durlen_weight*durlen_l \
                    + dec_cfg.home_weight*home_l \
                    + dec_cfg.bigram_prior_weight*bigram_prior_l \
                    + dec_cfg.nonhome_mse_weight*nonhome_mse_l
                total_va += float(total_l.item())
                ce_v += float(ce_l.item())
                emd_v += float(emd_l.item())
                tv_v += float(tv_l.item())
                durlen_v += float(durlen_l.item())
                home_v += float(home_l.item())
                bigram_prior_v += float(bigram_prior_l.item())
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
            if epoch > 0:
                print(f"Best validation total: {best_val_total:.4f} at epoch {epoch}")
        if epoch == args.epochs:
            torch.save(ckpt, Path(args.outdir)/"ckpt_final.pt")
            print(f"Final validation total: {avg_va:.4f} at epoch {epoch}")

    # Save history to CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(Path(args.outdir)/"history.csv", index=False)

if __name__ == "__main__":
    main()
