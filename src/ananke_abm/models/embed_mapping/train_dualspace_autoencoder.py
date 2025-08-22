# -*- coding: utf-8 -*-
# Training script for DualSpaceAE (models/dualspace.py)
# - Robust CSV ingestion (long-format schedules)
# - Optional FiLM meta from purposes.csv with mixed types
# - Hours-consistent loss targets; masking consistent with nn.Transformer
"""
uv run src\ananke_abm\models\embed_mapping\train_dualspace_autoencoder.py --schedules-csv src\ananke_abm\models\embed_mapping\data\schedules.csv --purposes-csv src\ananke_abm\models\embed_mapping\data\purposes.csv --id-cols person_id,day --purpose-col purpose --start-col start_time --duration-col duration --k-max 12 --use-film --meta-one-hot --epochs 60 --batch-size 256
"""

import os
import argparse
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F  # <-- needed for class_weighted_ce

# ---- Model pieces (must exist in your repo) ----
from ananke_abm.models.embed_mapping.models.dualspace import (
    DualSpaceAE,
    DualSpaceConfig,
    PurposeEmbeddingWithFiLM,
    label_smoothing_ce,
    laplacian_regularizer,
)

# ----------------------------
# Utilities
# ----------------------------

def class_weighted_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    logits: [B,K,C], targets: [B,K] int64
    """
    B, K, C = logits.shape
    logits = logits.view(B*K, C)
    targets = targets.view(B*K)
    loss = F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        ignore_index=ignore_index,
        reduction="mean",
        label_smoothing=label_smoothing,
    )
    return loss

def wasserstein_cdf_loss(w_pred: torch.Tensor, w_gt: torch.Tensor) -> torch.Tensor:
    """
    Discrete W1 via CDF L1 on the K-simplex.
    w_pred, w_gt: [B,K], each row sums to ~1 (clipped inside)
    """
    eps = 1e-8
    w_pred = w_pred.clamp_min(eps)
    w_pred = w_pred / w_pred.sum(dim=-1, keepdim=True).clamp_min(eps)
    w_gt = w_gt.clamp_min(eps)
    w_gt = w_gt / w_gt.sum(dim=-1, keepdim=True).clamp_min(eps)
    cdf_pred = w_pred.cumsum(dim=-1)
    cdf_gt = w_gt.cumsum(dim=-1)
    return (cdf_pred - cdf_gt).abs().sum(dim=-1).mean()

def duration_entropy(w_pred: torch.Tensor) -> torch.Tensor:
    """
    H(w) = -sum w log w averaged over batch.
    """
    eps = 1e-8
    w = w_pred.clamp_min(eps)
    H = -(w * w.log()).sum(dim=-1).mean()
    return H

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def infer_device(cuda: bool) -> torch.device:
    if cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def to_hours(x: np.ndarray, time_unit: str) -> np.ndarray:
    if time_unit == "hours":
        return x.astype(np.float32)
    elif time_unit == "minutes":
        return (x / 60.0).astype(np.float32)
    elif time_unit == "days":
        return (x * 24.0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported --time-unit={time_unit}")

# ----------------------------
# Data prep
# ----------------------------

def build_vocab_from_schedules(df: pd.DataFrame, purpose_col: str) -> Dict[str, int]:
    """Build purpose->idx; PAD=0 reserved. No UNK (expect clean preprocessed)."""
    purposes = df[purpose_col].astype(str).unique().tolist()
    purposes = sorted(purposes)
    vocab = {"<PAD>": 0}
    for i, p in enumerate(purposes, start=1):
        vocab[p] = i
    return vocab

def encode_purpose_column(df: pd.DataFrame, purpose_col: str, vocab: Dict[str,int]) -> np.ndarray:
    # map; if unseen purpose sneaks in, map to PAD (0) rather than crashing
    return df[purpose_col].astype(str).map(vocab).fillna(0).astype(int).to_numpy()

def group_key_columns(df: pd.DataFrame, id_cols: List[str]) -> List[str]:
    """Return the subset of id_cols that are actually present in df."""
    return [c for c in id_cols if c in df.columns]

def build_sequences(
    df_long: pd.DataFrame,
    id_cols: List[str],
    purpose_col: str,
    start_col: str,
    dur_col: str,
    vocab: Dict[str,int],
    time_unit: str,
    k_max: int,
    sort_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Inputs: long-format schedules with one row per segment.
    Returns: padded arrays [N,K], lengths [N]
    """
    if sort_cols is None:
        sort_cols = [start_col]

    key_cols = group_key_columns(df_long, id_cols)
    if not key_cols:
        # fallback: treat entire file as one schedule id = 0..N by grouping rows that have explicit schedule_id,
        # otherwise assume each unique (person_id, day) exists; if really nothing, create synthetic key
        raise ValueError(
            "No grouping ID columns found. Please provide at least one present in your schedules CSV "
            "(e.g., --id-cols person_id day or --id-cols schedule_id)."
        )

    # Ensure correct dtypes
    df_long = df_long.copy()
    df_long[start_col] = pd.to_numeric(df_long[start_col], errors="coerce")
    df_long[dur_col] = pd.to_numeric(df_long[dur_col], errors="coerce")
    df_long = df_long.dropna(subset=[start_col, dur_col, purpose_col])

    # sort within group
    df_long = df_long.sort_values(key_cols + sort_cols)

    # group pointers
    groups = df_long.groupby(key_cols, sort=False)
    N = len(groups)
    K = k_max

    P = np.zeros((N, K), dtype=np.int64)         # purpose indices
    S = np.zeros((N, K), dtype=np.float32)       # start hours
    D = np.zeros((N, K), dtype=np.float32)       # duration hours
    L = np.zeros((N,), dtype=np.int64)           # lengths

    for gi, (_, g) in enumerate(groups):
        p = encode_purpose_column(g, purpose_col, vocab)
        s = to_hours(g[start_col].to_numpy(), time_unit)
        d = to_hours(g[dur_col].to_numpy(), time_unit)

        # Keep first K segments. If >K, we truncate (log a warning once)
        if len(p) > K:
            if gi == 0:
                print(f"[warn] sequence length {len(p)} exceeds k_max={K}; truncating. "
                      f"Consider increasing --k-max.")
            p = p[:K]
            s = s[:K]
            d = d[:K]

        L[gi] = len(p)
        P[gi, :len(p)] = p
        S[gi, :len(p)] = s
        D[gi, :len(p)] = d

    return P, S, D, L

class SchedulesDataset(Dataset):
    def __init__(self, P, S, D, L, pad_idx: int):
        super().__init__()
        self.P = P
        self.S = S
        self.D = D
        self.L = L
        self.pad_idx = pad_idx
        self.K = P.shape[1]

    def __len__(self):
        return self.P.shape[0]

    def __getitem__(self, i):
        return {
            "purpose_idx": torch.from_numpy(self.P[i]),
            "start": torch.from_numpy(self.S[i]),
            "duration": torch.from_numpy(self.D[i]),
            "length": int(self.L[i]),
        }

def collate_batch(batch: List[Dict], pad_idx: int) -> Dict[str, torch.Tensor]:
    P = torch.stack([b["purpose_idx"] for b in batch], dim=0).long()  # [B,K]
    S = torch.stack([b["start"] for b in batch], dim=0).float()
    D = torch.stack([b["duration"] for b in batch], dim=0).float()
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    B, K = P.shape
    # pad_mask=True for PAD positions
    arange = torch.arange(K).unsqueeze(0).expand(B, K)
    pad_mask = arange >= lengths.unsqueeze(1)
    # Ensure PAD tokens at padded positions
    P = P.masked_fill(pad_mask, pad_idx)
    S = S.masked_fill(pad_mask, 0.0)
    D = D.masked_fill(pad_mask, 0.0)

    return {
        "purpose_idx": P,
        "start": S,
        "duration": D,
        "pad_mask": pad_mask,
    }

# ----------------------------
# Purpose meta (FiLM) handling
# ----------------------------

def auto_build_meta_matrix(
    purposes_df: pd.DataFrame,
    vocab: Dict[str,int],
    purpose_key_cols: List[str],
    auto_one_hot: bool,
    explicit_num_cols: Optional[List[str]] = None,
) -> Tuple[Optional[torch.Tensor], Dict]:
    """
    Build [V, meta_dim] aligned to vocab indices.
    - We try to map each purpose string in vocab (excluding <PAD>) to a row in purposes_df.
    - If explicit_num_cols given and present -> use those (coerce to numeric).
    - Else: use all numeric columns + (optionally) one-hot encode remaining categoricals.
    Return meta_tensor or None if failure; and a dict with info to save in checkpoint.
    """
    # derive purpose name column to join on:
    # try common column names
    candidate_name_cols = ["purpose", "name", "purpose_name", "label", "activity", "activity_name"]
    name_col = None
    for c in candidate_name_cols:
        if c in purposes_df.columns:
            name_col = c
            break
    if name_col is None:
        # if user specified purpose_key_cols, pick the first that exists
        for c in purpose_key_cols:
            if c in purposes_df.columns:
                name_col = c
                break
    if name_col is None:
        print("[FiLM] Could not find a purpose name column; FiLM disabled.")
        return None, {}

    # numeric part
    df = purposes_df.copy()

    # Keep only the latest occurrence per name if duplicated
    df = df.drop_duplicates(subset=[name_col], keep="first")

    # Determine features
    feats = []
    if explicit_num_cols:
        feats = [c for c in explicit_num_cols if c in df.columns]
        if not feats:
            print("[FiLM] Provided --meta-cols not found; falling back to auto-detection.")
    if not feats:
        # All numeric columns except obvious keys
        drop_keys = set([name_col, "id", "purpose_id", "code"])
        num_cols = []
        for c in df.columns:
            if c in drop_keys:
                continue
            series = pd.to_numeric(df[c], errors="coerce")
            if series.notna().sum() > 0 and (series.notna().mean() > 0.5):
                # consider numeric if >=50% parsable
                num_cols.append(c)
        feats = num_cols

    X_num = None
    if feats:
        X_num = df[feats].apply(pd.to_numeric, errors="coerce")
    else:
        X_num = pd.DataFrame(index=df.index)

    X_cat = None
    if auto_one_hot:
        # Categorical columns = neither in feats nor name_col
        cand = [c for c in df.columns if c not in feats + [name_col]]
        cat_cols = []
        for c in cand:
            if df[c].dtype == "object" or str(df[c].dtype).startswith("category"):
                nunique = df[c].nunique(dropna=True)
                if 2 <= nunique <= 20:
                    cat_cols.append(c)
        if cat_cols:
            X_cat = pd.get_dummies(df[cat_cols].astype("category"), dummy_na=False).astype("float32")
        else:
            X_cat = pd.DataFrame(index=df.index)
    else:
        X_cat = pd.DataFrame(index=df.index)

    X = pd.concat([X_num.astype("float32"), X_cat], axis=1)
    if X.shape[1] == 0:
        print("[FiLM] No usable meta features found; FiLM disabled.")
        return None, {}

    # Standardize numeric columns only (cat one-hots are already 0/1)
    if X_num.shape[1] > 0:
        X_num = X_num.astype("float32")
    means = X_num.mean() if X_num.shape[1] > 0 else pd.Series(dtype=float)
    stds = X_num.std().replace(0, 1.0) if X_num.shape[1] > 0 else pd.Series(dtype=float)
    if X_num.shape[1] > 0:
        X.loc[:, X_num.columns] = ((X_num - means) / stds).astype("float32")

    # Build [V, D] aligned to vocab index
    V = len(vocab)
    D = X.shape[1]
    M = np.zeros((V, D), dtype=np.float32)
    missing = 0
    for p, idx in vocab.items():
        if p == "<PAD>":
            continue
        row = df.loc[df[name_col].astype(str) == str(p)]
        if row.empty:
            missing += 1
            continue
        xrow = X.loc[row.index[0]].to_numpy(dtype=np.float32, copy=True)
        M[idx] = xrow
    if missing > 0:
        print(f"[FiLM] Warning: {missing} purposes from schedules not found in purposes.csv; "
              "their FiLM vectors will be zeros.")

    meta_tensor = torch.from_numpy(M)
    info = {
        "name_col": name_col,
        "feature_cols": X.columns.tolist(),
        "numeric_means": means.to_dict(),
        "numeric_stds": stds.to_dict(),
        "one_hot_included": auto_one_hot,
    }
    return meta_tensor, info

# ----------------------------
# Training / Eval
# ----------------------------

def train_one_epoch(
    model: DualSpaceAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_idx: int,
    kl_beta: float,
    lambda_lap: float,
    lap_L: Optional[torch.Tensor],
    label_smoothing: float,
    class_weights: Optional[torch.Tensor],        # NEW
    tf_prob: float,                                # NEW
    time_jitter_mins: float,                       # NEW
    wass_weight: float,                            # NEW
    startl1_weight: float,                         # NEW
    duration_entropy_weight: float,                # NEW
):

    model.train()
    total = 0.0
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        for k in batch:
            batch[k] = batch[k].to(device)

        # coin flip per batch for teacher forcing
        teacher_forced = (torch.rand((), device=device).item() < tf_prob)

        # forward (requires updated DualSpaceAE.forward to accept these kwargs)
        out = model(
            batch,
            teacher_forced=teacher_forced,
            time_jitter_minutes=(time_jitter_mins if teacher_forced else 0.0),
        )

        logits       = out["purpose_logits"]      # [B,K,V]
        pred_dur_h   = out["durations"]           # [B,K]
        pred_start_h = out["starts"]              # [B,K]
        w_pred       = out["w_pred"]              # [B,K] simplex over K

        # masks
        mask = (~batch["pad_mask"]).float()  # [B,K]

        # ---- PURPOSE: class-weighted CE (uses your precomputed weights) ----
        cw = class_weights.to(device) if class_weights is not None else None
        loss_ce = class_weighted_ce(
            logits,
            batch["purpose_idx"],
            ignore_index=pad_idx,
            class_weights=cw,
            label_smoothing=label_smoothing,
        )

        # ---- DURATION: Wasserstein/CDF between predicted and GT simplex ----
        w_gt = batch["duration"] / (batch["duration"].sum(dim=-1, keepdim=True) + 1e-8)
        loss_wass = wasserstein_cdf_loss(w_pred, w_gt)

        # ---- STARTS: L1 in hours (keep, but lower weight via startl1_weight) ----
        loss_start = ((pred_start_h - batch["start"]).abs() * mask).sum() / (mask.sum() + 1e-8)

        # ---- Optional VAE KL ----
        kl = torch.tensor(0.0, device=device)
        if "mu" in out and "logvar" in out:
            mu, logvar = out["mu"], out["logvar"]
            kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

        # ---- Entropy penalty to encourage decisive slot allocations (- H) ----
        H_w = duration_entropy(w_pred)
        loss = (
            loss_ce
            + wass_weight * loss_wass
            + startl1_weight * loss_start
            + duration_entropy_weight * H_w
            + kl_beta * kl
        )

        # Laplacian reg on embedding table if provided
        if lambda_lap > 0.0 and lap_L is not None:
            loss = loss + laplacian_regularizer(model.encoder.purpose_embed.embed, lap_L, lambda_lap)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total += float(loss.detach().cpu())
        n += 1
    return total / max(n, 1)

@torch.no_grad()
def eval_one_epoch(
    model: DualSpaceAE,
    loader: DataLoader,
    device: torch.device,
    pad_idx: int,
    kl_beta: float,
    label_smoothing: float,
    class_weights: Optional[torch.Tensor],     # NEW
    wass_weight: float,                         # NEW
    startl1_weight: float,                      # NEW
    duration_entropy_weight: float,             # NEW
):

    model.eval()
    total = 0.0
    n = 0
    for batch in tqdm(loader, desc="valid", leave=False):
        for k in batch:
            batch[k] = batch[k].to(device)

        out = model(batch, teacher_forced=True, time_jitter_minutes=0.0)
        logits       = out["purpose_logits"]
        pred_dur_h   = out["durations"]
        pred_start_h = out["starts"]
        w_pred       = out["w_pred"]

        mask = (~batch["pad_mask"]).float()

        cw = class_weights.to(device) if class_weights is not None else None
        loss_ce = class_weighted_ce(
            logits,
            batch["purpose_idx"],
            ignore_index=pad_idx,
            class_weights=cw,
            label_smoothing=label_smoothing,
        )

        w_gt = batch["duration"] / (batch["duration"].sum(dim=-1, keepdim=True) + 1e-8)
        loss_wass = wasserstein_cdf_loss(w_pred, w_gt)
        loss_start = ((pred_start_h - batch["start"]).abs() * mask).sum() / (mask.sum() + 1e-8)

        kl = torch.tensor(0.0, device=device)
        if "mu" in out and "logvar" in out:
            mu, logvar = out["mu"], out["logvar"]
            kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

        H_w = duration_entropy(w_pred)
        loss = (
            loss_ce
            + wass_weight * loss_wass
            + startl1_weight * loss_start
            + duration_entropy_weight * H_w
            + kl_beta * kl
        )

        total += float(loss.detach().cpu())
        n += 1
    return total / max(n, 1)

# ----------------------------
# Main
# ----------------------------

def main(args):
    set_seed(args.seed)
    device = infer_device(args.cuda)

    # --- Load schedules ---
    sch = pd.read_csv(args.schedules_csv)
    # Basic presence checks
    for col in [args.purpose_col, args.start_col, args.duration_col]:
        if col not in sch.columns:
            raise ValueError(f"Column '{col}' not found in schedules CSV.")
    id_cols = [c.strip() for c in args.id_cols.split(",")] if args.id_cols else []
    # Build vocab from observed purposes
    vocab = build_vocab_from_schedules(sch, args.purpose_col)
    pad_idx = vocab["<PAD>"]
    print(f"[data] vocab size (incl PAD) = {len(vocab)}")

    # --- Build sequences ---
    P, S, D, L = build_sequences(
        sch,
        id_cols=id_cols,
        purpose_col=args.purpose_col,
        start_col=args.start_col,
        dur_col=args.duration_col,
        vocab=vocab,
        time_unit=args.time_unit,
        k_max=args.k_max,
        sort_cols=[args.start_col],
    )
    print(f"[data] sequences: N={P.shape[0]}, K={P.shape[1]}, avg_len={L.mean():.2f}")

    # --- Dataset / split ---
    full_ds = SchedulesDataset(P, S, D, L, pad_idx=pad_idx)
    if args.val_frac > 0:
        n_val = int(round(len(full_ds) * args.val_frac))
    else:
        n_val = 0
    n_train = len(full_ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g) if n_val > 0 else (full_ds, None)

    collate = lambda batch: collate_batch(batch, pad_idx=pad_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate, drop_last=False)
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                collate_fn=collate, drop_last=False)
        
    # Estimate class frequencies from the training data
    from collections import Counter
    purpose_counts = Counter()
    for batch in train_loader:
        # batch["purpose_idx"]: [B,K]
        ids = batch["purpose_idx"].view(-1).tolist()
        for t in ids:
            if t != pad_idx:
                purpose_counts[t] += 1

    # Build weight vector (C,) on device later
    C = len(vocab)
    weights = torch.ones(C, dtype=torch.float32)
    # inverse sqrt frequency (clip to avoid inf for rare classes)
    for cls_id, cnt in purpose_counts.items():
        weights[cls_id] = 1.0 / (cnt ** 0.5)
    # set PAD weight to 0 explicitly if pad_idx exists
    weights[pad_idx] = 0.0
    class_weights = weights / (weights.max().clamp(min=1e-8))  # normalize for stability
    class_weights = class_weights.to(device)

    # --- Purpose meta (FiLM) ---
    meta_tensor = None
    meta_info = {}
    if args.use_film and os.path.exists(args.purposes_csv):
        pur = pd.read_csv(args.purposes_csv)
        explicit_cols = [c.strip() for c in args.meta_cols.split(",")] if args.meta_cols else None
        meta_tensor, meta_info = auto_build_meta_matrix(
            pur,
            vocab=vocab,
            purpose_key_cols=[args.purpose_key_col] if args.purpose_key_col else [],
            auto_one_hot=args.meta_one_hot,
            explicit_num_cols=explicit_cols,
        )
        if meta_tensor is not None:
            print(f"[FiLM] meta shape = {tuple(meta_tensor.shape)}")
        else:
            print("[FiLM] disabled (no usable meta).")
    else:
        if args.use_film:
            print(f"[FiLM] purposes_csv not found at {args.purposes_csv}; FiLM disabled.")

    # --- Config & model ---
    cfg = DualSpaceConfig(
        n_purposes=len(vocab),
        pad_idx=pad_idx,
        k_max=args.k_max,
        d_purpose=args.d_purpose,
        d_time=args.d_time,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        d_z=args.d_z,
        n_time_harmonics=args.fourier_harmonics,
        label_smoothing=args.label_smoothing,
        duration_temp=args.duration_temp,
        use_vae=args.use_vae,
        kl_beta=args.kl_beta,
        lambda_lap=args.lambda_lap,
        lambda_meta_probe=0.0,
        day_hours=args.day_hours,
    )

    purpose_embed = PurposeEmbeddingWithFiLM(
        n_purposes=len(vocab),
        emb_dim=cfg.d_purpose,
        pad_idx=pad_idx,
        meta=meta_tensor.to(device) if meta_tensor is not None else None
    )

    model = DualSpaceAE(cfg, purpose_embed, n_purposes=len(vocab), pad_idx=pad_idx).to(device)

    # Optional Laplacian regularizer matrix L (identity by default -> no effect)
    lap_L = None
    if args.lambda_lap > 0.0 and os.path.exists(args.laplacian_npy):
        try:
            L_mat = np.load(args.laplacian_npy).astype(np.float32)
            if L_mat.shape[0] != len(vocab) or L_mat.shape[1] != len(vocab):
                print(f"[lap] Provided Laplacian shape {L_mat.shape} != ({len(vocab)}, {len(vocab)}); ignoring.")
            else:
                lap_L = torch.from_numpy(L_mat).to(device)
                print("[lap] Using Laplacian regularization.")
        except Exception as e:
            print(f"[lap] Failed to load Laplacian: {e}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # NEW: LR scheduler that reduces LR when val loss plateaus
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,    
        patience=5,   # e.g., 5
        threshold=1e-3, # e.g., 1e-3
        cooldown=0,   # e.g., 0
        min_lr=1e-6,          # e.g., 1e-6
    )

    # --- Training loop ---
    best_val = float("inf")
    os.makedirs(args.out_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        # --- scheduled teacher forcing prob for this epoch ---
        E = args.epochs
        anneal_steps = max(1, int(args.tf_anneal_frac * E))
        if epoch >= anneal_steps:
            tf_prob = args.tf_prob_end
        else:
            tf_prob = args.tf_prob_start + (args.tf_prob_end - args.tf_prob_start) * ((epoch - 1) / max(1, anneal_steps - 1))

        tr_loss = train_one_epoch(
            model, train_loader, optimizer, device, pad_idx,
            kl_beta=args.kl_beta, lambda_lap=args.lambda_lap, lap_L=lap_L,
            label_smoothing=args.label_smoothing,
            class_weights=class_weights,                 # NEW
            tf_prob=tf_prob,                             # NEW
            time_jitter_mins=args.time_jitter_mins,      # NEW
            wass_weight=args.wass_weight,                # NEW
            startl1_weight=args.startl1_weight,          # NEW
            duration_entropy_weight=args.duration_entropy_weight,  # NEW
        )

        log = {"epoch": epoch, "train_loss": tr_loss}
        if val_loader is not None:
            val_loss = eval_one_epoch(
                model, val_loader, device, pad_idx,
                kl_beta=args.kl_beta,
                label_smoothing=args.label_smoothing,
                class_weights=class_weights,                 # NEW
                wass_weight=args.wass_weight,                # NEW
                startl1_weight=args.startl1_weight,          # NEW
                duration_entropy_weight=args.duration_entropy_weight,  # NEW
            )

            log["val_loss"] = val_loss
            print(f"[epoch {epoch}] train={tr_loss:.4f}  val={val_loss:.4f}")
            # after printing and any checkpointing for best model
            scheduler.step(val_loss)   # NEW: drive LR drops off validation loss

            if val_loss < best_val:
                best_val = val_loss
                save_path = os.path.join(args.out_dir, "best.pt")
                print("Saving best model to ", save_path)
                torch.save({
                    "model_state": model.state_dict(),
                    "cfg": vars(cfg),
                    "vocab": vocab,                       # dict[str -> int]
                    "idx_to_purpose": {i: p for p, i in vocab.items()},  # dict[int -> str]
                    "meta_info": meta_info,
                    "args": vars(args),
                }, save_path)
        else:
            print(f"[epoch {epoch}] train={tr_loss:.4f}")
            # still save last
            save_path = os.path.join(args.out_dir, "last.pt")
            torch.save({
                "model_state": model.state_dict(),
                "cfg": vars(cfg),
                "vocab": vocab,                       # dict[str -> int]
                "idx_to_purpose": {i: p for p, i in vocab.items()},  # dict[int -> str]
                "meta_info": meta_info,
                "args": vars(args),
            }, save_path)

    # Save final checkpoint
    save_path = os.path.join(args.out_dir, "final.pt")
    torch.save({
        "model_state": model.state_dict(),
        "cfg": vars(cfg),
        "vocab": vocab,                       # dict[str -> int]
        "idx_to_purpose": {i: p for p, i in vocab.items()},  # dict[int -> str]
        "meta_info": meta_info,
        "args": vars(args),
    }, save_path)
    print(f"[done] saved to {args.out_dir}")

# ----------------------------
# CLI
# ----------------------------

def cli():
    p = argparse.ArgumentParser("Train DualSpaceAE")

    # Data
    p.add_argument("--schedules-csv", type=str, required=True)
    p.add_argument("--purposes-csv", type=str, default="")
    p.add_argument("--id-cols", type=str, default="person_id,day", help="comma-separated keys to group a day schedule")
    p.add_argument("--purpose-col", type=str, default="purpose")
    p.add_argument("--start-col", type=str, default="start")
    p.add_argument("--duration-col", type=str, default="duration")
    p.add_argument("--time-unit", type=str, default="hours", choices=["hours", "minutes", "days"])
    p.add_argument("--k-max", type=int, default=10)

    # FiLM meta
    p.add_argument("--use-film", action="store_true")
    p.add_argument("--purpose-key-col", type=str, default="", help="column in purposes.csv to match purpose names (optional)")
    p.add_argument("--meta-cols", type=str, default="", help="comma-separated explicit numeric meta columns to use (optional)")
    p.add_argument("--meta-one-hot", action="store_true", help="auto one-hot small-cardinality categoricals")

    # Model
    p.add_argument("--d-purpose", type=int, default=64)
    p.add_argument("--d-time", type=int, default=16)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--d-z", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--fourier-harmonics", type=int, default=4)
    p.add_argument("--duration-temp", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--day-hours", type=float, default=24.0)

    # VAE / regularizers
    p.add_argument("--use-vae", action="store_true")
    p.add_argument("--kl-beta", type=float, default=0.0)
    p.add_argument("--lambda-lap", type=float, default=0.0)
    p.add_argument("--laplacian-npy", type=str, default="")

    # Loss weights
    p.add_argument("--w-dur", type=float, default=1.0)
    p.add_argument("--w-start", type=float, default=1.0)

    # Optim
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=0)

    # Train/val
    p.add_argument("--val-frac", type=float, default=0.1)

    # IO
    p.add_argument("--out-dir", type=str, default="checkpoints_dualspace")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--cuda", action="store_true")

    p.add_argument("--wass-weight", type=float, default=1.0)
    p.add_argument("--startl1-weight", type=float, default=0.25)
    p.add_argument("--duration-entropy-weight", type=float, default=0.01)
    p.add_argument("--tf-prob-start", type=float, default=1.0)
    p.add_argument("--tf-prob-end", type=float, default=0.3)
    p.add_argument("--tf-anneal-frac", type=float, default=0.7, help="fraction of total epochs to reach end prob")
    p.add_argument("--time-jitter-mins", type=float, default=15.0)

    args = p.parse_args()
    main(args)

if __name__ == "__main__":
    cli()
