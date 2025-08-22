
# validate_reconstruction.py
# Load the trained model and verify reconstruction validity + write sample outputs.

from typing import Dict, List, Tuple
import argparse
import numpy as np
import pandas as pd
import torch

from ananke_abm.models.embed_mapping.models.dualspace import DualSpaceAE, DualSpaceConfig, PurposeEmbeddingWithFiLM

def _auto_build_meta_matrix_for_val(
    purposes_csv: str,
    vocab: dict,
    expected_dim: int,
    meta_info: dict | None = None,
) -> torch.Tensor | None:
    """
    Build [V, expected_dim] meta aligned to vocab for validation.
    If meta_info is provided (saved by trainer), use exactly those features and z-scales.
    Otherwise, auto-detect numeric + small-cardinality categoricals, z-score numerics,
    then slice or zero-pad to expected_dim.
    """
    try:
        df = pd.read_csv(purposes_csv)
    except Exception as e:
        print(f"[validate] Could not read purposes_csv: {e}")
        return None

    # Try to pick purpose name column
    name_col = None
    for c in ["purpose", "name", "purpose_name", "activity", "activity_name", "label"]:
        if c in df.columns:
            name_col = c
            break
    if name_col is None:
        print("[validate] No purpose name column found in purposes_csv; proceeding without FiLM.")
        return None
    df = df.drop_duplicates(subset=[name_col], keep="first")

    # If we have training-time meta_info, reconstruct exactly
    if meta_info:
        feat_cols = meta_info.get("feature_cols", [])
        if not feat_cols:
            print("[validate] meta_info present but empty feature_cols; falling back to auto.")
        else:
            # Rebuild numeric + one-hot exactly as used in training
            X = pd.DataFrame(index=df.index)
            miss = []
            for c in feat_cols:
                if c in df.columns:
                    col = df[c]
                    if col.dtype == "object" or str(col.dtype).startswith("category"):
                        # already one-hot in training; re-one-hot now
                        one_hot = pd.get_dummies(col.astype("category"), prefix=c, dummy_na=False)
                        X = pd.concat([X, one_hot], axis=1)
                    else:
                        X[c] = pd.to_numeric(col, errors="coerce")
                else:
                    # Feature missing now; add zeros column
                    X[c] = 0.0
                    miss.append(c)
            if miss:
                print(f"[validate] Warning: missing feature cols now (filled zeros): {miss}")

            # Apply saved z-scales to numeric columns (by exact column name)
            means = meta_info.get("numeric_means", {}) or {}
            stds  = meta_info.get("numeric_stds", {}) or {}
            for c in X.columns:
                if c in means and c in stds:
                    sd = stds[c] if stds[c] not in (0.0, None) else 1.0
                    X[c] = (X[c].astype(float) - float(means[c])) / float(sd)

            # Align to vocab
            V = len(vocab)
            D = X.shape[1]
            M = np.zeros((V, D), dtype=np.float32)
            index_by_name = {str(df.loc[i, name_col]): i for i in df.index}
            for p, j in vocab.items():
                if p == "<PAD>":
                    continue
                i = index_by_name.get(str(p), None)
                if i is not None:
                    M[j, :] = X.iloc[i].to_numpy(dtype=np.float32, copy=True)
            # Finally adjust to expected_dim
            if D > expected_dim:
                M = M[:, :expected_dim]
            elif D < expected_dim:
                M = np.pad(M, ((0,0),(0, expected_dim - D)), mode="constant")
            return torch.from_numpy(M)

    # Auto-detect path (no meta_info)
    # Numeric columns (>=50% parsable), z-score
    drop_keys = {name_col, "id", "purpose_id", "code"}
    num_cols = []
    for c in df.columns:
        if c in drop_keys:
            continue
        series = pd.to_numeric(df[c], errors="coerce")
        if series.notna().mean() >= 0.5:
            num_cols.append(c)
    X_num = df[num_cols].apply(pd.to_numeric, errors="coerce") if num_cols else pd.DataFrame(index=df.index)
    means = X_num.mean() if X_num.shape[1] > 0 else pd.Series(dtype=float)
    stds  = X_num.std().replace(0, 1.0) if X_num.shape[1] > 0 else pd.Series(dtype=float)
    if X_num.shape[1] > 0:
        X_num = (X_num - means) / stds

    # Small-cardinality categoricals: 2..20 unique
    cat_cols = []
    for c in df.columns:
        if c in drop_keys or c in num_cols:
            continue
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category"):
            u = df[c].nunique(dropna=True)
            if 2 <= u <= 20:
                cat_cols.append(c)
    X_cat = pd.get_dummies(df[cat_cols].astype("category"), dummy_na=False) if cat_cols else pd.DataFrame(index=df.index)

    X = pd.concat([X_num, X_cat], axis=1)
    if X.shape[1] == 0:
        print("[validate] No usable meta features auto-detected; proceeding without FiLM.")
        return None

    V = len(vocab)
    D = X.shape[1]
    M = np.zeros((V, D), dtype=np.float32)
    index_by_name = {str(df.loc[i, name_col]): i for i in df.index}
    for p, j in vocab.items():
        if p == "<PAD>":
            continue
        i = index_by_name.get(str(p), None)
        if i is not None:
            M[j, :] = X.iloc[i].to_numpy(dtype=np.float32, copy=True)

    # Adjust to expected_dim
    if D > expected_dim:
        M = M[:, :expected_dim]
    elif D < expected_dim:
        M = np.pad(M, ((0,0),(0, expected_dim - D)), mode="constant")
    return torch.from_numpy(M)


def load_checkpoint(ckpt_path: str, purposes_csv: str, use_film: bool = True, device: str = "cpu"):
    ck = torch.load(ckpt_path, map_location=device)

    # Support both checkpoint formats:
    if "cfg" in ck and "vocab" in ck:
        # Newer trainer format
        cfg_dict = ck["cfg"]
        vocab_list = ck["vocab"]  # list of purpose strings with <PAD> at index 0
        purpose_to_idx = {p: i for i, p in enumerate(vocab_list)} if isinstance(vocab_list, list) else ck["vocab"]
        meta_info = ck.get("meta_info", None)
    else:
        raise ValueError("Invalid checkpoint format")

    cfg = DualSpaceConfig(**cfg_dict)

    # Determine expected FiLM input size from state_dict (if present)
    state = ck["model_state"]
    expected_meta_dim = 0
    for k, v in state.items():
        if k.endswith("purpose_embed.film.net.0.weight"):
            expected_meta_dim = v.shape[1]  # Linear(in=expected_meta_dim, out=hidden)
            break

    # Build purpose embedding (with meta if expected and requested)
    meta_tensor = None
    if use_film and expected_meta_dim > 0:
        meta_tensor = _auto_build_meta_matrix_for_val(
            purposes_csv=purposes_csv,
            vocab=purpose_to_idx if isinstance(purpose_to_idx, dict) else {p: i for i, p in enumerate(purpose_to_idx)},
            expected_dim=expected_meta_dim,
            meta_info=meta_info,
        )
        if meta_tensor is None:
            print("[validate] FiLM requested but no meta could be built; falling back to no-FiLM.")
            expected_meta_dim = 0

    purpose_embed = PurposeEmbeddingWithFiLM(
        n_purposes=len(purpose_to_idx) if isinstance(purpose_to_idx, dict) else len(purpose_to_idx),
        emb_dim=cfg.d_purpose,
        pad_idx=cfg.pad_idx,
        meta=meta_tensor.to(device) if (meta_tensor is not None) else None
    )
    model = DualSpaceAE(cfg, purpose_embed,
                        n_purposes=len(purpose_to_idx) if isinstance(purpose_to_idx, dict) else len(purpose_to_idx),
                        pad_idx=cfg.pad_idx).to(device)

    # If meta dims mismatch for any reason, allow non-strict load as a last resort
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f"[validate] strict load failed ({e}); retrying with strict=False.")
        model.load_state_dict(state, strict=False)

    model.eval()
    idx_to_purpose = {i: p for p, i in purpose_to_idx.items()} if isinstance(purpose_to_idx, dict) else {i: p for i, p in enumerate(purpose_to_idx)}
    return model, cfg, purpose_to_idx, idx_to_purpose

def sequences_from_csv(schedules_csv: str, k_max: int, purpose_to_idx: Dict[str,int]) -> List[Dict]:
    df = pd.read_csv(schedules_csv)

    # Normalize column names
    if "day" not in df.columns:
        df["day"] = 1
    persid_col = [x for x in ["persid", "person_id"] if x in df.columns][0]
    day_col = "day"
    seq_col = [x for x in ["stopno", "seq", "seq_id"] if x in df.columns][0]
    start_col = [x for x in ["startime", "start", "start_time"] if x in df.columns][0]
    duration_col = [x for x in ["total_duration", "duration", "duration_hours"] if x in df.columns][0]
    purpose_col = [x for x in ["purpose", "purpose_id"] if x in df.columns][0]

    required = {persid_col, day_col, seq_col, start_col, duration_col, purpose_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in schedules_csv: {missing}")

    df = df.sort_values([persid_col, day_col, seq_col])
    seqs = []
    for (pid, day), g in df.groupby([persid_col, day_col], sort=False):
        g = g.sort_values(start_col)
        L = len(g)
        p = [purpose_to_idx.get(x, 0) for x in g[purpose_col].tolist()]
        s = g[start_col].astype(float).tolist()
        d = g[duration_col].astype(float).tolist()
        if L > k_max:
            # merge tail durations into the last slot
            extra = sum(d[k_max-1:])
            d = d[:k_max-1] + [d[k_max-1] + extra]
            p = p[:k_max]
            s = s[:k_max]
            L = k_max
        # pad
        p = p + [0]*(k_max - L)
        s = s + [0.0]*(k_max - L)
        d = d + [0.0]*(k_max - L)
        seqs.append({"person_id": pid, "day": day, "L": L, "purpose_idx": p, "start": s, "duration": d})
    return seqs


# --- in decode_and_validate, ensure tensors live on the same device as the model ---
def decode_and_validate(model, cfg, seqs, idx_to_purpose, out_csv_path: str, n_examples: int = 20, day_hours: int = 24):
    rows = []
    valid_count = 0
    device = next(model.parameters()).device  # infer device from model
    for i, g in enumerate(seqs[:n_examples]):
        with torch.no_grad():
            purpose_idx = torch.tensor([g["purpose_idx"]], dtype=torch.long, device=device)
            start = torch.tensor([g["start"]], dtype=torch.float32, device=device)
            duration = torch.tensor([g["duration"]], dtype=torch.float32, device=device)
            L = g["L"]
            pad_mask = (torch.arange(cfg.k_max, device=device).unsqueeze(0) >= torch.tensor([L], device=device).unsqueeze(1))
            pad_mask = pad_mask.to(torch.bool)

            batch = {"purpose_idx": purpose_idx, "start": start, "duration": duration, "pad_mask": pad_mask}
            out = model(batch)

            logits = out["purpose_logits"][0]            # (K,C)
            pred_idx = logits.argmax(dim=-1).detach().cpu().numpy()
            pred_purposes = [idx_to_purpose[int(i)] for i in pred_idx]
            pred_d = out["durations"][0].detach().cpu().numpy()
            pred_s = out["starts"][0].detach().cpu().numpy()

            # Post-process: merge adjacent identical purposes
            pred_purposes, pred_s, pred_d = merge_adjacent(pred_purposes, pred_s, pred_d)
            is_valid = valid_schedule_check(pred_s, pred_d, day_hours=day_hours)
            valid_count += int(is_valid)

            # Ground truth (trim to L)
            gt_idx = g["purpose_idx"][:L]
            gt_purposes = [idx_to_purpose[int(i)] for i in gt_idx]
            gt_s = np.array(g["start"][:L], dtype=float)
            gt_d = np.array(g["duration"][:L], dtype=float)

            rows.append({
                "person_id": g["person_id"],
                "day": g["day"],
                "valid_pred": bool(is_valid),
                "pred_purposes": "|".join(pred_purposes),
                "pred_starts": "|".join([f"{x:.2f}" for x in pred_s.tolist()]),
                "pred_durations": "|".join([f"{x:.2f}" for x in pred_d.tolist()]),
                "gt_purposes": "|".join(gt_purposes),
                "gt_starts": "|".join([f"{x:.2f}" for x in gt_s.tolist()]),
                "gt_durations": "|".join([f"{x:.2f}" for x in gt_d.tolist()]),
            })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv_path, index=False)
    return valid_count, len(rows), out_csv_path


def valid_schedule_check(starts: np.ndarray, durations: np.ndarray, atol=1e-4, day_hours=24) -> bool:
    # start[0] == 0, sum durations == 24, non-overlap
    if not (abs(starts[0]) < atol): return False
    if not (abs(durations.sum() - day_hours) < 1e-3): return False
    ends = starts + durations
    if np.any(starts[1:] < ends[:-1] - 1e-4): return False
    if ends[-1] > day_hours + 1e-3: return False
    return True


def merge_adjacent(purposes: List[str], starts: np.ndarray, durations: np.ndarray) -> Tuple[List[str], np.ndarray, np.ndarray]:
    if len(purposes) == 0:
        return purposes, starts, durations
    new_p, new_s, new_d = [purposes[0]], [starts[0]], [durations[0]]
    for i in range(1, len(purposes)):
        if purposes[i] == new_p[-1]:
            new_d[-1] += durations[i]
        else:
            new_p.append(purposes[i]); new_s.append(starts[i]); new_d.append(durations[i])
    return new_p, np.array(new_s), np.array(new_d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="/mnt/data/dualspace_project/out/dualspace_ae.pt")
    ap.add_argument("--purposes_csv", type=str, default="/mnt/data/purposes.csv")
    ap.add_argument("--schedules_csv", type=str, default="/mnt/data/schedules.csv")
    ap.add_argument("--use_film", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--n_examples", type=int, default=20)
    ap.add_argument("--out_csv", type=str, default="/mnt/data/dualspace_project/out/reconstruction_samples.csv")
    ap.add_argument("--day_hours", type=int, default=24)
    args = ap.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    model, cfg, p2i, i2p = load_checkpoint(args.ckpt, args.purposes_csv, use_film=args.use_film, device=device)
    seqs = sequences_from_csv(args.schedules_csv, cfg.k_max, p2i)
    valid_count, total, path = decode_and_validate(model, cfg, seqs, i2p, args.out_csv, n_examples=args.n_examples, day_hours=args.day_hours)

    print(f"Decoded {total} examples. Valid full-day schedules: {valid_count}/{total}.")
    print(f"Wrote sample reconstructions to: {path}")


if __name__ == "__main__":
    main()
