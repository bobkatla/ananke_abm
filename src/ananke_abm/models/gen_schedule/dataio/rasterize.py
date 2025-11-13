from __future__ import annotations
import pandas as pd
import numpy as np
import json
import os
import torch
from ananke_abm.models.gen_schedule.dataio.splits import read_n_split_data

PURPOSE_COL = "purpose"

def build_purpose_map(df: pd.DataFrame):
    uniq = sorted(df[PURPOSE_COL].unique().tolist())
    return {p:i for i,p in enumerate(uniq)}

def rasterize_person(person_df, purpose_map, grid_min: int, horizon_min: int = 1440):
    """
    Convert one person's list of activities into a fixed-length array (L = horizon_min / grid_min),
    ensuring:
      - every activity gets at least one bin,
      - activities appear in the correct order,
      - if multiple short activities would map to the same bin, later ones are pushed
        to the next free bin (so each gets its own bin),
      - home-bound behavior is maintained (sequences end with "Home" activities).

    Post-processing ensures home-bound behavior by shifting non-home segments when needed.
    
    person_df is assumed to be sorted in activity order (e.g. by stopno).
    """
    L = horizon_min // grid_min
    arr = np.zeros(L, dtype=np.int64)

    next_free_bin = 0  # the earliest bin we’re allowed to use for the next activity

    for _, r in person_df.iterrows():
        s = int(r["starttime"])
        d = int(r["total_duration"])
        p_idx = purpose_map[r["purpose"]]

        if d <= 0:
            # skip zero or negative durations defensively
            continue

        # nominal start bin from actual start time
        nominal_a = max(0, s) // grid_min

        # enforce monotonic progression so activities never go backwards in time
        a = max(nominal_a, next_free_bin)

        if a >= L:
            # no room left in the horizon; truncate the rest of the day
            break

        # desired number of bins from duration (at least 1)
        desired_bins = max(1, int(np.ceil(d / float(grid_min))))

        b = min(L, a + desired_bins)
        if b <= a:
            # extremely edge-case; but guard anyway
            b = min(L, a + 1)

        arr[a:b] = p_idx
        next_free_bin = b

    # Post-processing: ensure home-bound behavior
    arr = _ensure_homebound_postprocess(arr, purpose_map)
    
    return arr

def _ensure_homebound_postprocess(arr, purpose_map, buffer_size=1):
    """
    Post-processing to ensure home-bound behavior by adjusting activities that don't end at home.
    
    Algorithm:
    1. Check if the sequence ends with home
    2. If not, scan backwards to find consecutive non-home segments from the end
    3. Find a preceding home segment that's longer than buffer_size
    4. Shift the non-home segment backwards to create space for home at the end
    """
    home_idx = purpose_map.get("Home", 0)
    L = len(arr)
    
    # Check if already ends with home
    if arr[L-1] == home_idx:
        return arr
    
    # Find the end of the non-home segment (scanning backwards from end)
    non_home_end = L - 1
    non_home_start = non_home_end
    
    # Scan backwards to find the full consecutive non-home segment
    while non_home_start >= 0 and arr[non_home_start] != home_idx:
        non_home_start -= 1
    
    # non_home_start now points to the last home bin before the non-home segment
    # non_home segment is from (non_home_start + 1) to non_home_end (inclusive)
    
    if non_home_start < 0:
        # Edge case: no home activities found before the non-home segment
        # Force the last buffer_size bins to be home
        arr[L-buffer_size:] = home_idx
        return arr
    
    # Find the home segment that precedes the non-home segment
    home_segment_end = non_home_start
    home_segment_start = home_segment_end
    
    # Scan backwards to find the full consecutive home segment
    while home_segment_start >= 0 and arr[home_segment_start] == home_idx:
        home_segment_start -= 1
    
    home_segment_start += 1  # Adjust to the actual start of home segment
    home_segment_length = home_segment_end - home_segment_start + 1
    
    # Check if home segment is long enough to "borrow" buffer_size bins
    if home_segment_length > buffer_size:
        # Shift the non-home segment backwards by buffer_size
        non_home_length = non_home_end - non_home_start
        
        # Copy the non-home segment to its new position
        new_start = non_home_start + 1 - buffer_size
        new_end = new_start + non_home_length - 1
        
        # Ensure we don't go below index 0
        if new_start >= 0:
            arr[new_start:new_end+1] = arr[non_home_start+1:non_home_end+1]
            # Fill the freed space at the end with home
            arr[new_end+1:] = home_idx
        else:
            # Not enough space to shift, just force the end to be home
            arr[L-buffer_size:] = home_idx
    else:
        # Home segment is too short, just force the end to be home
        arr[L-buffer_size:] = home_idx
    
    return arr

def compute_empirical_tod(Y, P):
    N,L = Y.shape
    m = np.zeros((L,P), dtype=np.float64)
    for t in range(L):
        col = Y[:,t]
        for p in range(P):
            m[t,p] = np.mean(col==p)
    return m

def prepare_from_csv(
        csv_path: str,
        out_path: str,
        grid_min: int=5, 
        horizon_min: int=1440,
        val_frac: float=0.1,
        seed: int=42,
        ):
    df = pd.read_csv(csv_path)
    if "startime" in df.columns and "starttime" not in df.columns:
        df = df.rename(columns={"startime":"starttime"})
    purpose_map = build_purpose_map(df)
    inv_map = {v:k for k,v in purpose_map.items()}
    L = horizon_min // grid_min

    seqs, pers = [], []
    for pid, grp in df.groupby("persid"):
        grp = grp.sort_values("stopno")
        y = rasterize_person(grp, purpose_map, grid_min, horizon_min)
        seqs.append(y)
        pers.append(pid)
    Y = np.stack(seqs, axis=0)

    # make sure no all home all day
    home_all_day = (Y == purpose_map["Home"]).all(axis=1)
    count_home_all_day = int(home_all_day.sum())
    assert count_home_all_day == 0, f"{count_home_all_day} persons have all activities as Home"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, Y=Y.astype(np.int64))
    # split
    train_dataset, val_dataset = read_n_split_data(
        val_frac=val_frac,
        data_npz_path=out_path,
        seed=seed,
    )
    torch.save({
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
    }, out_path.replace(".npz", "_splits.pt"))
    meta = {"grid_min": grid_min, "horizon_min": horizon_min, "L": int(L),
            "purpose_map": purpose_map, "inv_purpose_map": inv_map, "N": int(Y.shape[0])}
    with open(out_path.replace(".npz", "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    m = compute_empirical_tod(Y, P=len(purpose_map))
    np.save(out_path.replace(".npz", "_tod.npy"), m)
    # save purpose map
    with open(out_path.replace(".npz", "_purpose_map.json"), "w", encoding="utf-8") as f:
        json.dump(purpose_map, f, indent=2)
    return out_path, meta
