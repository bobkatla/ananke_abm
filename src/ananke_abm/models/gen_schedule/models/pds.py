import click
import os
import json
import numpy as np


def compute_time_of_day_marginal(Y_int, num_purposes):
    """
    Y_int: (N, T) int64 purpose ids.
    Returns m_tod: (P, T) where m_tod[p,t] = Pr(y_t == p) across people.
    """
    N, T = Y_int.shape
    m_tod = np.zeros((num_purposes, T), dtype=np.float64)
    # count occurrences per (p,t)
    for p in range(num_purposes):
        m_tod[p] = np.mean(Y_int == p, axis=0)  # (T,)
    return m_tod  # (P,T)


def compute_start_rate(Y_int, num_purposes):
    """
    Returns start_rate[p,t] = Pr(a new segment of purpose p starts at t),
    averaged across people.
    A new segment of p starts at t if:
      - t==0 and y[0]==p
      - or t>0 and y[t]==p and y[t-1]!=p
    Shape: (P,T)
    """
    N, T = Y_int.shape
    start_rate = np.zeros((num_purposes, T), dtype=np.float64)

    # boolean array of where a new-segment start happens for each person/time
    # We'll compute starts_any[b,t,p] implicitly without materializing full (N,T,P)
    for p in range(num_purposes):
        # mask for purpose p
        is_p = (Y_int == p)  # (N,T) bool
        starts = np.zeros((N, T), dtype=bool)
        # t == 0
        starts[:, 0] = is_p[:, 0]
        # t > 0
        starts[:, 1:] = is_p[:, 1:] & (~is_p[:, :-1])
        # probability across people
        start_rate[p] = starts.mean(axis=0)

    return start_rate  # (P,T)


def compute_presence_rate(Y_int, num_purposes):
    """
    presence_rate[p] = Pr(person ever does purpose p in their day).
    Shape: (P,)
    """
    N, T = Y_int.shape
    presence_rate = np.zeros((num_purposes,), dtype=np.float64)
    for p in range(num_purposes):
        any_p = np.any(Y_int == p, axis=1)  # (N,)
        presence_rate[p] = any_p.mean()
    return presence_rate  # (P,)


def summarize_first_start_minutes(Y_int, num_purposes, grid_min):
    """
    For each purpose p:
      - find the first time bin t where that purpose appears in each person's day
      - convert to minutes = t * grid_min
      - take mean/std across people who ever had that purpose
    Returns:
      start_mean_min[p], start_std_min[p]
    We'll also compute last occurrence as proxy for "end time".
    """
    N, T = Y_int.shape
    start_mean_min = np.full((num_purposes,), np.nan, dtype=np.float64)
    start_std_min = np.full((num_purposes,), np.nan, dtype=np.float64)
    end_mean_min = np.full((num_purposes,), np.nan, dtype=np.float64)
    end_std_min = np.full((num_purposes,), np.nan, dtype=np.float64)

    for p in range(num_purposes):
        # first occurrence index per person (or -1 if never)
        first_idx = np.argmax(Y_int == p, axis=1)  # gives 0 if never; we need to fix that
        has_p = np.any(Y_int == p, axis=1)        # (N,)
        # correct first_idx for people who never had p
        first_idx = np.where(has_p, first_idx, -1)

        # last occurrence index per person
        # trick: scan from right
        rev_idx = np.argmax(np.flip(Y_int == p, axis=1), axis=1)  # dist from end
        last_idx = (T - 1) - rev_idx
        last_idx = np.where(has_p, last_idx, -1)

        # collect valid ones
        valid_first = first_idx[first_idx >= 0]
        valid_last = last_idx[last_idx >= 0]

        if valid_first.size > 0:
            first_min = valid_first * grid_min
            start_mean_min[p] = first_min.mean()
            start_std_min[p] = first_min.std(ddof=0)

        if valid_last.size > 0:
            last_min = valid_last * grid_min
            end_mean_min[p] = last_min.mean()
            end_std_min[p] = last_min.std(ddof=0)

    return start_mean_min, start_std_min, end_mean_min, end_std_min


@click.command("compute-pds")
@click.option("--grid", type=click.Path(exists=True), required=True,
              help="Path to prepared grid npz (same format used for training, containing Y).")
@click.option("--out", type=click.Path(), required=True,
              help="Output prefix (we will write <out>_pds.npz).")
@click.option("--grid-min", type=int, required=True,
              help="Minutes per bin (e.g. 10). We pass this explicitly for now.")
@click.option("--purpose-json", type=click.Path(exists=True), default=None,
              help="Optional path to a JSON with purpose_map {purpose_name: idx}. "
                   "If not provided, we'll infer purpose_names as ['p0','p1',...].")
def compute_pds_cli(grid, out, grid_min, purpose_json):
    """
    Compute Purpose Distribution Space (PDS) stats from rasterized schedule data.
    Saves:
      <out>_pds.npz  (for the model)
      <out>_pds_summary.json  (human-readable summary)
    Prints a short JSON summary to stdout.
    """
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # ---- load Y ----
    data_obj = np.load(grid)
    if "Y" not in data_obj:
        raise RuntimeError("Expected 'Y' in grid npz.")
    Y_int = data_obj["Y"].astype(np.int64)  # (N,T)
    N, T = Y_int.shape

    # ---- infer purpose names / count ----
    if purpose_json and os.path.exists(purpose_json):
        with open(purpose_json, "r", encoding="utf-8") as f:
            pm = json.load(f)  # {purpose_name: idx}
        # invert to get idx->name in order 0..P-1
        inverse = {idx: name for name, idx in pm.items()}
        num_purposes = len(inverse)
        purpose_names_ordered = [inverse[i] for i in range(num_purposes)]
    else:
        # fallback: infer num_purposes from max label in Y
        num_purposes = int(Y_int.max()) + 1
        purpose_names_ordered = [f"p{p}" for p in range(num_purposes)]

    # ---- compute stats ----
    m_tod = compute_time_of_day_marginal(Y_int, num_purposes)        # (P,T)
    start_rate = compute_start_rate(Y_int, num_purposes)             # (P,T)
    presence_rate = compute_presence_rate(Y_int, num_purposes)       # (P,)
    start_mean_min, start_std_min, end_mean_min, end_std_min = summarize_first_start_minutes(
        Y_int, num_purposes, grid_min
    )

    # ---- save npz ----
    npz_path = f"{out}_pds.npz"
    np.savez_compressed(
        npz_path,
        m_tod=m_tod.astype(np.float32),
        start_rate=start_rate.astype(np.float32),
        presence_rate=presence_rate.astype(np.float32),
        start_mean_min=start_mean_min.astype(np.float32),
        start_std_min=start_std_min.astype(np.float32),
        end_mean_min=end_mean_min.astype(np.float32),
        end_std_min=end_std_min.astype(np.float32),
        purpose_names_ordered=np.array(purpose_names_ordered, dtype=object),
        grid_min=np.int32(grid_min),
        T=np.int32(T),
        N_persons=np.int32(N),
    )

    # ---- print short summary ----
    # We'll summarize per-purpose presence and first start time.
    summary = []
    for p, pname in enumerate(purpose_names_ordered):
        summary.append({
            "purpose": pname,
            "presence_rate": float(presence_rate[p]),
            "start_mean_min": float(start_mean_min[p]) if not np.isnan(start_mean_min[p]) else None,
            "start_std_min": float(start_std_min[p]) if not np.isnan(start_std_min[p]) else None,
        })

    overall_summary = {
        "N_persons": int(N),
        "T": int(T),
        "grid_min": int(grid_min),
        "purposes": summary,
        "npz_path": npz_path,
    }

    click.echo(json.dumps(overall_summary, indent=2))

    with open(f"{out}_pds_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)
